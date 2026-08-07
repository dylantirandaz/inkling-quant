"""Offline contracts for matched Inkling quality and performance measurement.

This module only parses and cross-checks checked-in records. It does not start
Modal, load a model, or execute llama.cpp.
"""

from __future__ import annotations

import hashlib
import json
import struct
import wave
import zlib
from collections import Counter
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Final, Literal, TypeAlias

import yaml
from pydantic import (
    ConfigDict,
    Field,
    StrictFloat,
    ValidationError,
    field_validator,
    model_validator,
)
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

from inkling_quant_lab.config import StrictFrozenModel
from inkling_quant_lab.exceptions import ConfigurationError
from inkling_quant_lab.gguf.inkling_matched import (
    InklingMatchedCellBundle,
    MatchedRuntimeConfig,
    load_matched_cell_bundle,
)
from inkling_quant_lab.gguf.inkling_measurement_control import (
    MeasurementLlamaBenchWorkloadIdentity,
    MeasurementServerWorkloadIdentity,
)
from inkling_quant_lab.security import sensitive_literal_path

MEASUREMENT_CONFIG_RELATIVE_PATH: Final = (
    "configs/experiments/inkling_q3_k_m_measurement_modal.yaml"
)
DIAGNOSTIC_DATASET_RELATIVE_PATH: Final = "configs/experiments/inkling_quality_diagnostic_v1.jsonl"
CORPUS_REFERENCE_RELATIVE_PATH: Final = (
    "configs/experiments/inkling_wikitext2_raw_test_reference.json"
)
CORPUS_MATERIALIZER_RELATIVE_PATH: Final = "scripts/materialize_inkling_measurement_corpus.py"
MATERIALIZED_CORPUS_PATH: Final = "/opt/inkling-measurement-data/wikitext-2-raw-v1/wiki.test.raw"
MEASUREMENT_MEDIA_MARKER: Final = "<__media_iql_smoke_v1__>"
CORPUS_REFERENCE_HASH_DOMAIN: Final = b"inkling-wikitext2-raw-test-reference-v2\0"
MEASUREMENT_PROTOCOL_HASH_DOMAIN: Final = b"inkling-measurement-protocol-identity-v1\0"
MEASUREMENT_WORKLOAD_HASH_DOMAIN: Final = b"inkling-measurement-workload-identity-v1\0"
MEASUREMENT_CHAT_SYSTEM_TOKEN: Final = "<|message_system|>"
MEASUREMENT_CHAT_USER_TOKEN: Final = "<|message_user|>"
MEASUREMENT_CHAT_MODEL_TOKEN: Final = "<|message_model|>"
MEASUREMENT_CHAT_TEXT_TOKEN: Final = "<|content_text|>"
MEASUREMENT_CHAT_END_MESSAGE_TOKEN: Final = "<|end_message|>"
MEASUREMENT_CHAT_EFFORT_TEXT: Final = "Thinking effort level: 0"

_SUITES: Final = (
    "text",
    "math",
    "code",
    "multilingual",
    "instruction",
    "vision",
    "audio",
    "post_training",
)
_PLANNED_STAGES: Final = (
    "verify_references",
    "verify_cuda_preflight",
    "stage_and_rehash_bf16",
    "measure_bf16_quality",
    "measure_bf16_performance",
    "release_bf16",
    "stage_and_rehash_q3",
    "measure_q3_quality",
    "measure_q3_performance",
    "release_q3",
    "compare_and_publish",
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _canonical_relative_path(value: str, *, label: str) -> str:
    if not value or "\\" in value or "\x00" in value:
        raise ValueError(f"{label} must be a canonical relative POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise ValueError(f"{label} must be a canonical relative POSIX path")
    return value


def _canonical_absolute_path(value: str, *, label: str) -> str:
    if (
        not value
        or "\\" in value
        or "\x00" in value
        or not value.startswith("/")
        or value.startswith("//")
    ):
        raise ValueError(f"{label} must be a canonical absolute POSIX path")
    path = PurePosixPath(value)
    if (
        value == "/"
        or not path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise ValueError(f"{label} must be a canonical absolute POSIX path")
    return value


def _project_file(project_root: Path, relative_path: str) -> Path:
    canonical = _canonical_relative_path(relative_path, label="project reference")
    candidate = (project_root / Path(canonical)).resolve()
    if not candidate.is_relative_to(project_root):
        raise ConfigurationError(
            "Measurement project reference resolves outside the project root",
            component="inkling_measurement_bundle",
            details={"relative_path": relative_path},
        )
    return candidate


def _reject_non_finite(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _strict_json(raw: str) -> object:
    return json.loads(
        raw,
        object_pairs_hook=_reject_duplicate_json_pairs,
        parse_constant=_reject_non_finite,
    )


def _runtime_text(value: object, *, label: str) -> str:
    """Return runtime text without allowing static literal narrowing to erase checks."""

    if type(value) is not str:
        raise TypeError(f"{label} must be a string")
    return value


class _DuplicateKeyRejectingSafeLoader(yaml.SafeLoader):
    def construct_mapping(
        self,
        node: MappingNode,
        deep: bool = False,
    ) -> dict[Any, Any]:
        if not isinstance(node, MappingNode):
            raise ConstructorError(
                None,
                None,
                f"expected a mapping node, but found {node.id}",
                node.start_mark,
            )
        self.flatten_mapping(node)
        result: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, Hashable):
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found unhashable key",
                    key_node.start_mark,
                )
            if key in result:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            result[key] = self.construct_object(value_node, deep=deep)
        return result


class FileIdentity(StrictFrozenModel):
    """One checked-in file bound by canonical path, byte hash, and size."""

    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(gt=0)

    @field_validator("path")
    @classmethod
    def canonical_path(cls, value: str) -> str:
        return _canonical_relative_path(value, label="file identity path")


class RecordIdentity(FileIdentity):
    """A checked-in self-hashed reference record."""

    reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CorpusReference(StrictFrozenModel):
    """Exact WikiText-2 raw test corpus provenance and byte identity."""

    schema_version: Literal["inkling-wikitext2-raw-test-reference-v2"]
    dataset_id: Literal["Salesforce/wikitext"]
    repository_id: Literal["ggml-org/ci"]
    repository_revision: Literal["927b3642933080f1b0e811e2f916e14c292992f9"]
    archive_url: Literal[
        "https://huggingface.co/datasets/ggml-org/ci/resolve/"
        "927b3642933080f1b0e811e2f916e14c292992f9/wikitext-2-raw-v1.zip"
    ]
    archive_sha256: Literal["ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11"]
    archive_size_bytes: Literal[4721645]
    archive_member: Literal["wikitext-2-raw/wiki.test.raw"]
    extraction: Literal["read_exact_zip_member_bytes_without_normalization"]
    split: Literal["test"]
    materializer_path: Literal["scripts/materialize_inkling_measurement_corpus.py"]
    materializer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    materialized_path: Literal["/opt/inkling-measurement-data/wikitext-2-raw-v1/wiki.test.raw"]
    corpus_sha256: Literal["173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08"]
    corpus_size_bytes: Literal[1290590]
    license: Literal["CC-BY-SA-3.0-and-GFDL"]
    reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("archive_member", "materializer_path")
    @classmethod
    def canonical_relative_paths(cls, value: str) -> str:
        return _canonical_relative_path(value, label="corpus reference path")

    @field_validator("materialized_path")
    @classmethod
    def canonical_materialized_path(cls, value: str) -> str:
        return _canonical_absolute_path(value, label="materialized corpus path")

    def canonical_payload_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"reference_sha256"})

    def computed_reference_sha256(self) -> str:
        payload = _canonical_json(self.canonical_payload_dict()).encode("utf-8")
        return hashlib.sha256(CORPUS_REFERENCE_HASH_DOMAIN + payload).hexdigest()

    def canonical_json(self) -> str:
        return _canonical_json(self.model_dump(mode="json"))

    @model_validator(mode="after")
    def exact_self_hash(self) -> CorpusReference:
        if self.reference_sha256 != self.computed_reference_sha256():
            raise ValueError("corpus reference self-hash does not match its payload")
        return self


class ChoiceScorer(StrictFrozenModel):
    kind: Literal["choice"]
    choices: tuple[str, str, str, str]
    expected: Literal["A", "B", "C", "D"]
    normalization: Literal["strip_outer_whitespace_then_single_ascii_choice"] = (
        "strip_outer_whitespace_then_single_ascii_choice"
    )

    @field_validator("choices")
    @classmethod
    def distinct_choices(cls, value: tuple[str, str, str, str]) -> tuple[str, str, str, str]:
        if any(not item.strip() for item in value) or len(set(value)) != 4:
            raise ValueError("choice scorer requires four distinct non-empty choices")
        return value


class IntegerScorer(StrictFrozenModel):
    kind: Literal["integer"]
    expected: int
    normalization: Literal["strip_outer_whitespace_then_single_base10_integer"] = (
        "strip_outer_whitespace_then_single_base10_integer"
    )


class ExactTextScorer(StrictFrozenModel):
    kind: Literal["exact_text"]
    expected: str = Field(min_length=1)
    case_sensitive: Literal[True] = True
    normalization: Literal["strip_outer_whitespace"] = "strip_outer_whitespace"


JsonScalar = str | int | float | bool | None


class JsonExactScorer(StrictFrozenModel):
    kind: Literal["json_exact"]
    expected: dict[str, JsonScalar]
    normalization: Literal["parse_then_canonical_json"] = "parse_then_canonical_json"

    @field_validator("expected")
    @classmethod
    def nonempty_object(cls, value: dict[str, JsonScalar]) -> dict[str, JsonScalar]:
        if not value:
            raise ValueError("json_exact expected object must not be empty")
        return value


DiagnosticScorer = Annotated[
    ChoiceScorer | IntegerScorer | ExactTextScorer | JsonExactScorer,
    Field(discriminator="kind"),
]


Rgb = tuple[int, int, int]


class SyntheticImageFixture(StrictFrozenModel):
    kind: Literal["image"]
    fixture_id: str = Field(pattern=r"^image_[a-z0-9_]+_v1$")
    generator: Literal["inkling_rgb8_png_v1"]
    width: Literal[32]
    height: Literal[32]
    algorithm: Literal[
        "solid",
        "checkerboard_4px",
        "vertical_split",
        "horizontal_split",
        "quadrants",
        "vertical_bands_4",
        "horizontal_bands_4",
        "center_square_16px",
    ]
    colors_rgb: tuple[Rgb, ...]

    @model_validator(mode="after")
    def exact_color_arity(self) -> SyntheticImageFixture:
        required = {
            "solid": 1,
            "checkerboard_4px": 2,
            "vertical_split": 2,
            "horizontal_split": 2,
            "quadrants": 4,
            "vertical_bands_4": 4,
            "horizontal_bands_4": 4,
            "center_square_16px": 2,
        }[self.algorithm]
        if len(self.colors_rgb) != required:
            raise ValueError(f"{self.algorithm} requires exactly {required} RGB colors")
        if any(channel < 0 or channel > 255 for color in self.colors_rgb for channel in color):
            raise ValueError("RGB channels must be integers in [0, 255]")
        return self


class SyntheticAudioFixture(StrictFrozenModel):
    kind: Literal["audio"]
    fixture_id: str = Field(pattern=r"^audio_[a-z0-9_]+_v1$")
    generator: Literal["inkling_pcm_s16le_wav_v1"]
    sample_rate_hz: Literal[16000]
    channels: Literal[1]
    duration_samples: Literal[8000]
    algorithm: Literal["silence", "square_tone", "pulse_train"]
    frequency_hz: int | None = Field(default=None, gt=0)
    amplitude: int = Field(ge=0, le=32767)
    pulse_count: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def exact_parameters(self) -> SyntheticAudioFixture:
        if self.algorithm == "silence":
            if self.frequency_hz is not None or self.pulse_count is not None or self.amplitude != 0:
                raise ValueError("silence requires no frequency or pulses and zero amplitude")
        elif self.algorithm == "square_tone":
            if self.frequency_hz is None or self.pulse_count is not None or self.amplitude == 0:
                raise ValueError("square_tone requires frequency and nonzero amplitude only")
            if self.sample_rate_hz % (2 * self.frequency_hz) != 0:
                raise ValueError("square_tone frequency must have an exact integer half-period")
        elif (
            self.frequency_hz is not None
            or self.pulse_count is None
            or self.amplitude == 0
            or self.duration_samples % (2 * self.pulse_count) != 0
        ):
            raise ValueError("pulse_train requires an exact integer on/off period")
        return self


SyntheticFixture = Annotated[
    SyntheticImageFixture | SyntheticAudioFixture,
    Field(discriminator="kind"),
]


def build_diagnostic_fixture_bytes(
    fixture: SyntheticImageFixture | SyntheticAudioFixture | None,
) -> bytes | None:
    """Build the exact reviewed synthetic media fixture, if one is configured."""

    if fixture is None:
        return None
    if fixture.kind == "image":
        colors = fixture.colors_rgb

        def pixel(x: int, y: int) -> Rgb:
            if fixture.algorithm == "solid":
                return colors[0]
            if fixture.algorithm == "checkerboard_4px":
                return colors[((x // 4) + (y // 4)) % 2]
            if fixture.algorithm == "vertical_split":
                return colors[int(x >= 16)]
            if fixture.algorithm == "horizontal_split":
                return colors[int(y >= 16)]
            if fixture.algorithm == "quadrants":
                return colors[int(x >= 16) + 2 * int(y >= 16)]
            if fixture.algorithm == "vertical_bands_4":
                return colors[min(x // 8, 3)]
            if fixture.algorithm == "horizontal_bands_4":
                return colors[min(y // 8, 3)]
            if fixture.algorithm == "center_square_16px":
                return colors[int(8 <= x < 24 and 8 <= y < 24)]
            raise ValueError(f"unsupported image fixture algorithm: {fixture.algorithm}")

        scanlines = b"".join(
            b"\x00" + bytes(channel for x in range(fixture.width) for channel in pixel(x, y))
            for y in range(fixture.height)
        )

        def png_chunk(kind: bytes, data: bytes) -> bytes:
            return (
                struct.pack(">I", len(data))
                + kind
                + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
            )

        return (
            b"\x89PNG\r\n\x1a\n"
            + png_chunk(
                b"IHDR",
                struct.pack(
                    ">IIBBBBB",
                    fixture.width,
                    fixture.height,
                    8,
                    2,
                    0,
                    0,
                    0,
                ),
            )
            + png_chunk(b"IDAT", zlib.compress(scanlines, level=9))
            + png_chunk(b"IEND", b"")
        )

    samples: list[int]
    if fixture.algorithm == "silence":
        samples = [0] * fixture.duration_samples
    elif fixture.algorithm == "square_tone":
        if fixture.frequency_hz is None:
            raise ValueError("square-tone fixture lacks its validated frequency")
        half_period = fixture.sample_rate_hz // (2 * fixture.frequency_hz)
        samples = [
            fixture.amplitude if (index // half_period) % 2 == 0 else -fixture.amplitude
            for index in range(fixture.duration_samples)
        ]
    elif fixture.algorithm == "pulse_train":
        if fixture.pulse_count is None:
            raise ValueError("pulse-train fixture lacks its validated pulse count")
        half_period = fixture.duration_samples // (2 * fixture.pulse_count)
        samples = [
            fixture.amplitude if (index // half_period) % 2 == 0 else 0
            for index in range(fixture.duration_samples)
        ]
    else:
        raise ValueError(f"unsupported audio fixture algorithm: {fixture.algorithm}")

    output = BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(fixture.channels)
        stream.setsampwidth(2)
        stream.setframerate(fixture.sample_rate_hz)
        stream.writeframes(b"".join(struct.pack("<h", value) for value in samples))
    return output.getvalue()


MeasurementPromptInterface: TypeAlias = Literal[
    "raw_instruction_then_lf_then_item_prompt",
    "chat_template_system_effort_none_then_user_then_generation_prompt",
]


@dataclass(frozen=True)
class MeasurementDiagnosticPromptRender:
    """The exact text a diagnostic request sends for one prompt interface.

    ``prompt_text`` carries the interface-rendered prompt without media, and
    ``prompt_string`` adds the media marker where llama.cpp substitutes the
    projected media.  The two are equal for a text-only item.
    """

    prompt_text: str
    prompt_string: str


def render_measurement_diagnostic_prompt(
    *,
    prompt_template: str,
    item_prompt: str,
    prompt_interface: MeasurementPromptInterface,
    has_media: bool,
) -> MeasurementDiagnosticPromptRender:
    """Render one diagnostic prompt for the configured server prompt interface.

    The chat interface reproduces the official Inkling template for a system
    instruction, the reasoning-effort system turn, an optional media user turn,
    the item user turn, and the generation prompt.  llama.cpp inserts
    ``<|content_image|>`` or ``<|content_audio_input|>`` and its closing token at
    the media marker, so the marker stands alone inside its user block.
    """

    match prompt_interface:
        case "raw_instruction_then_lf_then_item_prompt":
            prompt_text = f"{prompt_template}\n{item_prompt}"
            prompt_string = (
                f"{MEASUREMENT_MEDIA_MARKER}\n{prompt_text}" if has_media else prompt_text
            )
        case "chat_template_system_effort_none_then_user_then_generation_prompt":
            system_turns = (
                f"{MEASUREMENT_CHAT_SYSTEM_TOKEN}{MEASUREMENT_CHAT_TEXT_TOKEN}"
                f"{prompt_template}{MEASUREMENT_CHAT_END_MESSAGE_TOKEN}"
                f"{MEASUREMENT_CHAT_SYSTEM_TOKEN}{MEASUREMENT_CHAT_TEXT_TOKEN}"
                f"{MEASUREMENT_CHAT_EFFORT_TEXT}{MEASUREMENT_CHAT_END_MESSAGE_TOKEN}"
            )
            item_turn = (
                f"{MEASUREMENT_CHAT_USER_TOKEN}{MEASUREMENT_CHAT_TEXT_TOKEN}"
                f"{item_prompt}{MEASUREMENT_CHAT_END_MESSAGE_TOKEN}"
            )
            prompt_text = f"{system_turns}{item_turn}{MEASUREMENT_CHAT_MODEL_TOKEN}"
            media_turn = (
                f"{MEASUREMENT_CHAT_USER_TOKEN}{MEASUREMENT_MEDIA_MARKER}"
                f"{MEASUREMENT_CHAT_END_MESSAGE_TOKEN}"
            )
            prompt_string = (
                f"{system_turns}{media_turn}{item_turn}{MEASUREMENT_CHAT_MODEL_TOKEN}"
                if has_media
                else prompt_text
            )
    return MeasurementDiagnosticPromptRender(
        prompt_text=prompt_text,
        prompt_string=prompt_string,
    )


class DiagnosticItem(StrictFrozenModel):
    """One deterministic, machine-scored quality item."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal["inkling-quality-diagnostic-item-v1"]
    item_id: str = Field(pattern=r"^[a-z_]+_[0-9]{2}$")
    suite: Literal[
        "text",
        "math",
        "code",
        "multilingual",
        "instruction",
        "vision",
        "audio",
        "post_training",
    ]
    modality: Literal["text", "image", "audio"]
    prompt: str = Field(min_length=1)
    seed: Literal[42]
    temperature: StrictFloat
    max_new_tokens: int = Field(ge=1, le=64)
    fixture: SyntheticFixture | None = None
    scorer: DiagnosticScorer

    @field_validator("temperature")
    @classmethod
    def exact_temperature(cls, value: float) -> float:
        if value != 0.0:
            raise ValueError("diagnostic temperature must be exactly 0.0")
        return value

    @model_validator(mode="after")
    def modality_matches_suite(self) -> DiagnosticItem:
        expected_modality = (
            "image" if self.suite == "vision" else ("audio" if self.suite == "audio" else "text")
        )
        if self.modality != expected_modality:
            raise ValueError(f"{self.suite} items must use {expected_modality} modality")
        if (self.fixture is None) != (self.modality == "text"):
            raise ValueError("only multimodal items may have a synthetic fixture")
        if self.fixture is not None and self.fixture.kind != self.modality:
            raise ValueError("fixture kind must match item modality")
        return self


class QualityConfig(StrictFrozenModel):
    diagnostic: FileIdentity
    diagnostic_item_count: Literal[64]
    diagnostic_repetitions: Literal[1]
    corpus_reference: RecordIdentity
    perplexity_context_tokens: Literal[512]
    perplexity_chunks: Literal[64]
    perplexity_scored_tokens: Literal[16320]
    printed_perplexity_absolute_tolerance: StrictFloat
    prompt_template: Literal[
        "Answer the task directly and emit only the response form requested by the task."
    ]
    prompt_interface: MeasurementPromptInterface
    seeds: tuple[Literal[42]]
    temperature: StrictFloat
    partial_results_allowed: Literal[False]

    @model_validator(mode="after")
    def exact_float_protocol(self) -> QualityConfig:
        if self.printed_perplexity_absolute_tolerance != 0.0000501:
            raise ValueError("printed perplexity tolerance must be exactly 0.0000501")
        if self.temperature != 0.0:
            raise ValueError("quality temperature must be exactly 0.0")
        return self


class BenchCase(StrictFrozenModel):
    name: Literal["pp512", "pp2048", "tg128"]
    prompt_tokens: Literal[0, 512, 2048]
    generation_tokens: Literal[0, 128]


class LlamaBenchConfig(StrictFrozenModel):
    workload_identity: MeasurementLlamaBenchWorkloadIdentity
    repetitions: Literal[5]
    warmup_enabled: Literal[True]
    output_format: Literal["jsonl"]
    batch_size: Literal[2048]
    ubatch_size: Literal[512]
    threads: Literal[16]
    cases: tuple[BenchCase, BenchCase, BenchCase]

    @field_validator("workload_identity", mode="before")
    @classmethod
    def normalize_yaml_workload_identity(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        normalized = dict(value)
        ordered_sample_ids = normalized.get("ordered_sample_ids")
        cases = normalized.get("cases")
        if isinstance(ordered_sample_ids, list):
            normalized["ordered_sample_ids"] = tuple(ordered_sample_ids)
        if isinstance(cases, list):
            normalized["cases"] = tuple(cases)
        return normalized

    @model_validator(mode="after")
    def exact_cases(self) -> LlamaBenchConfig:
        observed = tuple(
            (case.name, case.prompt_tokens, case.generation_tokens) for case in self.cases
        )
        expected = (("pp512", 512, 0), ("pp2048", 2048, 0), ("tg128", 0, 128))
        if observed != expected:
            raise ValueError("llama-bench cases must be pp512, pp2048, then tg128")
        identity_cases = tuple(
            (case.sample_id, case.prompt_tokens, case.generation_tokens)
            for case in self.workload_identity.cases
        )
        if identity_cases != observed:
            raise ValueError("llama-bench cases differ from the inspectable workload identity")
        return self


class ServerBenchmarkConfig(StrictFrozenModel):
    workload_identity: MeasurementServerWorkloadIdentity
    concurrency: tuple[Literal[1], Literal[2], Literal[4]]
    load_pair_repetitions: Literal[3]
    single_request_warmups: Literal[2]
    concurrent_batch_warmups: Literal[1]
    measured_batches: Literal[5]
    prompt_tokens: Literal[512]
    output_tokens: Literal[128]
    streaming: Literal[True]
    telemetry_interval_seconds: StrictFloat
    cold_cache_conditioning: Literal[
        "file_level_posix_fadvise_posix_fadv_dontneed_on_all_staged_gguf_files"
    ]
    warm_load_protocol: Literal[
        "second_same_artifact_process_after_cold_termination_without_requested_cache_conditioning_or_eviction"
    ]
    metrics: tuple[
        Literal["cold_server_process_load_seconds"],
        Literal["warm_server_process_load_seconds"],
        Literal["ttft_seconds"],
        Literal["prompt_tokens_per_second"],
        Literal["decode_tokens_per_second"],
        Literal["aggregate_decode_tokens_per_second"],
        Literal["request_end_to_end_latency_seconds"],
        Literal["inter_token_latency_p50_seconds"],
        Literal["inter_token_latency_p95_seconds"],
        Literal["inter_token_latency_p99_seconds"],
        Literal["max_sampled_host_rss_bytes"],
        Literal["max_sampled_per_gpu_memory_bytes"],
        Literal["max_sampled_per_gpu_utilization_percent"],
    ]

    @field_validator("workload_identity", mode="before")
    @classmethod
    def normalize_yaml_workload_identity(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        normalized = dict(value)
        ordered_sample_ids = normalized.get("ordered_sample_ids")
        if isinstance(ordered_sample_ids, list):
            normalized["ordered_sample_ids"] = tuple(ordered_sample_ids)
        return normalized

    @field_validator("telemetry_interval_seconds")
    @classmethod
    def exact_telemetry_interval(cls, value: float) -> float:
        if value != 1.0:
            raise ValueError("telemetry interval must be exactly 1.0 seconds")
        return value

    @model_validator(mode="after")
    def exact_workload(self) -> ServerBenchmarkConfig:
        identity = self.workload_identity
        if (
            identity.prompt_tokens != self.prompt_tokens
            or identity.output_tokens != self.output_tokens
            or identity.streaming != self.streaming
        ):
            raise ValueError("server settings differ from the inspectable workload identity")
        return self


class PerformanceConfig(StrictFrozenModel):
    llama_bench: LlamaBenchConfig
    server: ServerBenchmarkConfig
    serialized_size_metrics: tuple[
        Literal["text_checkpoint_size_bytes"],
        Literal["multimodal_projector_size_bytes"],
        Literal["executable_gguf_bundle_size_bytes"],
    ]


class NonInferiorityConfig(StrictFrozenModel):
    mean_nll_delta_max_exclusive: StrictFloat
    suite_accuracy_loss_max_inclusive: StrictFloat
    overall_accuracy_loss_max_inclusive: StrictFloat
    bf16_overall_accuracy_min_inclusive: StrictFloat
    bf16_suite_accuracy_min_inclusive: StrictFloat
    require_all_items_scored: Literal[True]
    require_all_suites_interpretable: Literal[True]

    @model_validator(mode="after")
    def exact_thresholds(self) -> NonInferiorityConfig:
        expected = {
            "mean_nll_delta_max_exclusive": 0.1,
            "suite_accuracy_loss_max_inclusive": 0.125,
            "overall_accuracy_loss_max_inclusive": 0.05,
            "bf16_overall_accuracy_min_inclusive": 0.75,
            "bf16_suite_accuracy_min_inclusive": 0.5,
        }
        for field_name, expected_value in expected.items():
            if getattr(self, field_name) != expected_value:
                raise ValueError(f"{field_name} differs from the exact measurement protocol")
        return self


class PlacementConfig(StrictFrozenModel):
    backend: Literal["CUDA"]
    logical_devices: Literal["cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7"]
    gpu_layers: Literal["all"]
    cpu_moe_layers: Literal[0]
    split_mode: Literal["layer"]
    tensor_split: tuple[
        Literal[1],
        Literal[1],
        Literal[1],
        Literal[1],
        Literal[1],
        Literal[1],
        Literal[1],
        Literal[1],
    ]
    flash_attention: Literal["on"]
    mmap: Literal[True]
    projector_offload: Literal[True]
    fit: Literal["off"]
    cpu_fallback: Literal[False]


class MeasurementResources(StrictFrozenModel):
    provider: Literal["modal"]
    gpu_type: Literal["B300"]
    gpu_count: Literal[8]
    cpu_cores: Literal[16]
    memory_gib: Literal[64]
    ephemeral_disk_mib: Literal[2097152]
    startup_timeout_seconds: Literal[1800]
    function_timeout_seconds: Literal[86400]
    max_attempts: Literal[1]


class MeasurementStorage(StrictFrozenModel):
    """Isolated append-only storage for measurement authorization and evidence."""

    evidence_volume: Literal["inkling-measurement-evidence-v1"]
    evidence_volume_version: Literal[1]
    evidence_mount_path: Literal["/evidence"]
    evidence_read_only: Literal[False]
    evidence_create_if_missing: Literal[True]
    evidence_append_only_after_terminal: Literal[True]
    attempt_registry: Literal["inkling-measurement-attempt-registry-v1"]
    attempt_registry_append_only: Literal[True]


class MeasurementExecution(StrictFrozenModel):
    remote_execution_policy: Literal["fresh_content_addressed_confirmation_required"]
    remote_execution_default_enabled: Literal[False]
    confirmation_reuse_allowed: Literal[False]
    network_access: Literal[False]
    subject_mode: Literal["sequential_same_allocation"]
    subject_order: tuple[Literal["bf16"], Literal["q3"]]
    subject_staging: Literal["sequential_verified_ephemeral_copy"]
    subject_staging_root: Literal["/cache/inkling-measurement-subject"]
    subject_staging_headroom_mib: Literal[131072]
    release_staged_subject_before_next: Literal[True]
    rehash_during_verified_staging_copy: Literal[True]
    fresh_process_per_tool_invocation: Literal[True]
    bench_cases_share_one_model_load: Literal[True]
    server_quality_and_performance_share_one_model_load: Literal[True]
    rehash_all_subject_files: Literal[True]
    partial_success_allowed: Literal[False]
    planned_stages: tuple[str, ...]

    @model_validator(mode="after")
    def exact_execution_order(self) -> MeasurementExecution:
        if self.subject_order != ("bf16", "q3"):
            raise ValueError("measurement subjects must run BF16 then Q3")
        if self.planned_stages != _PLANNED_STAGES:
            raise ValueError("measurement stages differ from the checked order")
        return self


class MeasurementEvidence(StrictFrozenModel):
    record_prompt_text: Literal[False]
    record_output_text: Literal[False]
    record_token_ids: Literal[True]
    record_item_scores: Literal[True]
    record_failures: Literal[True]
    record_artifact_hashes: Literal[True]
    record_runtime_identity: Literal[True]
    record_hardware_identity: Literal[True]
    record_command_arguments: Literal[True]
    record_raw_trial_timings: Literal[True]
    immutable_after_success: Literal[True]


class MeasurementClaims(StrictFrozenModel):
    compatibility_scope: Literal["single_exact_matrix_cell"]
    mtp_included: Literal[False]
    mtp_supported: Literal[False]
    routing_drift_supported: Literal[False]
    routing_drift_status: Literal["unsupported_by_pinned_runtime"]
    quality_retention_requires_non_inferiority_pass: Literal[True]
    speedup_requires_equivalent_matched_trials: Literal[True]
    single_run_causation_claim_allowed: Literal[False]
    scope_warning: Literal[
        "Read the machine-readable record before use. Do not apply a result to a "
        "different model, dataset, runtime, software, hardware, or protocol."
    ]


class InklingMeasurementConfig(StrictFrozenModel):
    """Complete checked plan for one matched BF16-versus-Q3 Modal run."""

    schema_version: Literal["inkling-measurement-config-v1"]
    model_id: Literal["thinkingmachines/Inkling"]
    revision: Literal["86b4d430ab871652a707666b89203a866888c5e5"]
    architecture: Literal["InklingForConditionalGeneration"]
    matched_cell_config: FileIdentity
    bf16_subject_reference: RecordIdentity
    q3_verified_export_reference: RecordIdentity
    source_adoption_reference: RecordIdentity
    base_runtime: MatchedRuntimeConfig
    measurement_patch: FileIdentity
    measurement_patch_apply_after: Literal["base_runtime.instrumentation_patch"]
    measurement_token_nll_schema_version: Literal["iql-token-nll-v1"]
    measurement_binary_identity_policy: Literal["rebuild_after_patches_then_hash_and_record"]
    quality: QualityConfig
    performance: PerformanceConfig
    non_inferiority: NonInferiorityConfig
    placement: PlacementConfig
    resources: MeasurementResources
    storage: MeasurementStorage
    execution: MeasurementExecution
    evidence: MeasurementEvidence
    claims: MeasurementClaims

    @model_validator(mode="after")
    def safe_contract(self) -> InklingMeasurementConfig:
        literal_secret = sensitive_literal_path(self.model_dump(mode="json"))
        if literal_secret is not None:
            raise ValueError(
                "measurement configuration contains literal credential material at "
                + ".".join(literal_secret)
            )
        return self

    def canonical_json(self) -> str:
        return _canonical_json(self.model_dump(mode="json"))

    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def measurement_protocol_sha256(config: InklingMeasurementConfig) -> str:
    """Hash the exact runtime, method, and reporting rules for one measurement."""

    if not isinstance(config, InklingMeasurementConfig):
        raise TypeError("measurement protocol identity requires a validated config")
    payload = {
        "schema_version": "inkling-measurement-protocol-identity-v1",
        "base_runtime": config.base_runtime.model_dump(mode="json"),
        "measurement_patch": config.measurement_patch.model_dump(mode="json"),
        "measurement_patch_apply_after": config.measurement_patch_apply_after,
        "measurement_token_nll_schema_version": (config.measurement_token_nll_schema_version),
        "measurement_binary_identity_policy": (config.measurement_binary_identity_policy),
        "quality": config.quality.model_dump(
            mode="json",
            exclude={"diagnostic", "corpus_reference"},
        ),
        "performance": config.performance.model_dump(mode="json"),
        "non_inferiority": config.non_inferiority.model_dump(mode="json"),
        "placement": config.placement.model_dump(mode="json"),
        "resources": config.resources.model_dump(mode="json"),
        "execution": config.execution.model_dump(mode="json"),
        "evidence": config.evidence.model_dump(mode="json"),
        "claims": config.claims.model_dump(mode="json"),
    }
    return hashlib.sha256(
        MEASUREMENT_PROTOCOL_HASH_DOMAIN + _canonical_json(payload).encode("utf-8")
    ).hexdigest()


def measurement_workload_sha256(config: InklingMeasurementConfig) -> str:
    """Hash the exact model subjects, data, and workloads used by the comparison."""

    if not isinstance(config, InklingMeasurementConfig):
        raise TypeError("measurement workload identity requires a validated config")
    payload = {
        "schema_version": "inkling-measurement-workload-identity-v1",
        "model_id": config.model_id,
        "model_revision": config.revision,
        "architecture": config.architecture,
        "matched_cell_config": config.matched_cell_config.model_dump(mode="json"),
        "bf16_subject_reference": config.bf16_subject_reference.model_dump(mode="json"),
        "q3_verified_export_reference": (
            config.q3_verified_export_reference.model_dump(mode="json")
        ),
        "source_adoption_reference": (config.source_adoption_reference.model_dump(mode="json")),
        "diagnostic_dataset": config.quality.diagnostic.model_dump(mode="json"),
        "corpus_reference": config.quality.corpus_reference.model_dump(mode="json"),
        "quality_workload": config.quality.model_dump(
            mode="json",
            exclude={"diagnostic", "corpus_reference"},
        ),
        "performance_workload": config.performance.model_dump(mode="json"),
    }
    return hashlib.sha256(
        MEASUREMENT_WORKLOAD_HASH_DOMAIN + _canonical_json(payload).encode("utf-8")
    ).hexdigest()


class InklingMeasurementBundle(StrictFrozenModel):
    """Measurement plan plus all locally verified immutable input records."""

    config: InklingMeasurementConfig
    matched: InklingMatchedCellBundle
    corpus: CorpusReference
    diagnostic_items: tuple[DiagnosticItem, ...]


def parse_measurement_config_bytes(
    raw_bytes: bytes,
    *,
    source: str | Path = "<bytes>",
) -> InklingMeasurementConfig:
    try:
        raw = yaml.load(raw_bytes.decode("utf-8"), Loader=_DuplicateKeyRejectingSafeLoader)
        if not isinstance(raw, Mapping):
            raise ValueError("measurement config root must be a mapping")
        return InklingMeasurementConfig.model_validate(raw)
    except (UnicodeDecodeError, ValueError, yaml.YAMLError, ValidationError) as error:
        raise ConfigurationError(
            f"Unable to parse Inkling measurement config {source}: {error}",
            component="inkling_measurement_config",
        ) from error


def load_measurement_config(path: str | Path) -> InklingMeasurementConfig:
    config_path = Path(path)
    try:
        raw_bytes = config_path.read_bytes()
    except OSError as error:
        raise ConfigurationError(
            f"Unable to load Inkling measurement config {config_path}: {error}",
            component="inkling_measurement_config",
        ) from error
    return parse_measurement_config_bytes(raw_bytes, source=config_path)


def load_corpus_reference(path: str | Path) -> CorpusReference:
    reference_path = Path(path)
    try:
        raw_bytes = reference_path.read_bytes()
        raw = _strict_json(raw_bytes.decode("utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError("corpus reference root must be a JSON object")
        reference = CorpusReference.model_validate(raw)
    except (OSError, UnicodeDecodeError, ValueError, ValidationError) as error:
        raise ConfigurationError(
            f"Unable to load corpus reference {reference_path}: {error}",
            component="inkling_measurement_corpus",
        ) from error
    if raw_bytes != (reference.canonical_json() + "\n").encode("utf-8"):
        raise ConfigurationError(
            "corpus reference must use canonical JSON plus one newline",
            component="inkling_measurement_corpus",
        )
    return reference


def load_diagnostic_items(path: str | Path) -> tuple[DiagnosticItem, ...]:
    dataset_path = Path(path)
    try:
        raw_bytes = dataset_path.read_bytes()
        raw_lines = raw_bytes.decode("utf-8").splitlines(keepends=True)
        if len(raw_lines) != 64 or any(not line.endswith("\n") for line in raw_lines):
            raise ValueError("diagnostic dataset must contain exactly 64 newline-terminated lines")
        items: list[DiagnosticItem] = []
        for line_number, line in enumerate(raw_lines, start=1):
            raw = _strict_json(line)
            if not isinstance(raw, Mapping):
                raise ValueError(f"line {line_number} root must be a JSON object")
            item = DiagnosticItem.model_validate(raw)
            canonical = (
                item.model_dump_json(
                    by_alias=True,
                    exclude_none=True,
                )
                + "\n"
            )
            if line != canonical:
                raise ValueError(f"line {line_number} is not canonical diagnostic JSONL")
            items.append(item)
    except (OSError, UnicodeDecodeError, ValueError, ValidationError) as error:
        raise ConfigurationError(
            f"Unable to load diagnostic dataset {dataset_path}: {error}",
            component="inkling_measurement_diagnostic",
        ) from error

    expected_ids = tuple(f"{suite}_{index:02d}" for suite in _SUITES for index in range(1, 9))
    observed_ids = tuple(item.item_id for item in items)
    if observed_ids != expected_ids:
        raise ConfigurationError(
            "diagnostic item IDs must use the checked stable order",
            component="inkling_measurement_diagnostic",
        )
    suite_counts = Counter(item.suite for item in items)
    if suite_counts != Counter({suite: 8 for suite in _SUITES}):
        raise ConfigurationError(
            "diagnostic dataset must contain exactly eight items per suite",
            component="inkling_measurement_diagnostic",
        )
    scorer_kinds = {item.scorer.kind for item in items}
    if scorer_kinds != {"choice", "integer", "exact_text", "json_exact"}:
        raise ConfigurationError(
            "diagnostic dataset must exercise all four machine scorers",
            component="inkling_measurement_diagnostic",
        )
    return tuple(items)


def _verify_file(path: Path, expected: FileIdentity) -> None:
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ConfigurationError(
            f"Unable to read measurement input {path}: {error}",
            component="inkling_measurement_bundle",
        ) from error
    observed_hash = hashlib.sha256(data).hexdigest()
    if len(data) != expected.size_bytes or observed_hash != expected.sha256:
        raise ConfigurationError(
            "Measurement input byte identity differs from the checked config",
            component="inkling_measurement_bundle",
            details={"path": expected.path},
        )


def _verify_corpus_materializer(root: Path, reference: CorpusReference) -> None:
    materializer_path = _project_file(root, reference.materializer_path)
    try:
        materializer_bytes = materializer_path.read_bytes()
    except OSError as error:
        raise ConfigurationError(
            f"Unable to read corpus materializer {materializer_path}: {error}",
            component="inkling_measurement_corpus",
        ) from error
    if (
        not materializer_bytes
        or hashlib.sha256(materializer_bytes).hexdigest() != reference.materializer_sha256
    ):
        raise ConfigurationError(
            "Corpus materializer byte identity differs from its checked reference",
            component="inkling_measurement_corpus",
            details={"path": reference.materializer_path},
        )


def load_measurement_bundle(
    project_root: str | Path,
    *,
    config_relative_path: str = MEASUREMENT_CONFIG_RELATIVE_PATH,
) -> InklingMeasurementBundle:
    """Load and cross-check the measurement plan without external work."""

    root = Path(project_root).expanduser().resolve()
    if not root.is_dir():
        raise ConfigurationError(
            f"Measurement project root is not a directory: {root}",
            component="inkling_measurement_bundle",
        )
    config_path = _project_file(root, config_relative_path)
    config = load_measurement_config(config_path)

    for identity in (
        config.matched_cell_config,
        config.bf16_subject_reference,
        config.q3_verified_export_reference,
        config.source_adoption_reference,
        config.measurement_patch,
        config.quality.diagnostic,
        config.quality.corpus_reference,
    ):
        _verify_file(_project_file(root, identity.path), identity)

    matched = load_matched_cell_bundle(
        root,
        config_relative_path=config.matched_cell_config.path,
    )
    corpus = load_corpus_reference(_project_file(root, config.quality.corpus_reference.path))
    _verify_corpus_materializer(root, corpus)
    diagnostic_items = load_diagnostic_items(_project_file(root, config.quality.diagnostic.path))

    mismatches: list[str] = []
    if config.base_runtime != matched.config.runtime:
        mismatches.append("base_runtime")
    if config.bf16_subject_reference.path != matched.config.bf16_subject_reference_path:
        mismatches.append("bf16_reference_path")
    if config.bf16_subject_reference.reference_sha256 != matched.bf16.reference_sha256:
        mismatches.append("bf16_reference_sha256")
    if config.q3_verified_export_reference.path != matched.config.q3_verified_export_reference_path:
        mismatches.append("q3_reference_path")
    if config.q3_verified_export_reference.reference_sha256 != matched.q3.reference_sha256:
        mismatches.append("q3_reference_sha256")
    if config.source_adoption_reference.path != matched.config.source_adoption_reference_path:
        mismatches.append("source_reference_path")
    if config.source_adoption_reference.reference_sha256 != matched.source.reference_sha256:
        mismatches.append("source_reference_sha256")
    if config.quality.corpus_reference.reference_sha256 != corpus.reference_sha256:
        mismatches.append("corpus_reference_sha256")
    if (
        _runtime_text(corpus.materializer_path, label="corpus materializer path")
        != CORPUS_MATERIALIZER_RELATIVE_PATH
    ):
        mismatches.append("corpus_materializer_path")
    if (
        _runtime_text(corpus.materialized_path, label="materialized corpus path")
        != MATERIALIZED_CORPUS_PATH
    ):
        mismatches.append("materialized_corpus_path")
    if len(diagnostic_items) != config.quality.diagnostic_item_count:
        mismatches.append("diagnostic_item_count")
    observed_model_identity = (
        _runtime_text(config.model_id, label="measurement model ID"),
        _runtime_text(config.revision, label="measurement model revision"),
        _runtime_text(config.architecture, label="measurement architecture"),
    )
    matched_model_identity = (
        _runtime_text(matched.config.model_id, label="matched model ID"),
        _runtime_text(matched.config.revision, label="matched model revision"),
        _runtime_text(matched.config.architecture, label="matched architecture"),
    )
    if observed_model_identity != matched_model_identity:
        mismatches.append("model_identity")
    if mismatches:
        raise ConfigurationError(
            "Measurement plan is not bound to one exact matched bundle",
            component="inkling_measurement_bundle",
            details={"mismatches": sorted(set(mismatches))},
        )
    return InklingMeasurementBundle(
        config=config,
        matched=matched,
        corpus=corpus,
        diagnostic_items=diagnostic_items,
    )


__all__ = [
    "CORPUS_MATERIALIZER_RELATIVE_PATH",
    "CORPUS_REFERENCE_RELATIVE_PATH",
    "DIAGNOSTIC_DATASET_RELATIVE_PATH",
    "MATERIALIZED_CORPUS_PATH",
    "MEASUREMENT_CONFIG_RELATIVE_PATH",
    "MEASUREMENT_MEDIA_MARKER",
    "MEASUREMENT_PROTOCOL_HASH_DOMAIN",
    "MEASUREMENT_WORKLOAD_HASH_DOMAIN",
    "DiagnosticItem",
    "InklingMeasurementBundle",
    "InklingMeasurementConfig",
    "MeasurementDiagnosticPromptRender",
    "MeasurementPromptInterface",
    "build_diagnostic_fixture_bytes",
    "load_corpus_reference",
    "load_diagnostic_items",
    "load_measurement_bundle",
    "load_measurement_config",
    "measurement_protocol_sha256",
    "measurement_workload_sha256",
    "parse_measurement_config_bytes",
    "render_measurement_diagnostic_prompt",
]
