"""Strict parsers for content-addressed matched-measurement raw evidence.

The contracts in this module are deliberately host-side only.  They validate
canonical records already produced by the approved Modal CUDA measurement; they
do not load a model, execute llama.cpp, launch Modal, or provide a CPU substitute
for accelerator validation.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import statistics
from collections.abc import Sequence
from itertools import pairwise
from pathlib import PurePosixPath
from typing import Final, Literal, TypeAlias, cast

from pydantic import (
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from inkling_quant_lab.config import StrictFrozenModel
from inkling_quant_lab.gguf.inkling_measurement import (
    DiagnosticItem,
    MeasurementPromptInterface,
    build_diagnostic_fixture_bytes,
    render_measurement_diagnostic_prompt,
)
from inkling_quant_lab.gguf.inkling_measurement_execution import (
    PINNED_LLAMA_CPP_BUILD_COMMIT,
    LlamaBenchCommandSpec,
    LlamaPerplexityCommandSpec,
    LlamaServerCommandSpec,
    bind_exact_cuda_topology,
    build_llama_bench_command,
    build_llama_perplexity_command,
    build_llama_server_command,
    diagnostic_expected_normalized_sha256,
)
from inkling_quant_lab.gguf.inkling_smoke import (
    ArtifactLoadEvidence,
    LoaderOffloadEvidence,
    parse_artifact_load_evidence,
    parse_loader_offload_evidence,
)

MeasurementRawEvidenceKind: TypeAlias = Literal[
    "token_nll",
    "raw_trials",
    "resource_telemetry",
    "backend_audit",
]
MeasurementRawSubject: TypeAlias = Literal["bf16", "q3"]
MeasurementRawWorkload: TypeAlias = Literal[
    "perplexity",
    "server_quality_and_performance",
    "llama_bench",
]
MeasurementQualitySuite: TypeAlias = Literal[
    "text",
    "math",
    "code",
    "multilingual",
    "instruction",
    "vision",
    "audio",
    "post_training",
]
MeasurementBenchCase: TypeAlias = Literal["pp512", "pp2048", "tg128"]

MEASUREMENT_RAW_EVIDENCE_KIND_ORDER: Final = (
    "token_nll",
    "raw_trials",
    "resource_telemetry",
    "backend_audit",
)
MEASUREMENT_RAW_WORKLOAD_ORDER: Final = (
    "perplexity",
    "server_quality_and_performance",
    "llama_bench",
)
MEASUREMENT_QUALITY_SUITE_ORDER: Final = (
    "text",
    "math",
    "code",
    "multilingual",
    "instruction",
    "vision",
    "audio",
    "post_training",
)
MEASUREMENT_BENCH_CASE_ORDER: Final = ("pp512", "pp2048", "tg128")
MEASUREMENT_SERVER_CONCURRENCY_ORDER: Final = (1, 2, 4)
MEASUREMENT_TOKEN_NLL_RECORD_COUNT: Final = 16_320
MEASUREMENT_DIAGNOSTIC_ITEM_COUNT: Final = 64
MEASUREMENT_PRINTED_PERPLEXITY_ABSOLUTE_TOLERANCE: Final = 0.0000501
MEASUREMENT_RAW_EVIDENCE_MAX_BYTES: Final[dict[str, int]] = {
    "token_nll": 16 * 1024 * 1024,
    "raw_trials": 64 * 1024 * 1024,
    "resource_telemetry": 256 * 1024 * 1024,
    "backend_audit": 64 * 1024 * 1024,
}
CAPTURED_TOOL_LOG_DELIMITER: Final = "\n===== IQL_CAPTURED_STDOUT_STDERR_V1 =====\n"
MEASUREMENT_REMOTE_CORPUS_PATH: Final = (
    "/opt/inkling-measurement-data/wikitext-2-raw-v1/wiki.test.raw"
)

_RUN_ID_PATTERN: Final = r"^[a-z0-9][a-z0-9._-]{0,95}$"
_SHA256_PATTERN: Final = r"^[0-9a-f]{64}$"
_CALL_ID_PATTERN: Final = r"^fc-[A-Za-z0-9]+$"
_ITEM_ID_PATTERN: Final = r"^[a-z_]+_[0-9]{2}$"
_GPU_UUID_PATTERN: Final = r"^GPU-[A-Za-z0-9-]+$"
_UINT64_MASK: Final = (1 << 64) - 1


class MeasurementRawEvidenceError(ValueError):
    """Base error for malformed, non-canonical, or incompatible raw evidence."""


class MeasurementRawEvidenceSizeError(MeasurementRawEvidenceError):
    """Raw evidence is empty or exceeds its approved byte bound."""


class MeasurementRawEvidenceCanonicalError(MeasurementRawEvidenceError):
    """Raw evidence is not strict canonical UTF-8 JSON or JSONL."""


class MeasurementRawEvidenceBindingError(MeasurementRawEvidenceError):
    """Raw evidence records do not belong to one exact accepted attempt."""


class _StrictRawModel(StrictFrozenModel):
    """Finite, immutable, extra-forbid base for every raw evidence model."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )


def canonical_measurement_raw_json_bytes(value: object) -> bytes:
    """Encode one canonical UTF-8 JSON value followed by one line feed."""

    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise MeasurementRawEvidenceCanonicalError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise MeasurementRawEvidenceCanonicalError(f"non-finite JSON number is forbidden: {value}")


def _strict_json_value(payload: bytes, *, label: str) -> object:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise MeasurementRawEvidenceCanonicalError(f"{label} is not strict UTF-8") from error
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except json.JSONDecodeError as error:
        raise MeasurementRawEvidenceCanonicalError(f"{label} is not valid JSON") from error


def _require_bounded_payload(
    payload: bytes,
    *,
    kind: MeasurementRawEvidenceKind,
) -> None:
    if type(payload) is not bytes:
        raise TypeError("raw evidence payload must be exact bytes")
    maximum = MEASUREMENT_RAW_EVIDENCE_MAX_BYTES[kind]
    if not payload or len(payload) > maximum:
        raise MeasurementRawEvidenceSizeError(
            f"{kind} evidence must contain 1 through {maximum} bytes"
        )


def _parse_canonical_object(
    payload: bytes,
    *,
    kind: MeasurementRawEvidenceKind,
    model: type[_StrictRawModel],
) -> _StrictRawModel:
    _require_bounded_payload(payload, kind=kind)
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise MeasurementRawEvidenceCanonicalError(f"{kind} JSON must end in exactly one line feed")
    value = _strict_json_value(payload, label=kind)
    if not isinstance(value, dict):
        raise MeasurementRawEvidenceCanonicalError(f"{kind} evidence must be one JSON object")
    if canonical_measurement_raw_json_bytes(value) != payload:
        raise MeasurementRawEvidenceCanonicalError(f"{kind} evidence is not canonical JSON")
    return model.model_validate_json(payload, strict=True)


def _parse_canonical_jsonl(
    payload: bytes,
    *,
    kind: MeasurementRawEvidenceKind,
    model: type[_StrictRawModel],
) -> tuple[_StrictRawModel, ...]:
    _require_bounded_payload(payload, kind=kind)
    if not payload.endswith(b"\n"):
        raise MeasurementRawEvidenceCanonicalError(f"{kind} JSONL must end in one line feed")
    lines = payload[:-1].split(b"\n")
    if not lines or any(not line for line in lines):
        raise MeasurementRawEvidenceCanonicalError(f"{kind} JSONL contains an empty record")
    parsed: list[_StrictRawModel] = []
    for ordinal, line in enumerate(lines):
        value = _strict_json_value(line, label=f"{kind} row {ordinal}")
        if not isinstance(value, dict):
            raise MeasurementRawEvidenceCanonicalError(f"{kind} row {ordinal} is not a JSON object")
        if canonical_measurement_raw_json_bytes(value) != line + b"\n":
            raise MeasurementRawEvidenceCanonicalError(
                f"{kind} row {ordinal} is not canonical JSON"
            )
        parsed.append(model.model_validate_json(line, strict=True))
    return tuple(parsed)


def _canonical_absolute_posix_path(value: str) -> str:
    if (
        not value
        or "\x00" in value
        or "\\" in value
        or not value.startswith("/")
        or value.startswith("//")
    ):
        raise ValueError("path must be a single-rooted absolute POSIX path without backslashes")
    path = PurePosixPath(value)
    if value != path.as_posix() or any(part in {".", ".."} for part in path.parts):
        raise ValueError("path must use its canonical absolute POSIX spelling")
    return value


def _float_equal(
    observed: float,
    expected: float,
    *,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-12,
) -> bool:
    return math.isfinite(expected) and math.isclose(
        observed,
        expected,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )


def _r7_percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


class MeasurementFiveBatchMetricSummary(_StrictRawModel):
    """Mean, median, and sample deviation for five retained server batches."""

    trial_count: Literal[5]
    samples: tuple[
        StrictFloat,
        StrictFloat,
        StrictFloat,
        StrictFloat,
        StrictFloat,
    ]
    mean: StrictFloat = Field(gt=0.0)
    median: StrictFloat = Field(gt=0.0)
    sample_standard_deviation: StrictFloat = Field(ge=0.0)

    @model_validator(mode="after")
    def statistics_are_derived(self) -> MeasurementFiveBatchMetricSummary:
        if any(value <= 0.0 for value in self.samples):
            raise ValueError("server batch metric samples must be positive")
        expected = (
            statistics.fmean(self.samples),
            statistics.median(self.samples),
            statistics.stdev(self.samples),
        )
        observed = (
            self.mean,
            self.median,
            self.sample_standard_deviation,
        )
        if any(
            not _float_equal(item, target) for item, target in zip(observed, expected, strict=True)
        ):
            raise ValueError("server batch metric statistics differ from retained samples")
        return self


class MeasurementFiveBatchNonnegativeMetricSummary(_StrictRawModel):
    """Five-batch statistics for a metric whose samples may legitimately be zero."""

    trial_count: Literal[5]
    samples: tuple[
        StrictFloat,
        StrictFloat,
        StrictFloat,
        StrictFloat,
        StrictFloat,
    ]
    mean: StrictFloat = Field(ge=0.0)
    median: StrictFloat = Field(ge=0.0)
    sample_standard_deviation: StrictFloat = Field(ge=0.0)

    @model_validator(mode="after")
    def statistics_are_derived(self) -> MeasurementFiveBatchNonnegativeMetricSummary:
        if any(value < 0.0 for value in self.samples):
            raise ValueError("server batch metric samples must be nonnegative")
        expected = (
            statistics.fmean(self.samples),
            statistics.median(self.samples),
            statistics.stdev(self.samples),
        )
        observed = (
            self.mean,
            self.median,
            self.sample_standard_deviation,
        )
        if any(
            not _float_equal(item, target) for item, target in zip(observed, expected, strict=True)
        ):
            raise ValueError("server batch metric statistics differ from retained samples")
        return self


class MeasurementRepeatedLoadDurations(_StrictRawModel):
    """Ordered real load durations with exact repetition and dispersion."""

    trial_count: StrictInt = Field(ge=2, le=64)
    durations_seconds: tuple[StrictFloat, ...] = Field(min_length=2, max_length=64)
    median_seconds: StrictFloat = Field(gt=0.0)
    sample_standard_deviation_seconds: StrictFloat = Field(ge=0.0)

    @model_validator(mode="after")
    def statistics_are_derived(self) -> MeasurementRepeatedLoadDurations:
        if len(self.durations_seconds) != self.trial_count:
            raise ValueError("load-duration count differs from retained trials")
        if any(value <= 0.0 for value in self.durations_seconds):
            raise ValueError("load-duration trials must be positive")
        if not _float_equal(
            self.median_seconds,
            statistics.median(self.durations_seconds),
        ):
            raise ValueError("load-duration median differs from retained trials")
        if not _float_equal(
            self.sample_standard_deviation_seconds,
            statistics.stdev(self.durations_seconds),
        ):
            raise ValueError("load-duration sample deviation differs from retained trials")
        return self


def _sha256_canonical(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + canonical_measurement_raw_json_bytes(value)).hexdigest()


class MeasurementAttemptBindings(_StrictRawModel):
    """Exact scope shared by every raw record for one accepted subject attempt."""

    schema_version: Literal["inkling-measurement-raw-bindings-v1"] = (
        "inkling-measurement-raw-bindings-v1"
    )
    run_id: StrictStr = Field(pattern=_RUN_ID_PATTERN)
    subject: MeasurementRawSubject
    reviewed_config_file_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    resolved_config_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    protocol_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    workload_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    launch_intent_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    post_spawn_acceptance_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    call_id: StrictStr = Field(pattern=_CALL_ID_PATTERN)
    attempt_claim_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)


class MeasurementTokenNllRow(_StrictRawModel):
    """One scored token, with no prompts, outputs, or attempt metadata."""

    chunk_index: StrictInt = Field(ge=0, le=63)
    token_index: StrictInt = Field(ge=257, le=32767)
    token_id: StrictInt = Field(ge=0)
    nll: StrictFloat = Field(ge=0.0)


class MeasurementTokenNllEvidence(_StrictRawModel):
    """The complete canonical sequence of 64 by 255 token-NLL rows."""

    rows: tuple[MeasurementTokenNllRow, ...] = Field(
        min_length=MEASUREMENT_TOKEN_NLL_RECORD_COUNT,
        max_length=MEASUREMENT_TOKEN_NLL_RECORD_COUNT,
    )

    @model_validator(mode="after")
    def rows_have_exact_order(self) -> MeasurementTokenNllEvidence:
        for ordinal, row in enumerate(self.rows):
            chunk_index, within_chunk = divmod(ordinal, 255)
            expected_token_index = chunk_index * 512 + 257 + within_chunk
            if row.chunk_index != chunk_index or row.token_index != expected_token_index:
                raise ValueError("token NLL rows must use each chunk's scored indexes 257..511")
        return self


class MeasurementStagedArtifact(_StrictRawModel):
    """One source artifact copied and hashed in a single staging pass."""

    source_path: StrictStr
    staged_path: StrictStr
    sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    size_bytes: StrictInt = Field(gt=0)
    source_passes: Literal[1]

    @field_validator("source_path", "staged_path")
    @classmethod
    def path_is_canonical(cls, value: str) -> str:
        return _canonical_absolute_posix_path(value)


class MeasurementSubjectStaging(_StrictRawModel):
    """Complete same-pass staged inventory for one measured subject."""

    schema_version: Literal["inkling-measurement-subject-staging-v1"]
    subject: MeasurementRawSubject
    source_volume_read_only: Literal[True]
    copy_and_hash_same_source_pass: Literal[True]
    source_passes_per_artifact: Literal[1]
    staging_root: StrictStr
    artifact_count: Literal[57, 60]
    required_bytes: StrictInt = Field(gt=0)
    required_headroom_bytes: Literal[137438953472]
    free_bytes_before_staging: StrictInt = Field(gt=0)
    artifacts: tuple[MeasurementStagedArtifact, ...] = Field(
        min_length=57,
        max_length=60,
    )
    inventory_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)

    @field_validator("staging_root")
    @classmethod
    def root_is_canonical(cls, value: str) -> str:
        return _canonical_absolute_posix_path(value)

    @model_validator(mode="after")
    def inventory_is_exact(self) -> MeasurementSubjectStaging:
        expected_root = f"/cache/inkling-measurement-subject/{self.subject}"
        if self.staging_root != expected_root:
            raise ValueError("staging root differs from the bound subject")
        expected_count = 57 if self.subject == "bf16" else 60
        if self.artifact_count != expected_count or len(self.artifacts) != expected_count:
            raise ValueError("staging artifact_count differs from the inventory")
        if (
            len({item.source_path for item in self.artifacts}) != expected_count
            or len({item.staged_path for item in self.artifacts}) != expected_count
        ):
            raise ValueError("staged source and destination paths must be unique")
        shard_label = "inkling-BF16" if self.subject == "bf16" else "inkling-Q3_K_M"
        shard_directory = "bf16" if self.subject == "bf16" else "q3_k_m"
        expected_suffixes = tuple(
            f"/{shard_directory}/{shard_label}-{ordinal:05d}-of-00049.gguf"
            for ordinal in range(1, 50)
        )
        if self.subject == "bf16":
            expected_suffixes += (
                "/convert_text_bf16.success.json",
                "/mmproj/mmproj-BF16.gguf",
            )
        else:
            expected_suffixes += (
                "/mmproj/mmproj-BF16.gguf",
                "/verification/export_manifest.json",
                "/verify_export.success.json",
                "/quantize_text.success.json",
                "/convert_multimodal_projector.success.json",
            )
        expected_suffixes += (
            "/chat_template.jinja",
            "/processor_config.json",
            "/special_tokens_map.json",
            "/tiktoken/tokenizer.model",
            "/tokenizer.json",
            "/tokenizer_config.json",
        )
        if any(
            not item.source_path.endswith(suffix)
            for item, suffix in zip(
                self.artifacts,
                expected_suffixes,
                strict=True,
            )
        ):
            raise ValueError(
                "staged full subject inventory names or order differ from the protocol"
            )
        prefix = self.staging_root + "/"
        if any(not item.staged_path.startswith(prefix) for item in self.artifacts):
            raise ValueError("staged artifact is outside the subject staging root")
        if self.required_bytes != sum(item.size_bytes for item in self.artifacts):
            raise ValueError("required staging bytes differ from artifact sizes")
        if self.free_bytes_before_staging < self.required_bytes + self.required_headroom_bytes:
            raise ValueError("staging did not retain the required 128 GiB headroom")
        identity = {
            "artifacts": [item.model_dump(mode="json") for item in self.artifacts],
            "required_bytes": self.required_bytes,
            "required_headroom_bytes": self.required_headroom_bytes,
        }
        expected_hash = hashlib.sha256(canonical_measurement_raw_json_bytes(identity)).hexdigest()
        if self.inventory_sha256 != expected_hash:
            raise ValueError("staging inventory SHA-256 differs from its contents")
        return self


class MeasurementProcessTimings(_StrictRawModel):
    """One measured process's monotonic boundaries and derived duration."""

    process_id: StrictInt = Field(gt=0)
    process_started_monotonic_seconds: StrictFloat = Field(gt=0.0)
    process_finished_monotonic_seconds: StrictFloat = Field(gt=0.0)
    elapsed_seconds: StrictFloat = Field(gt=0.0)

    @model_validator(mode="after")
    def elapsed_is_derived(self) -> MeasurementProcessTimings:
        expected = self.process_finished_monotonic_seconds - self.process_started_monotonic_seconds
        if expected <= 0.0 or not _float_equal(self.elapsed_seconds, expected):
            raise ValueError("process elapsed time differs from monotonic boundaries")
        return self


class MeasurementPerplexityTrial(MeasurementProcessTimings):
    """Pinned perplexity command result; token rows live in the separate blob."""

    command: tuple[StrictStr, ...] = Field(min_length=1)
    corpus_reference_sha256: Literal[
        "5dfc8c426a1509c28d119857f437365c90a4bd57e229705d60e6fd3c1c65b95d"
    ]
    corpus_sha256: Literal["173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08"]
    corpus_size_bytes: Literal[1290590]
    perplexity: StrictFloat = Field(gt=0.0)
    uncertainty: StrictFloat = Field(ge=0.0)
    token_nll_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    token_nll_size_bytes: StrictInt = Field(
        gt=0,
        le=MEASUREMENT_RAW_EVIDENCE_MAX_BYTES["token_nll"],
    )
    stdout_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    stdout_size_bytes: StrictInt = Field(ge=0)
    stderr_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    stderr_size_bytes: StrictInt = Field(ge=0)

    @field_validator("command")
    @classmethod
    def command_strings_are_safe(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item or "\x00" in item for item in value):
            raise ValueError("perplexity command contains an empty or NUL argument")
        return value


class MeasurementDiagnosticTimings(_StrictRawModel):
    """One diagnostic response's llama.cpp timing counters."""

    prompt_n: StrictInt = Field(gt=0)
    predicted_n: StrictInt = Field(gt=0, le=64)
    prompt_ms: StrictFloat = Field(gt=0.0)
    predicted_ms: StrictFloat = Field(gt=0.0)
    prompt_per_second: StrictFloat = Field(gt=0.0)
    predicted_per_second: StrictFloat = Field(gt=0.0)

    @model_validator(mode="after")
    def rates_are_derived(self) -> MeasurementDiagnosticTimings:
        prompt_rate = 1000.0 * self.prompt_n / self.prompt_ms
        predicted_rate = 1000.0 * self.predicted_n / self.predicted_ms
        if not _float_equal(self.prompt_per_second, prompt_rate, rel_tol=5e-6, abs_tol=5e-6):
            raise ValueError("diagnostic prompt rate differs from count and time")
        if not _float_equal(
            self.predicted_per_second,
            predicted_rate,
            rel_tol=5e-6,
            abs_tol=5e-6,
        ):
            raise ValueError("diagnostic decode rate differs from count and time")
        return self


class MeasurementDiagnosticTrial(_StrictRawModel):
    """The one deterministic trial retained for one diagnostic item."""

    trial_index: Literal[1]
    request_started_monotonic_seconds: StrictFloat = Field(gt=0.0)
    request_finished_monotonic_seconds: StrictFloat = Field(gt=0.0)
    request_wall_seconds: StrictFloat = Field(gt=0.0)
    token_ids: tuple[StrictInt, ...] = Field(min_length=1, max_length=64)
    output_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    response_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    normalization_succeeded: StrictBool
    normalized_sha256: StrictStr | None = Field(pattern=_SHA256_PATTERN)
    score: StrictBool
    timings: MeasurementDiagnosticTimings

    @field_validator("token_ids")
    @classmethod
    def token_ids_are_nonnegative(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(token_id < 0 for token_id in value):
            raise ValueError("diagnostic token IDs must be nonnegative")
        return value

    @model_validator(mode="after")
    def token_count_matches_timings(self) -> MeasurementDiagnosticTrial:
        elapsed = self.request_finished_monotonic_seconds - self.request_started_monotonic_seconds
        if elapsed <= 0.0 or not _float_equal(self.request_wall_seconds, elapsed):
            raise ValueError("diagnostic request wall time differs from monotonic boundaries")
        if self.timings.predicted_n != len(self.token_ids):
            raise ValueError("diagnostic predicted_n differs from retained token IDs")
        has_normalized_value = self.normalized_sha256 is not None
        if self.normalization_succeeded != has_normalized_value:
            raise ValueError(
                "diagnostic normalized hash must appear exactly when normalization succeeds"
            )
        if self.score and not self.normalization_succeeded:
            raise ValueError("diagnostic score cannot pass when normalization fails")
        return self


class MeasurementDiagnosticItem(_StrictRawModel):
    """One prompt-hash-bound, machine-scored diagnostic result."""

    item_id: StrictStr = Field(pattern=_ITEM_ID_PATTERN)
    suite: MeasurementQualitySuite
    modality: Literal["text", "image", "audio"]
    request_body_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    prompt_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    fixture_sha256: StrictStr | None = Field(default=None, pattern=_SHA256_PATTERN)
    fixture_size_bytes: StrictInt | None = Field(default=None, gt=0)
    seed: Literal[42]
    temperature: StrictFloat
    max_new_tokens: StrictInt = Field(ge=1, le=64)
    scorer_kind: Literal["choice", "integer", "exact_text", "json_exact"]
    score: StrictBool
    trials: tuple[MeasurementDiagnosticTrial] = Field(min_length=1, max_length=1)
    prompt_text_recorded: Literal[False]
    output_text_recorded: Literal[False]

    @field_validator("temperature")
    @classmethod
    def temperature_is_exact(cls, value: float) -> float:
        if value != 0.0:
            raise ValueError("diagnostic temperature must be exactly zero")
        return value

    @model_validator(mode="after")
    def modality_and_trial_are_exact(self) -> MeasurementDiagnosticItem:
        expected_modality = (
            "image" if self.suite == "vision" else "audio" if self.suite == "audio" else "text"
        )
        if self.modality != expected_modality:
            raise ValueError("diagnostic modality differs from its suite")
        has_fixture = self.fixture_sha256 is not None
        if has_fixture != (self.fixture_size_bytes is not None):
            raise ValueError("diagnostic fixture hash and size must appear together")
        if has_fixture != (self.modality != "text"):
            raise ValueError("only image and audio diagnostics may bind fixtures")
        trial = self.trials[0]
        if trial.score != self.score:
            raise ValueError("diagnostic item score differs from its only trial")
        if len(trial.token_ids) > self.max_new_tokens:
            raise ValueError("diagnostic output exceeds max_new_tokens")
        return self


class MeasurementHardwareGpu(_StrictRawModel):
    """One exact CUDA-ordinal B300 identity row."""

    cuda_ordinal: StrictInt = Field(ge=0, le=7)
    uuid: StrictStr = Field(pattern=_GPU_UUID_PATTERN)
    name: Literal["NVIDIA B300 SXM6 AC"]
    memory_total_mib: Literal[275040]
    driver_version: StrictStr = Field(pattern=r"^[0-9]+(?:\.[0-9]+)+$")
    compute_capability: Literal["10.3"]


class MeasurementCudaRuntimeDeviceProbe(_StrictRawModel):
    """One successful CUDA Runtime data-path check for one GPU."""

    cuda_ordinal: StrictInt = Field(ge=0, le=7)
    logical_device: Literal[
        "cuda:0",
        "cuda:1",
        "cuda:2",
        "cuda:3",
        "cuda:4",
        "cuda:5",
        "cuda:6",
        "cuda:7",
    ]
    allocation_size_bytes: Literal[16]
    memset_byte_value: StrictInt = Field(ge=1, le=8)
    copied_payload_hex: StrictStr = Field(pattern=r"^[0-9a-f]{32}$")
    cuda_set_device_result: Literal[0]
    cuda_malloc_result: Literal[0]
    cuda_memset_result: Literal[0]
    cuda_synchronize_after_memset_result: Literal[0]
    cuda_memcpy_device_to_host_result: Literal[0]
    cuda_synchronize_after_copy_result: Literal[0]
    payload_verified: Literal[True]
    cuda_free_result: Literal[0]

    @model_validator(mode="after")
    def probe_is_exact(self) -> MeasurementCudaRuntimeDeviceProbe:
        expected_byte = self.cuda_ordinal + 1
        if self.logical_device != f"cuda:{self.cuda_ordinal}":
            raise ValueError("CUDA Runtime probe logical device differs from its ordinal")
        if self.memset_byte_value != expected_byte:
            raise ValueError("CUDA Runtime probe byte must be unique for its ordinal")
        if self.copied_payload_hex != f"{expected_byte:02x}" * self.allocation_size_bytes:
            raise ValueError("CUDA Runtime copied payload differs from the written bytes")
        return self


class MeasurementCudaRuntimePreflight(_StrictRawModel):
    """Exact successful libcudart check for all eight GPUs."""

    schema_version: Literal["inkling-measurement-cuda-runtime-preflight-v1"]
    protocol: Literal["libcudart-set-malloc-memset-sync-d2h-sync-free-v1"]
    libcudart_soname: StrictStr = Field(pattern=r"^libcudart\.so(?:\.[0-9]+)*$")
    libcudart_path: StrictStr
    libcudart_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    libcudart_size_bytes: StrictInt = Field(gt=0)
    execution_process: Literal["short-lived-subprocess"]
    child_process_exit_code: Literal[0]
    cuda_get_device_count_result: Literal[0]
    observed_device_count: Literal[8]
    probes: tuple[
        MeasurementCudaRuntimeDeviceProbe,
        MeasurementCudaRuntimeDeviceProbe,
        MeasurementCudaRuntimeDeviceProbe,
        MeasurementCudaRuntimeDeviceProbe,
        MeasurementCudaRuntimeDeviceProbe,
        MeasurementCudaRuntimeDeviceProbe,
        MeasurementCudaRuntimeDeviceProbe,
        MeasurementCudaRuntimeDeviceProbe,
    ]
    all_devices_usable: Literal[True]

    @field_validator("libcudart_path")
    @classmethod
    def runtime_path_is_canonical(cls, value: str) -> str:
        return _canonical_absolute_posix_path(value)

    @model_validator(mode="after")
    def preflight_is_complete(self) -> MeasurementCudaRuntimePreflight:
        if tuple(probe.cuda_ordinal for probe in self.probes) != tuple(range(8)):
            raise ValueError("CUDA Runtime probes must use exact ordinal order 0..7")
        if tuple(probe.memset_byte_value for probe in self.probes) != tuple(range(1, 9)):
            raise ValueError("CUDA Runtime probe bytes must be unique and ordered")
        return self


class MeasurementCudaPeerEdge(_StrictRawModel):
    """One of the exact 56 directed CUDA peer edges."""

    source_cuda_ordinal: StrictInt = Field(ge=0, le=7)
    source_uuid: StrictStr = Field(pattern=_GPU_UUID_PATTERN)
    destination_cuda_ordinal: StrictInt = Field(ge=0, le=7)
    destination_uuid: StrictStr = Field(pattern=_GPU_UUID_PATTERN)
    can_access_peer: StrictBool
    performance_rank: StrictInt = Field(ge=0)
    access_supported: StrictBool
    native_atomic_supported: StrictBool
    cuda_array_access_supported: StrictBool
    only_partial_native_atomic_supported: StrictBool

    @model_validator(mode="after")
    def edge_is_consistent(self) -> MeasurementCudaPeerEdge:
        if (
            self.source_cuda_ordinal == self.destination_cuda_ordinal
            or self.source_uuid.lower() == self.destination_uuid.lower()
        ):
            raise ValueError("CUDA peer edge must join distinct GPUs")
        if self.can_access_peer != self.access_supported:
            raise ValueError("CUDA peer access queries disagree")
        if self.native_atomic_supported and self.only_partial_native_atomic_supported:
            raise ValueError("CUDA peer atomic support cannot be both full and partial")
        return self


class MeasurementCudaPeerTopology(_StrictRawModel):
    """Complete ordered peer topology for the exact eight-GPU process."""

    schema_version: Literal["inkling-matched-cuda-peer-topology-v1"]
    protocol: Literal["cuda-driver-p2p-attributes-v1"]
    cuda_driver_api_version: StrictInt = Field(gt=0)
    gpu_uuids: tuple[StrictStr, ...] = Field(min_length=8, max_length=8)
    edges: tuple[MeasurementCudaPeerEdge, ...] = Field(
        min_length=56,
        max_length=56,
    )

    @model_validator(mode="after")
    def topology_is_complete(self) -> MeasurementCudaPeerTopology:
        if len({uuid.lower() for uuid in self.gpu_uuids}) != 8:
            raise ValueError("hardware GPU UUIDs must be unique")
        expected_pairs = tuple(
            (source, destination)
            for source in range(8)
            for destination in range(8)
            if source != destination
        )
        observed_pairs = tuple(
            (edge.source_cuda_ordinal, edge.destination_cuda_ordinal) for edge in self.edges
        )
        if observed_pairs != expected_pairs:
            raise ValueError("peer topology must contain all 56 edges in checked order")
        for edge in self.edges:
            if (
                edge.source_uuid.lower() != self.gpu_uuids[edge.source_cuda_ordinal].lower()
                or edge.destination_uuid.lower()
                != self.gpu_uuids[edge.destination_cuda_ordinal].lower()
            ):
                raise ValueError("peer edge UUID differs from the ordered GPU inventory")
        return self


class MeasurementHardwareIdentity(_StrictRawModel):
    """Self-hashed identity for the actual eight-B300 Modal CUDA allocation."""

    schema_version: Literal["inkling-measurement-hardware-identity-v1"]
    backend: Literal["CUDA"]
    logical_devices: tuple[
        Literal["cuda:0"],
        Literal["cuda:1"],
        Literal["cuda:2"],
        Literal["cuda:3"],
        Literal["cuda:4"],
        Literal["cuda:5"],
        Literal["cuda:6"],
        Literal["cuda:7"],
    ]
    gpus: tuple[
        MeasurementHardwareGpu,
        MeasurementHardwareGpu,
        MeasurementHardwareGpu,
        MeasurementHardwareGpu,
        MeasurementHardwareGpu,
        MeasurementHardwareGpu,
        MeasurementHardwareGpu,
        MeasurementHardwareGpu,
    ]
    peer_topology: MeasurementCudaPeerTopology
    cuda_driver_path: StrictStr
    cuda_runtime_preflight: MeasurementCudaRuntimePreflight
    precision: Literal["model-native-subject-precision"]
    gpu_layers: Literal["all"]
    cpu_moe_layers: Literal[0]
    cpu_fallback: Literal[False]
    identity_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)

    @field_validator("cuda_driver_path")
    @classmethod
    def driver_path_is_canonical(cls, value: str) -> str:
        return _canonical_absolute_posix_path(value)

    @model_validator(mode="after")
    def identity_is_exact_and_self_hashed(self) -> MeasurementHardwareIdentity:
        if tuple(gpu.cuda_ordinal for gpu in self.gpus) != tuple(range(8)):
            raise ValueError("hardware GPUs must use exact CUDA ordinal order 0..7")
        if len({gpu.uuid.lower() for gpu in self.gpus}) != 8:
            raise ValueError("hardware GPU UUIDs must be unique")
        if tuple(gpu.uuid.lower() for gpu in self.gpus) != tuple(
            uuid.lower() for uuid in self.peer_topology.gpu_uuids
        ):
            raise ValueError("hardware GPU inventory differs from peer topology")
        driver_versions = {gpu.driver_version for gpu in self.gpus}
        if len(driver_versions) != 1:
            raise ValueError("hardware GPU driver versions differ")
        without_hash = self.model_dump(mode="json", exclude={"identity_sha256"})
        expected = hashlib.sha256(canonical_measurement_raw_json_bytes(without_hash)).hexdigest()
        if self.identity_sha256 != expected:
            raise ValueError("hardware identity SHA-256 differs from its full contents")
        return self


class MeasurementLlamaBenchCase(_StrictRawModel):
    """Five measured samples for one pinned llama-bench workload."""

    case: MeasurementBenchCase
    build_commit: StrictStr = Field(min_length=7, max_length=40)
    test_time_utc: StrictStr = Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
    model_path: StrictStr
    model_type: StrictStr = Field(min_length=1)
    model_size_bytes: StrictInt = Field(gt=0)
    model_parameter_count: StrictInt = Field(gt=0)
    prompt_tokens: StrictInt = Field(ge=0)
    generated_tokens: StrictInt = Field(ge=0)
    sample_nanoseconds: tuple[StrictInt, StrictInt, StrictInt, StrictInt, StrictInt]
    sample_tokens_per_second: tuple[StrictFloat, StrictFloat, StrictFloat, StrictFloat, StrictFloat]
    average_nanoseconds: StrictInt = Field(gt=0)
    standard_deviation_nanoseconds: StrictInt = Field(ge=0)
    average_tokens_per_second: StrictFloat = Field(gt=0.0)
    standard_deviation_tokens_per_second: StrictFloat = Field(ge=0.0)
    gpu_info: StrictStr = Field(min_length=1)
    backends: Literal["CUDA"]

    @field_validator("model_path")
    @classmethod
    def model_path_is_canonical(cls, value: str) -> str:
        return _canonical_absolute_posix_path(value)

    @model_validator(mode="after")
    def samples_and_summaries_are_derived(self) -> MeasurementLlamaBenchCase:
        expected_shape = {
            "pp512": (512, 0),
            "pp2048": (2048, 0),
            "tg128": (0, 128),
        }[self.case]
        if (self.prompt_tokens, self.generated_tokens) != expected_shape:
            raise ValueError("llama-bench case token counts differ from the protocol")
        if any(value <= 0 for value in self.sample_nanoseconds):
            raise ValueError("llama-bench nanosecond samples must be positive")
        token_count = self.prompt_tokens + self.generated_tokens
        derived = tuple(1_000_000_000.0 * token_count / value for value in self.sample_nanoseconds)
        serialized = tuple(float(format(value, ".6g")) for value in derived)
        if self.sample_tokens_per_second != serialized:
            raise ValueError("llama-bench sample rates differ from nanosecond samples")
        if self.average_nanoseconds != _cpp_uint64_average(self.sample_nanoseconds):
            raise ValueError("llama-bench average nanoseconds differs from samples")
        if self.standard_deviation_nanoseconds != _cpp_uint64_stdev(self.sample_nanoseconds):
            raise ValueError("llama-bench nanosecond deviation differs from samples")
        expected_mean = float(format(_ordered_mean(derived), ".6f"))
        expected_stdev = float(format(_cpp_double_stdev(derived), ".6f"))
        if self.average_tokens_per_second != expected_mean:
            raise ValueError("llama-bench average throughput differs from samples")
        if self.standard_deviation_tokens_per_second != expected_stdev:
            raise ValueError("llama-bench throughput deviation differs from samples")
        return self


def _ordered_mean(values: Sequence[float]) -> float:
    total = 0.0
    for value in values:
        total += value
    return total / len(values)


def _cpp_uint64_average(values: Sequence[int]) -> int:
    total = 0
    for value in values:
        total = (total + value) & _UINT64_MASK
    return total // len(values)


def _cpp_uint64_stdev(values: Sequence[int]) -> int:
    mean = _cpp_uint64_average(values)
    square_sum = 0
    for value in values:
        square_sum = (square_sum + ((value * value) & _UINT64_MASK)) & _UINT64_MASK
    divisor = len(values) - 1
    mean_square_times_count = ((mean * mean) & _UINT64_MASK) * len(values)
    variance = (
        square_sum // divisor - (mean_square_times_count & _UINT64_MASK) // divisor
    ) & _UINT64_MASK
    return int(math.sqrt(variance))


def _cpp_double_stdev(values: Sequence[float]) -> float:
    mean = _ordered_mean(values)
    square_sum = 0.0
    for value in values:
        square_sum += value * value
    divisor = len(values) - 1
    variance = square_sum / divisor - mean * mean * len(values) / divisor
    if variance < 0.0:
        raise ValueError("llama-bench sample variance is negative")
    return math.sqrt(variance)


class MeasurementLlamaBenchTrials(MeasurementProcessTimings):
    """One model load shared by the exact three ordered benchmark cases."""

    command: tuple[StrictStr, ...] = Field(min_length=1)
    cases: tuple[
        MeasurementLlamaBenchCase,
        MeasurementLlamaBenchCase,
        MeasurementLlamaBenchCase,
    ]
    stdout_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    stdout_size_bytes: StrictInt = Field(ge=0)
    stderr_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    stderr_size_bytes: StrictInt = Field(ge=0)
    warmup_enabled: Literal[True]
    single_model_load: Literal[True]

    @model_validator(mode="after")
    def case_order_and_model_are_exact(self) -> MeasurementLlamaBenchTrials:
        if tuple(item.case for item in self.cases) != MEASUREMENT_BENCH_CASE_ORDER:
            raise ValueError("llama-bench cases must be pp512, pp2048, tg128")
        if len({item.model_path for item in self.cases}) != 1:
            raise ValueError("llama-bench cases must share one model")
        return self


class _MeasurementServerResponse(_StrictRawModel):
    """Shared fields for one exact 128-token server response."""

    request_body_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    token_ids: tuple[StrictInt, ...] = Field(min_length=128, max_length=128)
    output_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    response_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    request_started_monotonic_seconds: StrictFloat = Field(gt=0.0)
    first_token_monotonic_seconds: StrictFloat = Field(gt=0.0)
    last_token_monotonic_seconds: StrictFloat = Field(gt=0.0)
    request_finished_monotonic_seconds: StrictFloat = Field(gt=0.0)
    wall_seconds: StrictFloat = Field(gt=0.0)
    ttft_seconds: StrictFloat = Field(gt=0.0)
    prompt_n: Literal[512]
    predicted_n: Literal[128]
    prompt_ms: StrictFloat = Field(gt=0.0)
    predicted_ms: StrictFloat = Field(gt=0.0)
    prompt_tokens_per_second: StrictFloat = Field(gt=0.0)
    decode_tokens_per_second: StrictFloat = Field(gt=0.0)
    inter_token_latency_p50_seconds: StrictFloat = Field(ge=0.0)
    inter_token_latency_p95_seconds: StrictFloat = Field(ge=0.0)
    inter_token_latency_p99_seconds: StrictFloat = Field(ge=0.0)
    raw_inter_token_latency_seconds: tuple[StrictFloat, ...] = Field(
        min_length=127,
        max_length=127,
    )
    prompt_text_recorded: Literal[False]
    output_text_recorded: Literal[False]

    @model_validator(mode="after")
    def request_boundaries_and_latencies_are_derived(
        self,
    ) -> _MeasurementServerResponse:
        if any(token_id < 0 for token_id in self.token_ids):
            raise ValueError("server token IDs must be nonnegative")
        if self.predicted_n != len(self.token_ids):
            raise ValueError("server predicted_n differs from retained token IDs")
        start = self.request_started_monotonic_seconds
        first = self.first_token_monotonic_seconds
        last = self.last_token_monotonic_seconds
        finish = self.request_finished_monotonic_seconds
        if not start < first <= last <= finish:
            raise ValueError("server request monotonic boundaries are not ordered")
        if not _float_equal(self.wall_seconds, finish - start):
            raise ValueError("server wall time differs from monotonic boundaries")
        if not _float_equal(self.ttft_seconds, first - start):
            raise ValueError("server TTFT differs from first-token boundary")
        prompt_rate = 1000.0 * self.prompt_n / self.prompt_ms
        decode_rate = 1000.0 * self.predicted_n / self.predicted_ms
        if not _float_equal(
            self.prompt_tokens_per_second,
            prompt_rate,
            rel_tol=5e-6,
            abs_tol=5e-6,
        ):
            raise ValueError("server prompt rate differs from count and time")
        if not _float_equal(
            self.decode_tokens_per_second,
            decode_rate,
            rel_tol=5e-6,
            abs_tol=5e-6,
        ):
            raise ValueError("server decode rate differs from count and time")
        intervals = self.raw_inter_token_latency_seconds
        if any(value < 0.0 for value in intervals):
            raise ValueError("server inter-token intervals must be nonnegative")
        if not _float_equal(sum(intervals), last - first, rel_tol=1e-7, abs_tol=1e-8):
            raise ValueError("server inter-token intervals differ from token boundaries")
        expected = (
            _r7_percentile(intervals, 50.0),
            _r7_percentile(intervals, 95.0),
            _r7_percentile(intervals, 99.0),
        )
        observed = (
            self.inter_token_latency_p50_seconds,
            self.inter_token_latency_p95_seconds,
            self.inter_token_latency_p99_seconds,
        )
        if any(
            not _float_equal(item, target) for item, target in zip(observed, expected, strict=True)
        ):
            raise ValueError("server request latency percentiles differ from raw ITLs")
        return self


class MeasurementServerRequest(_MeasurementServerResponse):
    """One indexed request inside a concurrent server batch."""

    request_index: StrictInt = Field(gt=0, le=4)


class MeasurementServerSingleWarmup(_MeasurementServerResponse):
    """One of the two single-request warm-ups shared by all cells."""

    warmup_index: Literal[1, 2]


class MeasurementServerBatch(_StrictRawModel):
    """One concurrent warm-up or measured server batch."""

    batch_index: StrictInt = Field(ge=0, le=5)
    concurrency: Literal[1, 2, 4]
    batch_started_monotonic_seconds: StrictFloat = Field(gt=0.0)
    batch_finished_monotonic_seconds: StrictFloat = Field(gt=0.0)
    batch_wall_seconds: StrictFloat = Field(gt=0.0)
    decode_boundary: Literal["earliest_first_token_to_latest_last_token_127_intervals_per_request"]
    aggregate_decode_token_intervals: StrictInt = Field(gt=0, le=508)
    batch_duration_seconds: StrictFloat = Field(gt=0.0)
    aggregate_decode_tokens_per_second: StrictFloat = Field(gt=0.0)
    requests: tuple[MeasurementServerRequest, ...] = Field(
        min_length=1,
        max_length=4,
    )

    @model_validator(mode="after")
    def batch_is_derived(self) -> MeasurementServerBatch:
        if len(self.requests) != self.concurrency:
            raise ValueError("server batch request count differs from concurrency")
        if tuple(item.request_index for item in self.requests) != tuple(
            range(1, self.concurrency + 1)
        ):
            raise ValueError("server request indexes are incomplete or out of order")
        start = self.batch_started_monotonic_seconds
        finish = self.batch_finished_monotonic_seconds
        if start >= finish or not _float_equal(self.batch_wall_seconds, finish - start):
            raise ValueError("server batch wall time differs from its boundaries")
        if any(
            request.request_started_monotonic_seconds < start
            or request.request_finished_monotonic_seconds > finish
            for request in self.requests
        ):
            raise ValueError("server request lies outside its batch boundaries")
        decode_start = min(request.first_token_monotonic_seconds for request in self.requests)
        decode_finish = max(request.last_token_monotonic_seconds for request in self.requests)
        duration = decode_finish - decode_start
        if not _float_equal(self.batch_duration_seconds, duration):
            raise ValueError("server batch decode duration differs from token boundaries")
        expected_intervals = 127 * self.concurrency
        if self.aggregate_decode_token_intervals != expected_intervals:
            raise ValueError("aggregate decode interval count differs from raw responses")
        expected_rate = expected_intervals / duration
        if not _float_equal(
            self.aggregate_decode_tokens_per_second,
            expected_rate,
        ):
            raise ValueError("aggregate server throughput differs from decode duration")
        return self


class MeasurementResourceSampleSummary(_StrictRawModel):
    """Resource maxima from samples in a declared inclusive telemetry window."""

    window_started_monotonic_seconds: StrictFloat = Field(gt=0.0)
    window_finished_monotonic_seconds: StrictFloat = Field(gt=0.0)
    sample_count: StrictInt = Field(gt=0)
    max_sampled_host_rss_bytes: StrictInt = Field(gt=0)
    max_sampled_per_gpu_memory_bytes: tuple[
        StrictInt,
        StrictInt,
        StrictInt,
        StrictInt,
        StrictInt,
        StrictInt,
        StrictInt,
        StrictInt,
    ]
    max_sampled_per_gpu_utilization_percent: tuple[
        StrictFloat,
        StrictFloat,
        StrictFloat,
        StrictFloat,
        StrictFloat,
        StrictFloat,
        StrictFloat,
        StrictFloat,
    ]

    @model_validator(mode="after")
    def window_and_ranges_are_valid(self) -> MeasurementResourceSampleSummary:
        if self.window_started_monotonic_seconds >= self.window_finished_monotonic_seconds:
            raise ValueError("resource window boundaries are not increasing")
        if any(value <= 0 for value in self.max_sampled_per_gpu_memory_bytes):
            raise ValueError("per-GPU sampled memory maxima must be positive")
        if any(not 0.0 <= value <= 100.0 for value in self.max_sampled_per_gpu_utilization_percent):
            raise ValueError("per-GPU utilization must be from 0 through 100")
        return self


class MeasurementServerCellBatchMetrics(_StrictRawModel):
    """Five-batch distributions for every published server timing/rate metric.

    Each end-to-end sample is the arithmetic mean of the retained request wall
    times in one measured batch. The other request-level samples use the same
    within-batch arithmetic-mean rule. Aggregate throughput and inter-token
    percentiles are computed directly for each retained batch.
    """

    mean_request_end_to_end_latency_seconds: MeasurementFiveBatchMetricSummary
    mean_ttft_seconds: MeasurementFiveBatchMetricSummary
    mean_prompt_tokens_per_second: MeasurementFiveBatchMetricSummary
    mean_decode_tokens_per_second: MeasurementFiveBatchMetricSummary
    aggregate_decode_tokens_per_second: MeasurementFiveBatchMetricSummary
    inter_token_latency_p50_seconds: MeasurementFiveBatchNonnegativeMetricSummary
    inter_token_latency_p95_seconds: MeasurementFiveBatchNonnegativeMetricSummary
    inter_token_latency_p99_seconds: MeasurementFiveBatchNonnegativeMetricSummary


def _summarize_five_batch_samples(
    values: Sequence[float],
) -> MeasurementFiveBatchMetricSummary:
    if len(values) != 5:
        raise ValueError("server batch metric requires exactly five samples")
    samples = cast(
        "tuple[float, float, float, float, float]",
        tuple(values),
    )
    return MeasurementFiveBatchMetricSummary(
        trial_count=5,
        samples=samples,
        mean=statistics.fmean(samples),
        median=statistics.median(samples),
        sample_standard_deviation=statistics.stdev(samples),
    )


def _summarize_five_batch_nonnegative_samples(
    values: Sequence[float],
) -> MeasurementFiveBatchNonnegativeMetricSummary:
    if len(values) != 5:
        raise ValueError("server batch metric requires exactly five samples")
    samples = cast(
        "tuple[float, float, float, float, float]",
        tuple(values),
    )
    return MeasurementFiveBatchNonnegativeMetricSummary(
        trial_count=5,
        samples=samples,
        mean=statistics.fmean(samples),
        median=statistics.median(samples),
        sample_standard_deviation=statistics.stdev(samples),
    )


class MeasurementServerCell(_StrictRawModel):
    """Warm-up protocol and five measured batches for one concurrency."""

    concurrency: Literal[1, 2, 4]
    single_request_warmups_completed: Literal[0, 2]
    concurrent_batch_warmup_completed: Literal[True]
    concurrent_batch_warmup: MeasurementServerBatch
    warmup_output_token_counts: tuple[StrictInt, ...] = Field(max_length=2)
    concurrent_warmup_request_count: Literal[1, 2, 4]
    measured_batches: tuple[
        MeasurementServerBatch,
        MeasurementServerBatch,
        MeasurementServerBatch,
        MeasurementServerBatch,
        MeasurementServerBatch,
    ]
    measured_request_count: StrictInt = Field(gt=0)
    mean_ttft_seconds: StrictFloat = Field(gt=0.0)
    mean_prompt_tokens_per_second: StrictFloat = Field(gt=0.0)
    mean_decode_tokens_per_second: StrictFloat = Field(gt=0.0)
    aggregate_decode_tokens_per_second_trials: tuple[
        StrictFloat, StrictFloat, StrictFloat, StrictFloat, StrictFloat
    ]
    mean_aggregate_decode_tokens_per_second: StrictFloat = Field(gt=0.0)
    inter_token_latency_method: Literal[
        "r7_linear_interpolation_over_all_measured_request_intervals"
    ]
    raw_inter_token_interval_count: StrictInt = Field(gt=0)
    inter_token_latency_p50_seconds: StrictFloat = Field(ge=0.0)
    inter_token_latency_p95_seconds: StrictFloat = Field(ge=0.0)
    inter_token_latency_p99_seconds: StrictFloat = Field(ge=0.0)
    resource_sample_summary: MeasurementResourceSampleSummary

    @model_validator(mode="after")
    def cell_counts_and_rollups_are_derived(self) -> MeasurementServerCell:
        expected_single = 2 if self.concurrency == 1 else 0
        expected_output_counts: tuple[int, ...] = (128, 128) if self.concurrency == 1 else ()
        if self.single_request_warmups_completed != expected_single:
            raise ValueError("single-request warmups may be credited only to c1")
        if self.warmup_output_token_counts != expected_output_counts:
            raise ValueError("single-request warmup output counts differ from protocol")
        warmup = self.concurrent_batch_warmup
        if (
            warmup.batch_index != 0
            or warmup.concurrency != self.concurrency
            or self.concurrent_warmup_request_count != self.concurrency
        ):
            raise ValueError("concurrent warm-up differs from its server cell")
        if tuple(batch.batch_index for batch in self.measured_batches) != (1, 2, 3, 4, 5):
            raise ValueError("measured server batches must use indexes 1 through 5")
        if any(batch.concurrency != self.concurrency for batch in self.measured_batches):
            raise ValueError("measured server batch concurrency differs from its cell")
        requests = tuple(request for batch in self.measured_batches for request in batch.requests)
        if len(requests) != 5 * self.concurrency:
            raise ValueError("server cell does not contain five batches of requests")
        if self.measured_request_count != len(requests):
            raise ValueError("measured request count differs from retained requests")
        intervals = tuple(
            value for request in requests for value in request.raw_inter_token_latency_seconds
        )
        if (
            self.raw_inter_token_interval_count != len(intervals)
            or len(intervals) != 5 * self.concurrency * 127
        ):
            raise ValueError("server cell did not retain every measured ITL")
        expected_values = {
            "mean_ttft_seconds": statistics.fmean(request.ttft_seconds for request in requests),
            "mean_prompt_tokens_per_second": statistics.fmean(
                request.prompt_tokens_per_second for request in requests
            ),
            "mean_decode_tokens_per_second": statistics.fmean(
                request.decode_tokens_per_second for request in requests
            ),
            "mean_aggregate_decode_tokens_per_second": statistics.fmean(
                batch.aggregate_decode_tokens_per_second for batch in self.measured_batches
            ),
            "inter_token_latency_p50_seconds": _r7_percentile(intervals, 50.0),
            "inter_token_latency_p95_seconds": _r7_percentile(intervals, 95.0),
            "inter_token_latency_p99_seconds": _r7_percentile(intervals, 99.0),
        }
        if self.aggregate_decode_tokens_per_second_trials != tuple(
            batch.aggregate_decode_tokens_per_second for batch in self.measured_batches
        ):
            raise ValueError("aggregate decode trials differ from measured batches")
        for field_name, expected in expected_values.items():
            if not _float_equal(getattr(self, field_name), expected):
                raise ValueError(f"{field_name} differs from retained server trials")
        measured_start = min(
            batch.batch_started_monotonic_seconds for batch in self.measured_batches
        )
        measured_finish = max(
            batch.batch_finished_monotonic_seconds for batch in self.measured_batches
        )
        if (
            self.resource_sample_summary.window_started_monotonic_seconds > measured_start
            or self.resource_sample_summary.window_finished_monotonic_seconds < measured_finish
        ):
            raise ValueError("resource window does not enclose all measured batches")
        return self


def recompute_server_cell_batch_metrics(
    cell: MeasurementServerCell,
) -> MeasurementServerCellBatchMetrics:
    """Derive five batch-level samples and their mean, median, and sample SD."""

    return MeasurementServerCellBatchMetrics(
        mean_request_end_to_end_latency_seconds=_summarize_five_batch_samples(
            tuple(
                statistics.fmean(request.wall_seconds for request in batch.requests)
                for batch in cell.measured_batches
            )
        ),
        mean_ttft_seconds=_summarize_five_batch_samples(
            tuple(
                statistics.fmean(request.ttft_seconds for request in batch.requests)
                for batch in cell.measured_batches
            )
        ),
        mean_prompt_tokens_per_second=_summarize_five_batch_samples(
            tuple(
                statistics.fmean(request.prompt_tokens_per_second for request in batch.requests)
                for batch in cell.measured_batches
            )
        ),
        mean_decode_tokens_per_second=_summarize_five_batch_samples(
            tuple(
                statistics.fmean(request.decode_tokens_per_second for request in batch.requests)
                for batch in cell.measured_batches
            )
        ),
        aggregate_decode_tokens_per_second=_summarize_five_batch_samples(
            tuple(batch.aggregate_decode_tokens_per_second for batch in cell.measured_batches)
        ),
        inter_token_latency_p50_seconds=_summarize_five_batch_nonnegative_samples(
            tuple(
                _r7_percentile(
                    tuple(
                        value
                        for request in batch.requests
                        for value in request.raw_inter_token_latency_seconds
                    ),
                    50.0,
                )
                for batch in cell.measured_batches
            )
        ),
        inter_token_latency_p95_seconds=_summarize_five_batch_nonnegative_samples(
            tuple(
                _r7_percentile(
                    tuple(
                        value
                        for request in batch.requests
                        for value in request.raw_inter_token_latency_seconds
                    ),
                    95.0,
                )
                for batch in cell.measured_batches
            )
        ),
        inter_token_latency_p99_seconds=_summarize_five_batch_nonnegative_samples(
            tuple(
                _r7_percentile(
                    tuple(
                        value
                        for request in batch.requests
                        for value in request.raw_inter_token_latency_seconds
                    ),
                    99.0,
                )
                for batch in cell.measured_batches
            )
        ),
    )


class MeasurementColdCacheConditioning(_StrictRawModel):
    """File-level advisory cache conditioning before the first server load."""

    schema_version: Literal["inkling-measurement-cold-cache-conditioning-v1"]
    method: Literal["file_level_posix_fadvise_posix_fadv_dontneed_on_all_staged_gguf_files"]
    advice: Literal["POSIX_FADV_DONTNEED"]
    staged_paths: tuple[StrictStr, ...] = Field(min_length=50, max_length=50)
    artifact_count: Literal[50]
    advised_bytes: StrictInt = Field(gt=0)
    completed_monotonic_seconds: StrictFloat = Field(gt=0.0)
    all_advice_calls_succeeded: Literal[True]
    global_cache_flush_claimed: Literal[False]

    @field_validator("staged_paths")
    @classmethod
    def paths_are_canonical(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            _canonical_absolute_posix_path(value)
        if len(set(values)) != len(values):
            raise ValueError("cold-cache conditioning paths must be unique")
        return values


class MeasurementColdServerLoad(_StrictRawModel):
    """Readiness-only first load with strict artifact and CUDA offload evidence."""

    schema_version: Literal["inkling-measurement-cold-server-load-v1"]
    command: tuple[StrictStr, ...] = Field(min_length=1, max_length=64)
    process_id: StrictInt = Field(gt=0)
    process_started_monotonic_seconds: StrictFloat = Field(gt=0.0)
    server_ready_monotonic_seconds: StrictFloat = Field(gt=0.0)
    process_finished_monotonic_seconds: StrictFloat = Field(gt=0.0)
    cold_server_process_load_seconds: StrictFloat = Field(gt=0.0)
    hardware_identity_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    readiness_only: Literal[True]
    generation_requests_executed: Literal[0]
    log: StrictStr = Field(min_length=1, max_length=32 * 1024 * 1024)
    log_size_bytes: StrictInt = Field(gt=0)
    log_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    loader_offload: LoaderOffloadEvidence
    artifact_load: ArtifactLoadEvidence

    @field_validator("log")
    @classmethod
    def log_has_no_nul(cls, value: str) -> str:
        if "\x00" in value:
            raise ValueError("cold server log must not contain NUL")
        return value

    @model_validator(mode="after")
    def cold_load_is_exact(self) -> MeasurementColdServerLoad:
        start = self.process_started_monotonic_seconds
        ready = self.server_ready_monotonic_seconds
        finish = self.process_finished_monotonic_seconds
        if not start < ready < finish:
            raise ValueError("cold server process boundaries are not increasing")
        if not _float_equal(self.cold_server_process_load_seconds, ready - start):
            raise ValueError("cold server load time differs from ready boundary")
        payload = self.log.encode("utf-8", errors="strict")
        if self.log_size_bytes != len(payload):
            raise ValueError("cold server log size differs from full UTF-8 bytes")
        if self.log_sha256 != hashlib.sha256(payload).hexdigest():
            raise ValueError("cold server log SHA-256 differs from full UTF-8 bytes")
        if parse_loader_offload_evidence(self.log, expected_gpu_count=8) != self.loader_offload:
            raise ValueError("cold server CUDA loader evidence differs from its full log")
        parsed_artifacts = parse_artifact_load_evidence(
            self.log,
            expected_first_shard_path=self.artifact_load.first_shard_path,
            expected_projector_path=self.artifact_load.projector_path,
        )
        if parsed_artifacts != self.artifact_load:
            raise ValueError("cold server artifact-load evidence differs from its full log")
        return self


class MeasurementServerLoadObservation(_StrictRawModel):
    """Full log provenance and loader proof for one process-to-readiness load."""

    command: tuple[StrictStr, ...] = Field(min_length=1, max_length=64)
    process_id: StrictInt = Field(gt=0)
    process_started_monotonic_seconds: StrictFloat = Field(gt=0.0)
    server_ready_monotonic_seconds: StrictFloat = Field(gt=0.0)
    process_finished_monotonic_seconds: StrictFloat = Field(gt=0.0)
    process_load_seconds: StrictFloat = Field(gt=0.0)
    log: StrictStr = Field(min_length=1, max_length=32 * 1024 * 1024)
    log_size_bytes: StrictInt = Field(gt=0)
    log_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    loader_offload: LoaderOffloadEvidence
    artifact_load: ArtifactLoadEvidence

    @field_validator("log")
    @classmethod
    def log_has_no_nul(cls, value: str) -> str:
        if "\x00" in value:
            raise ValueError("load-trial server log must not contain NUL")
        return value

    @model_validator(mode="after")
    def load_is_derived_from_boundaries(self) -> MeasurementServerLoadObservation:
        start = self.process_started_monotonic_seconds
        ready = self.server_ready_monotonic_seconds
        finish = self.process_finished_monotonic_seconds
        if not start < ready < finish:
            raise ValueError("load-trial process boundaries are not increasing")
        if not _float_equal(self.process_load_seconds, ready - start):
            raise ValueError("load-trial duration differs from its ready boundary")
        if any(not argument or "\x00" in argument for argument in self.command):
            raise ValueError("load-trial command contains an empty or NUL argument")
        payload = self.log.encode("utf-8", errors="strict")
        if self.log_size_bytes != len(payload):
            raise ValueError("load-trial server log size differs from full UTF-8 bytes")
        if self.log_sha256 != hashlib.sha256(payload).hexdigest():
            raise ValueError("load-trial server log SHA-256 differs from full UTF-8 bytes")
        if parse_loader_offload_evidence(self.log, expected_gpu_count=8) != self.loader_offload:
            raise ValueError("load-trial CUDA loader evidence differs from its full log")
        parsed_artifacts = parse_artifact_load_evidence(
            self.log,
            expected_first_shard_path=self.artifact_load.first_shard_path,
            expected_projector_path=self.artifact_load.projector_path,
        )
        if parsed_artifacts != self.artifact_load:
            raise ValueError("load-trial artifact-load evidence differs from its full log")
        return self


class MeasurementServerLoadPairTrial(_StrictRawModel):
    """One cache-conditioned cold load followed by its exact warm load."""

    trial_index: StrictInt = Field(gt=0, le=64)
    cold_cache_conditioning: MeasurementColdCacheConditioning
    cold: MeasurementServerLoadObservation
    warm: MeasurementServerLoadObservation
    warm_load_is_next_model_load_after_cold: Literal[True]
    explicit_cache_conditioning_or_eviction_requested_between_loads: Literal[False]
    cold_to_warm_restart_gap_seconds: StrictFloat = Field(ge=0.0)

    @model_validator(mode="after")
    def pair_is_exact_and_ordered(self) -> MeasurementServerLoadPairTrial:
        if self.cold.command != self.warm.command:
            raise ValueError("paired cold and warm load commands differ")
        if (
            self.cold.loader_offload != self.warm.loader_offload
            or self.cold.artifact_load != self.warm.artifact_load
        ):
            raise ValueError("paired cold and warm loader evidence differs")
        if self.cold.process_id == self.warm.process_id:
            raise ValueError("paired cold and warm loads must use distinct processes")
        if (
            self.cold_cache_conditioning.completed_monotonic_seconds
            >= self.cold.process_started_monotonic_seconds
        ):
            raise ValueError("load-pair conditioning must finish before the cold load")
        if (
            self.cold.process_finished_monotonic_seconds
            >= self.warm.process_started_monotonic_seconds
        ):
            raise ValueError("paired warm load must start after the cold process terminates")
        expected_gap = (
            self.warm.process_started_monotonic_seconds
            - self.cold.process_finished_monotonic_seconds
        )
        if not _float_equal(self.cold_to_warm_restart_gap_seconds, expected_gap):
            raise ValueError("load-pair restart gap differs from process boundaries")
        return self


class MeasurementServerTrials(_StrictRawModel):
    """Repeated cold/warm loads followed by the selected warm workload server."""

    schema_version: Literal["inkling-measurement-subject-server-v1"]
    subject: MeasurementRawSubject
    load_pair_repetitions: StrictInt = Field(ge=2, le=64)
    load_pair_trial_scope: Literal[
        "process_start_to_readiness_for_ordered_same_artifact_cold_then_warm_pairs"
    ]
    load_pair_trials: tuple[MeasurementServerLoadPairTrial, ...] = Field(
        min_length=2,
        max_length=64,
    )
    cold_server_load_trials: MeasurementRepeatedLoadDurations
    warm_server_load_trials: MeasurementRepeatedLoadDurations
    workload_load_pair_trial_index: StrictInt = Field(gt=0, le=64)
    cold_cache_conditioning: MeasurementColdCacheConditioning
    cold_load: MeasurementColdServerLoad
    warm_load_is_next_model_load_after_cold: Literal[True]
    explicit_cache_conditioning_or_eviction_requested_between_server_loads: Literal[False]
    cold_to_warm_restart_gap_seconds: StrictFloat = Field(ge=0.0)
    command: tuple[StrictStr, ...] = Field(min_length=1)
    process_id: StrictInt = Field(gt=0)
    process_started_monotonic_seconds: StrictFloat = Field(gt=0.0)
    server_ready_monotonic_seconds: StrictFloat = Field(gt=0.0)
    process_finished_monotonic_seconds: StrictFloat = Field(gt=0.0)
    warm_server_process_load_seconds: StrictFloat = Field(gt=0.0)
    vocab_size: StrictInt = Field(gt=0)
    diagnostic_items_completed_before_performance: Literal[64]
    diagnostic_repetitions: Literal[1]
    single_request_warmups: tuple[
        MeasurementServerSingleWarmup,
        MeasurementServerSingleWarmup,
    ]
    prompt_token_ids: tuple[StrictInt, ...] = Field(min_length=512, max_length=512)
    prompt_token_ids_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    prompt_token_count: Literal[512]
    output_tokens: Literal[128]
    seed: Literal[42]
    temperature: StrictFloat
    streaming: Literal[True]
    cache_prompt: Literal[False]
    return_tokens: Literal[True]
    ignore_eos: Literal[True]
    request_body_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    concurrency: tuple[
        MeasurementServerCell,
        MeasurementServerCell,
        MeasurementServerCell,
    ]
    log_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    log_size_bytes: StrictInt = Field(gt=0)
    prompt_text_recorded: Literal[False]
    output_text_recorded: Literal[False]

    @model_validator(mode="after")
    def server_protocol_is_exact(self) -> MeasurementServerTrials:
        if len(self.load_pair_trials) != self.load_pair_repetitions:
            raise ValueError("load-pair count differs from retained trial records")
        if tuple(trial.trial_index for trial in self.load_pair_trials) != tuple(
            range(1, self.load_pair_repetitions + 1)
        ):
            raise ValueError("load-pair trial indexes must be contiguous from one")
        process_ids = tuple(
            process_id
            for trial in self.load_pair_trials
            for process_id in (trial.cold.process_id, trial.warm.process_id)
        )
        if len(set(process_ids)) != 2 * self.load_pair_repetitions:
            raise ValueError("repeated load trials must use distinct process IDs")
        if any(
            later.cold_cache_conditioning.completed_monotonic_seconds
            <= earlier.warm.process_finished_monotonic_seconds
            for earlier, later in pairwise(self.load_pair_trials)
        ):
            raise ValueError("next cold-cache conditioning must follow the prior warm termination")
        if any(
            trial.cold.command != self.command or trial.warm.command != self.command
            for trial in self.load_pair_trials
        ):
            raise ValueError("repeated load command differs from the workload server")
        if (
            self.cold_server_load_trials.trial_count != self.load_pair_repetitions
            or self.warm_server_load_trials.trial_count != self.load_pair_repetitions
        ):
            raise ValueError("load-pair repetition count differs from retained durations")
        if self.cold_server_load_trials.durations_seconds != tuple(
            trial.cold.process_load_seconds for trial in self.load_pair_trials
        ) or self.warm_server_load_trials.durations_seconds != tuple(
            trial.warm.process_load_seconds for trial in self.load_pair_trials
        ):
            raise ValueError("load-duration arrays differ from retained trial records")
        if self.workload_load_pair_trial_index != self.load_pair_repetitions:
            raise ValueError("the final load pair must supply the warm workload server")
        workload_load_index = self.workload_load_pair_trial_index - 1
        workload_trial = self.load_pair_trials[workload_load_index]
        if workload_trial.cold_cache_conditioning != self.cold_cache_conditioning:
            raise ValueError("selected load-pair conditioning differs from the full receipt")
        selected_cold = workload_trial.cold
        if (
            selected_cold.command != self.cold_load.command
            or selected_cold.process_id != self.cold_load.process_id
            or selected_cold.process_started_monotonic_seconds
            != self.cold_load.process_started_monotonic_seconds
            or selected_cold.server_ready_monotonic_seconds
            != self.cold_load.server_ready_monotonic_seconds
            or selected_cold.process_finished_monotonic_seconds
            != self.cold_load.process_finished_monotonic_seconds
            or not _float_equal(
                selected_cold.process_load_seconds,
                self.cold_load.cold_server_process_load_seconds,
            )
            or selected_cold.log != self.cold_load.log
            or selected_cold.log_size_bytes != self.cold_load.log_size_bytes
            or selected_cold.log_sha256 != self.cold_load.log_sha256
            or selected_cold.loader_offload != self.cold_load.loader_offload
            or selected_cold.artifact_load != self.cold_load.artifact_load
        ):
            raise ValueError("selected compact cold load differs from its full evidence")
        selected_warm = workload_trial.warm
        if (
            selected_warm.command != self.command
            or selected_warm.process_id != self.process_id
            or selected_warm.process_started_monotonic_seconds
            != self.process_started_monotonic_seconds
            or selected_warm.server_ready_monotonic_seconds != self.server_ready_monotonic_seconds
            or selected_warm.process_finished_monotonic_seconds
            != self.process_finished_monotonic_seconds
            or not _float_equal(
                selected_warm.process_load_seconds,
                self.warm_server_process_load_seconds,
            )
            or selected_warm.log_size_bytes != self.log_size_bytes
            or selected_warm.log_sha256 != self.log_sha256
        ):
            raise ValueError("selected compact warm load differs from its full evidence")
        if not _float_equal(
            self.cold_load.cold_server_process_load_seconds,
            self.cold_server_load_trials.durations_seconds[workload_load_index],
        ):
            raise ValueError("selected cold load differs from its repeated load trial")
        if not _float_equal(
            self.warm_server_process_load_seconds,
            self.warm_server_load_trials.durations_seconds[workload_load_index],
        ):
            raise ValueError("selected warm load differs from its repeated load trial")
        start = self.process_started_monotonic_seconds
        ready = self.server_ready_monotonic_seconds
        finish = self.process_finished_monotonic_seconds
        if not start < ready < finish:
            raise ValueError("server process boundaries are not increasing")
        if not _float_equal(self.warm_server_process_load_seconds, ready - start):
            raise ValueError("warm server load time differs from ready boundary")
        if self.cold_load.command != self.command:
            raise ValueError("cold and warm server commands must be identical")
        if self.cold_load.process_id == self.process_id:
            raise ValueError("cold and warm server loads must use distinct processes")
        if self.cold_load.process_finished_monotonic_seconds >= start:
            raise ValueError("warm server must start after the cold server terminates")
        expected_gap = start - self.cold_load.process_finished_monotonic_seconds
        if not _float_equal(self.cold_to_warm_restart_gap_seconds, expected_gap):
            raise ValueError("cold-to-warm restart gap differs from process boundaries")
        if (
            self.cold_cache_conditioning.completed_monotonic_seconds
            >= self.cold_load.process_started_monotonic_seconds
        ):
            raise ValueError("cold-cache conditioning must finish before the cold load starts")
        if tuple(item.warmup_index for item in self.single_request_warmups) != (1, 2):
            raise ValueError("single-request warmups must use indexes one and two")
        if any(token_id < 0 for token_id in self.prompt_token_ids):
            raise ValueError("server prompt token IDs must be nonnegative")
        expected_prompt_hash = hashlib.sha256(
            canonical_measurement_raw_json_bytes(list(self.prompt_token_ids))
        ).hexdigest()
        if self.prompt_token_ids_sha256 != expected_prompt_hash:
            raise ValueError("server prompt-token SHA-256 differs from token IDs")
        if self.temperature != 0.0:
            raise ValueError("server benchmark temperature must be exactly zero")
        expected_request_body_hash = hashlib.sha256(
            canonical_measurement_raw_json_bytes(
                {
                    "prompt": list(self.prompt_token_ids),
                    "seed": self.seed,
                    "temperature": self.temperature,
                    "n_predict": self.output_tokens,
                    "stream": self.streaming,
                    "cache_prompt": self.cache_prompt,
                    "return_tokens": self.return_tokens,
                    "ignore_eos": self.ignore_eos,
                }
            )
        ).hexdigest()
        if self.request_body_sha256 != expected_request_body_hash:
            raise ValueError("server request-body SHA-256 differs from the exact protocol")
        if (
            tuple(item.concurrency for item in self.concurrency)
            != MEASUREMENT_SERVER_CONCURRENCY_ORDER
        ):
            raise ValueError("server cells must use concurrency order 1, 2, 4")
        responses = (
            *self.single_request_warmups,
            *(
                request
                for cell in self.concurrency
                for batch in (cell.concurrent_batch_warmup, *cell.measured_batches)
                for request in batch.requests
            ),
        )
        if any(response.request_body_sha256 != self.request_body_sha256 for response in responses):
            raise ValueError("server response differs from the exact request protocol")
        all_boundaries = [
            boundary
            for cell in self.concurrency
            for batch in (cell.concurrent_batch_warmup, *cell.measured_batches)
            for boundary in (
                batch.batch_started_monotonic_seconds,
                batch.batch_finished_monotonic_seconds,
            )
        ]
        if any(boundary < ready or boundary > finish for boundary in all_boundaries):
            raise ValueError("server workload lies outside the server process lifetime")
        return self


class MeasurementRawTrialsEvidence(_StrictRawModel):
    """All non-telemetry raw trials for one subject and accepted attempt."""

    schema_version: Literal["inkling-measurement-raw-trials-v1"]
    bindings: MeasurementAttemptBindings
    hardware_identity: MeasurementHardwareIdentity
    staging: MeasurementSubjectStaging
    perplexity: MeasurementPerplexityTrial
    diagnostics: tuple[MeasurementDiagnosticItem, ...] = Field(
        min_length=64,
        max_length=64,
    )
    llama_bench: MeasurementLlamaBenchTrials
    server: MeasurementServerTrials
    prompt_text_recorded: Literal[False]
    output_text_recorded: Literal[False]

    @model_validator(mode="after")
    def all_trials_are_complete_and_bound(self) -> MeasurementRawTrialsEvidence:
        subject = self.bindings.subject
        if self.staging.subject != subject or self.server.subject != subject:
            raise MeasurementRawEvidenceBindingError("raw trials contain another subject")
        conditioned_artifacts = tuple(
            artifact
            for artifact in self.staging.artifacts
            if artifact.staged_path.endswith(".gguf")
        )
        if len(conditioned_artifacts) != 50:
            raise ValueError("cold-cache conditioning requires 49 shards and one projector")
        expected_conditioning_paths = tuple(
            artifact.staged_path for artifact in conditioned_artifacts
        )
        expected_conditioning_bytes = sum(artifact.size_bytes for artifact in conditioned_artifacts)
        for load_pair_trial in self.server.load_pair_trials:
            conditioning = load_pair_trial.cold_cache_conditioning
            if conditioning.staged_paths != expected_conditioning_paths:
                raise ValueError(
                    "cold-cache conditioning paths differ from the ordered staged GGUF inventory"
                )
            if conditioning.advised_bytes != expected_conditioning_bytes:
                raise ValueError("cold-cache conditioning bytes differ from staged GGUF sizes")
        projectors = tuple(
            artifact
            for artifact in conditioned_artifacts
            if artifact.source_path.endswith("/mmproj/mmproj-BF16.gguf")
        )
        if len(projectors) != 1:
            raise ValueError("cold-cache conditioning inventory must contain one projector")
        cold_artifacts = self.server.cold_load.artifact_load
        if (
            cold_artifacts.first_shard_path != conditioned_artifacts[0].staged_path
            or cold_artifacts.projector_path != projectors[0].staged_path
        ):
            raise ValueError("cold server artifact evidence differs from staged GGUF paths")
        if self.server.cold_load.hardware_identity_sha256 != self.hardware_identity.identity_sha256:
            raise MeasurementRawEvidenceBindingError(
                "cold server hardware identity differs from the accepted allocation"
            )
        expected_items = tuple(
            (f"{suite}_{index:02d}", suite)
            for suite in MEASUREMENT_QUALITY_SUITE_ORDER
            for index in range(1, 9)
        )
        observed_items = tuple((item.item_id, item.suite) for item in self.diagnostics)
        if observed_items != expected_items:
            raise ValueError("diagnostics must be all 64 items in reviewed order")
        cursor = self.server.server_ready_monotonic_seconds
        for item in self.diagnostics:
            diagnostic_trial = item.trials[0]
            if diagnostic_trial.request_started_monotonic_seconds < cursor:
                raise ValueError("diagnostic requests overlap or are out of order")
            cursor = diagnostic_trial.request_finished_monotonic_seconds
        for warmup in self.server.single_request_warmups:
            if warmup.request_started_monotonic_seconds < cursor:
                raise ValueError("single-request warmups overlap or precede diagnostics")
            cursor = warmup.request_finished_monotonic_seconds
        for cell in self.server.concurrency:
            for batch in (cell.concurrent_batch_warmup, *cell.measured_batches):
                if batch.batch_started_monotonic_seconds < cursor:
                    raise ValueError("server batches overlap or are out of protocol order")
                cursor = batch.batch_finished_monotonic_seconds
        if cursor > self.server.process_finished_monotonic_seconds:
            raise ValueError("server workload extends beyond the warm-server lifetime")
        process_ids = (
            self.perplexity.process_id,
            *(
                process_id
                for trial in self.server.load_pair_trials
                for process_id in (trial.cold.process_id, trial.warm.process_id)
            ),
            self.llama_bench.process_id,
        )
        if len(set(process_ids)) != 2 * self.server.load_pair_repetitions + 2:
            raise ValueError(
                "perplexity, repeated server loads, and bench must use distinct processes"
            )
        _validate_executed_command_scope(self)
        return self

    @property
    def hardware_identity_sha256(self) -> str:
        """Return the validated self-hash for terminal and subject binding."""

        return self.hardware_identity.identity_sha256


def _command_option_integer(command: tuple[str, ...], option: str) -> int:
    indexes = tuple(index for index, argument in enumerate(command) if argument == option)
    if len(indexes) != 1 or indexes[0] + 1 >= len(command):
        raise ValueError(f"command must contain exactly one {option} value")
    raw_value = command[indexes[0] + 1]
    if not raw_value.isascii() or not raw_value.isdecimal():
        raise ValueError(f"command {option} value must be a base-10 integer")
    return int(raw_value)


def _validate_executed_command_scope(
    raw_trials: MeasurementRawTrialsEvidence,
) -> None:
    """Bind retained argv and llama-bench identity to the reviewed protocol."""

    artifacts = raw_trials.staging.artifacts
    model_path = artifacts[0].staged_path
    projectors = tuple(
        artifact.staged_path
        for artifact in artifacts
        if artifact.source_path.endswith("/mmproj/mmproj-BF16.gguf")
    )
    if len(projectors) != 1:
        raise ValueError("staging must contain one exact multimodal projector")
    projector_path = projectors[0]
    topology = bind_exact_cuda_topology(
        tuple(f"CUDA{ordinal}" for ordinal in range(8)),
        (1,) * 8,
    )

    expected_perplexity = build_llama_perplexity_command(
        LlamaPerplexityCommandSpec(
            model_path=model_path,
            corpus_path=MEASUREMENT_REMOTE_CORPUS_PATH,
            context_size=512,
            batch_size=512,
            ubatch_size=512,
            chunks=64,
            topology=topology,
        )
    )
    if raw_trials.perplexity.command != expected_perplexity:
        raise ValueError("perplexity command differs from the reviewed protocol")

    server_port = _command_option_integer(raw_trials.server.command, "--port")
    expected_server = build_llama_server_command(
        LlamaServerCommandSpec(
            model_path=model_path,
            projector_path=projector_path,
            context_size=8192,
            batch_size=2048,
            ubatch_size=512,
            parallel_slots=4,
            port=server_port,
            topology=topology,
        )
    )
    if raw_trials.server.command != expected_server:
        raise ValueError("server command differs from the reviewed protocol")

    expected_bench = build_llama_bench_command(
        LlamaBenchCommandSpec(
            model_path=model_path,
            repetitions=5,
            batch_size=2048,
            ubatch_size=512,
            threads=16,
            topology=topology,
        )
    )
    if raw_trials.llama_bench.command != expected_bench:
        raise ValueError("llama-bench command differs from the reviewed protocol")

    bench_cases = raw_trials.llama_bench.cases
    if any(case.build_commit != PINNED_LLAMA_CPP_BUILD_COMMIT for case in bench_cases):
        raise ValueError("llama-bench build commit differs from the pinned runtime")
    if any(case.model_path != model_path for case in bench_cases):
        raise ValueError("llama-bench model path differs from staged subject")
    if (
        len({case.model_size_bytes for case in bench_cases}) != 1
        or len({case.model_parameter_count for case in bench_cases}) != 1
    ):
        raise ValueError("llama-bench cases report different loaded model identities")
    expected_gpu_info = ", ".join(gpu.name for gpu in raw_trials.hardware_identity.gpus)
    if any(case.gpu_info != expected_gpu_info for case in bench_cases):
        raise ValueError("llama-bench GPU report differs from the exact eight-B300 cell")


class MeasurementTelemetryGpu(_StrictRawModel):
    """One ordered CUDA-ordinal resource sample."""

    cuda_ordinal: StrictInt = Field(ge=0, le=7)
    uuid: StrictStr = Field(pattern=_GPU_UUID_PATTERN)
    memory_used_mib: StrictInt = Field(ge=0)
    utilization_percent: StrictInt = Field(ge=0, le=100)


class MeasurementResourceTelemetryRow(_StrictRawModel):
    """One canonical JSONL telemetry sample bound to a process and attempt."""

    schema_version: Literal["inkling-measurement-resource-telemetry-row-v1"]
    bindings: MeasurementAttemptBindings
    workload: MeasurementRawWorkload
    process_id: StrictInt = Field(gt=0)
    sample_index: StrictInt = Field(ge=0)
    requested_sampling_interval_seconds: StrictFloat
    sampled_at_monotonic_seconds: StrictFloat = Field(gt=0.0)
    host_rss_bytes: StrictInt = Field(ge=0)
    gpus: tuple[
        MeasurementTelemetryGpu,
        MeasurementTelemetryGpu,
        MeasurementTelemetryGpu,
        MeasurementTelemetryGpu,
        MeasurementTelemetryGpu,
        MeasurementTelemetryGpu,
        MeasurementTelemetryGpu,
        MeasurementTelemetryGpu,
    ]

    @model_validator(mode="after")
    def gpu_order_is_exact(self) -> MeasurementResourceTelemetryRow:
        if self.requested_sampling_interval_seconds != 1.0:
            raise ValueError("requested telemetry sampling interval must be exactly 1.0 seconds")
        if tuple(gpu.cuda_ordinal for gpu in self.gpus) != tuple(range(8)):
            raise ValueError("telemetry GPUs must use CUDA ordinal order 0..7")
        if len({gpu.uuid.lower() for gpu in self.gpus}) != 8:
            raise ValueError("telemetry GPU UUIDs must be unique")
        return self


class MeasurementResourceTelemetryEvidence(_StrictRawModel):
    """All telemetry rows for the exact three ordered subject processes."""

    requested_sampling_interval_seconds: StrictFloat
    rows: tuple[MeasurementResourceTelemetryRow, ...] = Field(min_length=3)

    @model_validator(mode="after")
    def rows_are_one_bound_ordered_attempt(
        self,
    ) -> MeasurementResourceTelemetryEvidence:
        binding = self.rows[0].bindings
        if self.requested_sampling_interval_seconds != 1.0 or any(
            row.requested_sampling_interval_seconds != self.requested_sampling_interval_seconds
            for row in self.rows
        ):
            raise ValueError("telemetry rows differ from the requested 1.0-second interval")
        if any(row.bindings != binding for row in self.rows):
            raise MeasurementRawEvidenceBindingError(
                "telemetry rows have incompatible attempt bindings"
            )
        grouped: list[list[MeasurementResourceTelemetryRow]] = []
        cursor = 0
        for workload in MEASUREMENT_RAW_WORKLOAD_ORDER:
            group: list[MeasurementResourceTelemetryRow] = []
            while cursor < len(self.rows) and self.rows[cursor].workload == workload:
                group.append(self.rows[cursor])
                cursor += 1
            if not group:
                raise ValueError(f"telemetry lacks workload {workload}")
            grouped.append(group)
        if cursor != len(self.rows):
            raise ValueError("telemetry workload groups are out of checked order")
        process_ids: list[int] = []
        expected_uuids: tuple[str, ...] | None = None
        for group in grouped:
            process_id = group[0].process_id
            process_ids.append(process_id)
            if any(row.process_id != process_id for row in group):
                raise ValueError("one telemetry workload spans multiple processes")
            if tuple(row.sample_index for row in group) != tuple(range(len(group))):
                raise ValueError("telemetry sample indexes must be contiguous from zero")
            timestamps = tuple(row.sampled_at_monotonic_seconds for row in group)
            if any(later <= earlier for earlier, later in pairwise(timestamps)):
                raise ValueError("telemetry timestamps must increase per process")
            for row in group:
                uuids = tuple(gpu.uuid.lower() for gpu in row.gpus)
                if expected_uuids is None:
                    expected_uuids = uuids
                elif uuids != expected_uuids:
                    raise ValueError("telemetry GPU UUID inventory changed")
        if len(set(process_ids)) != 3:
            raise ValueError("telemetry workloads must use distinct processes")
        return self


class MeasurementBackendAuditWorkload(_StrictRawModel):
    """Full UTF-8 audit log for one measured workload."""

    workload: MeasurementRawWorkload
    process_id: StrictInt = Field(gt=0)
    command: tuple[StrictStr, ...] = Field(min_length=1)
    capture_mode: Literal["captured_stdout_stderr", "combined_server_log"]
    stdout_stderr_delimiter: StrictStr | None
    log: StrictStr
    log_size_bytes: StrictInt = Field(gt=0)
    log_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)

    @field_validator("log")
    @classmethod
    def log_is_full_strict_utf8(cls, value: str) -> str:
        if not value or "\x00" in value:
            raise ValueError("backend audit log must be non-empty text without NUL")
        try:
            value.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            raise ValueError("backend audit log contains an invalid Unicode surrogate") from error
        return value

    @model_validator(mode="after")
    def capture_and_hash_are_exact(self) -> MeasurementBackendAuditWorkload:
        is_server = self.workload == "server_quality_and_performance"
        expected_mode = "combined_server_log" if is_server else "captured_stdout_stderr"
        if self.capture_mode != expected_mode:
            raise ValueError("backend audit capture mode differs from workload")
        if is_server:
            if self.stdout_stderr_delimiter is not None:
                raise ValueError("server audit must not declare a capture delimiter")
            if CAPTURED_TOOL_LOG_DELIMITER in self.log:
                raise ValueError("server audit unexpectedly contains tool delimiter")
        elif (
            self.stdout_stderr_delimiter != CAPTURED_TOOL_LOG_DELIMITER
            or self.log.count(CAPTURED_TOOL_LOG_DELIMITER) != 1
        ):
            raise ValueError("captured tool audit must contain its delimiter exactly once")
        payload = self.log.encode("utf-8", errors="strict")
        if self.log_size_bytes != len(payload):
            raise ValueError("backend audit log size differs from full UTF-8 bytes")
        if self.log_sha256 != hashlib.sha256(payload).hexdigest():
            raise ValueError("backend audit log SHA-256 differs from full UTF-8 bytes")
        if any(not argument or "\x00" in argument for argument in self.command):
            raise ValueError("backend audit command contains an empty or NUL argument")
        return self


class MeasurementBackendAuditEvidence(_StrictRawModel):
    """Full audit logs for the exact three ordered workload processes."""

    schema_version: Literal["inkling-measurement-backend-audit-v1"]
    bindings: MeasurementAttemptBindings
    workloads: tuple[
        MeasurementBackendAuditWorkload,
        MeasurementBackendAuditWorkload,
        MeasurementBackendAuditWorkload,
    ]

    @model_validator(mode="after")
    def workload_order_is_exact(self) -> MeasurementBackendAuditEvidence:
        if tuple(item.workload for item in self.workloads) != MEASUREMENT_RAW_WORKLOAD_ORDER:
            raise ValueError("backend audit workloads must use the checked order")
        if len({item.process_id for item in self.workloads}) != 3:
            raise ValueError("backend audit workloads must use distinct processes")
        return self


MeasurementParsedRawEvidence: TypeAlias = (
    MeasurementTokenNllEvidence
    | MeasurementRawTrialsEvidence
    | MeasurementResourceTelemetryEvidence
    | MeasurementBackendAuditEvidence
)


def parse_token_nll_raw_evidence(payload: bytes) -> MeasurementTokenNllEvidence:
    """Parse exactly 16,320 canonical ordered token-NLL JSONL rows."""

    parsed = _parse_canonical_jsonl(
        payload,
        kind="token_nll",
        model=MeasurementTokenNllRow,
    )
    if len(parsed) != MEASUREMENT_TOKEN_NLL_RECORD_COUNT:
        raise MeasurementRawEvidenceCanonicalError(
            "token_nll evidence must contain exactly 16,320 rows"
        )
    return MeasurementTokenNllEvidence(rows=cast("tuple[MeasurementTokenNllRow, ...]", parsed))


def parse_raw_trials_evidence(payload: bytes) -> MeasurementRawTrialsEvidence:
    """Parse one canonical complete raw-trials object."""

    return cast(
        "MeasurementRawTrialsEvidence",
        _parse_canonical_object(
            payload,
            kind="raw_trials",
            model=MeasurementRawTrialsEvidence,
        ),
    )


def parse_resource_telemetry_evidence(
    payload: bytes,
) -> MeasurementResourceTelemetryEvidence:
    """Parse canonical telemetry JSONL for all three ordered processes."""

    parsed = _parse_canonical_jsonl(
        payload,
        kind="resource_telemetry",
        model=MeasurementResourceTelemetryRow,
    )
    return MeasurementResourceTelemetryEvidence(
        requested_sampling_interval_seconds=1.0,
        rows=cast("tuple[MeasurementResourceTelemetryRow, ...]", parsed),
    )


def parse_backend_audit_evidence(payload: bytes) -> MeasurementBackendAuditEvidence:
    """Parse the canonical full-log backend-audit object."""

    return cast(
        "MeasurementBackendAuditEvidence",
        _parse_canonical_object(
            payload,
            kind="backend_audit",
            model=MeasurementBackendAuditEvidence,
        ),
    )


def parse_measurement_raw_evidence(
    payload: bytes,
    *,
    kind: MeasurementRawEvidenceKind,
) -> MeasurementParsedRawEvidence:
    """Dispatch one raw blob through its approved canonical parser."""

    if kind == "token_nll":
        return parse_token_nll_raw_evidence(payload)
    if kind == "raw_trials":
        return parse_raw_trials_evidence(payload)
    if kind == "resource_telemetry":
        return parse_resource_telemetry_evidence(payload)
    return parse_backend_audit_evidence(payload)


class MeasurementTokenNllSummary(_StrictRawModel):
    """Recomputed NLL and perplexity from the complete token rows."""

    scored_tokens: Literal[16320]
    mean_nll: StrictFloat = Field(ge=0.0)
    computed_perplexity: StrictFloat = Field(gt=0.0)


class MeasurementDiagnosticSuiteSummary(_StrictRawModel):
    """Recomputed score for one exact eight-item diagnostic suite."""

    suite: MeasurementQualitySuite
    item_count: Literal[8]
    correct_items: StrictInt = Field(ge=0, le=8)
    accuracy: StrictFloat = Field(ge=0.0, le=1.0)


class MeasurementSubjectQualitySummary(_StrictRawModel):
    """Recomputed single-subject quality values from raw evidence."""

    subject: MeasurementRawSubject
    token_nll: MeasurementTokenNllSummary
    printed_perplexity: StrictFloat = Field(gt=0.0)
    printed_perplexity_uncertainty: StrictFloat = Field(ge=0.0)
    printed_perplexity_absolute_tolerance: StrictFloat
    diagnostic_items: Literal[64]
    correct_items: StrictInt = Field(ge=0, le=64)
    overall_accuracy: StrictFloat = Field(ge=0.0, le=1.0)
    suites: tuple[
        MeasurementDiagnosticSuiteSummary,
        MeasurementDiagnosticSuiteSummary,
        MeasurementDiagnosticSuiteSummary,
        MeasurementDiagnosticSuiteSummary,
        MeasurementDiagnosticSuiteSummary,
        MeasurementDiagnosticSuiteSummary,
        MeasurementDiagnosticSuiteSummary,
        MeasurementDiagnosticSuiteSummary,
    ]

    @field_validator("printed_perplexity_absolute_tolerance")
    @classmethod
    def printed_perplexity_tolerance_is_exact(cls, value: float) -> float:
        if value != MEASUREMENT_PRINTED_PERPLEXITY_ABSOLUTE_TOLERANCE:
            raise ValueError("printed perplexity tolerance differs from the reviewed protocol")
        return value


class MeasurementBenchCaseSummary(_StrictRawModel):
    """Recomputed mean, median, and sample deviation for one bench case."""

    case: MeasurementBenchCase
    sample_count: Literal[5]
    average_tokens_per_second: StrictFloat = Field(gt=0.0)
    median_tokens_per_second: StrictFloat = Field(gt=0.0)
    standard_deviation_tokens_per_second: StrictFloat = Field(ge=0.0)


class MeasurementServerCellSummary(_StrictRawModel):
    """Recomputed performance and resource values for one server cell."""

    concurrency: Literal[1, 2, 4]
    measured_batches: Literal[5]
    measured_requests: StrictInt = Field(gt=0)
    mean_ttft_seconds: StrictFloat = Field(gt=0.0)
    mean_prompt_tokens_per_second: StrictFloat = Field(gt=0.0)
    mean_decode_tokens_per_second: StrictFloat = Field(gt=0.0)
    mean_aggregate_decode_tokens_per_second: StrictFloat = Field(gt=0.0)
    batch_metrics: MeasurementServerCellBatchMetrics
    inter_token_latency_p50_seconds: StrictFloat = Field(ge=0.0)
    inter_token_latency_p95_seconds: StrictFloat = Field(ge=0.0)
    inter_token_latency_p99_seconds: StrictFloat = Field(ge=0.0)
    resource_sample_summary: MeasurementResourceSampleSummary

    @model_validator(mode="after")
    def aggregate_fields_match_batch_distributions(
        self,
    ) -> MeasurementServerCellSummary:
        if self.measured_requests != self.measured_batches * self.concurrency:
            raise ValueError("server summary request count differs from batches")
        aggregate_pairs = (
            (self.mean_ttft_seconds, self.batch_metrics.mean_ttft_seconds.mean),
            (
                self.mean_prompt_tokens_per_second,
                self.batch_metrics.mean_prompt_tokens_per_second.mean,
            ),
            (
                self.mean_decode_tokens_per_second,
                self.batch_metrics.mean_decode_tokens_per_second.mean,
            ),
            (
                self.mean_aggregate_decode_tokens_per_second,
                self.batch_metrics.aggregate_decode_tokens_per_second.mean,
            ),
        )
        if any(
            not _float_equal(aggregate, batch_mean) for aggregate, batch_mean in aggregate_pairs
        ):
            raise ValueError("server aggregate differs from its five batch samples")
        p50 = self.batch_metrics.inter_token_latency_p50_seconds.samples
        p95 = self.batch_metrics.inter_token_latency_p95_seconds.samples
        p99 = self.batch_metrics.inter_token_latency_p99_seconds.samples
        if any(
            not lower <= middle <= upper for lower, middle, upper in zip(p50, p95, p99, strict=True)
        ):
            raise ValueError("per-batch inter-token latency percentiles are not ordered")
        return self


class MeasurementSubjectPerformanceSummary(_StrictRawModel):
    """Recomputed single-subject deployment metrics from raw trials."""

    subject: MeasurementRawSubject
    text_checkpoint_size_bytes: StrictInt = Field(gt=0)
    multimodal_projector_size_bytes: StrictInt = Field(gt=0)
    executable_gguf_bundle_size_bytes: StrictInt = Field(gt=0)
    load_pair_repetitions: StrictInt = Field(ge=2, le=64)
    workload_load_pair_trial_index: StrictInt = Field(gt=0, le=64)
    cold_server_load_trials: MeasurementRepeatedLoadDurations
    warm_server_load_trials: MeasurementRepeatedLoadDurations
    cold_server_process_load_seconds: StrictFloat = Field(gt=0.0)
    warm_server_process_load_seconds: StrictFloat = Field(gt=0.0)
    bench_cases: tuple[
        MeasurementBenchCaseSummary,
        MeasurementBenchCaseSummary,
        MeasurementBenchCaseSummary,
    ]
    server_cells: tuple[
        MeasurementServerCellSummary,
        MeasurementServerCellSummary,
        MeasurementServerCellSummary,
    ]

    @model_validator(mode="after")
    def sizes_and_load_trials_are_exact(self) -> MeasurementSubjectPerformanceSummary:
        if self.executable_gguf_bundle_size_bytes != (
            self.text_checkpoint_size_bytes + self.multimodal_projector_size_bytes
        ):
            raise ValueError("executable GGUF bundle size differs from text plus projector")
        if (
            self.cold_server_load_trials.trial_count != self.load_pair_repetitions
            or self.warm_server_load_trials.trial_count != self.load_pair_repetitions
        ):
            raise ValueError("performance load count differs from retained load trials")
        if self.workload_load_pair_trial_index > self.load_pair_repetitions:
            raise ValueError("performance workload load-pair index exceeds repetitions")
        selected = self.workload_load_pair_trial_index - 1
        if not _float_equal(
            self.cold_server_process_load_seconds,
            self.cold_server_load_trials.durations_seconds[selected],
        ):
            raise ValueError("performance cold load differs from its selected trial")
        if not _float_equal(
            self.warm_server_process_load_seconds,
            self.warm_server_load_trials.durations_seconds[selected],
        ):
            raise ValueError("performance warm load differs from its selected trial")
        return self


class MeasurementPairingProjectionHashes(_StrictRawModel):
    """Subject-independent workload projections for strict paired comparison."""

    token_nll_pairing_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    diagnostic_pairing_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    performance_pairing_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)


def recompute_token_nll_summary(
    evidence: MeasurementTokenNllEvidence,
) -> MeasurementTokenNllSummary:
    """Compute mean NLL and exp(mean NLL) from all retained rows."""

    mean_nll = statistics.fmean(row.nll for row in evidence.rows)
    try:
        perplexity = math.exp(mean_nll)
    except OverflowError as error:
        raise MeasurementRawEvidenceError(
            "mean token NLL cannot produce finite perplexity"
        ) from error
    if not math.isfinite(perplexity):
        raise MeasurementRawEvidenceError("mean token NLL cannot produce finite perplexity")
    return MeasurementTokenNllSummary(
        scored_tokens=MEASUREMENT_TOKEN_NLL_RECORD_COUNT,
        mean_nll=mean_nll,
        computed_perplexity=perplexity,
    )


def recompute_subject_quality_summary(
    token_nll: MeasurementTokenNllEvidence,
    raw_trials: MeasurementRawTrialsEvidence,
) -> MeasurementSubjectQualitySummary:
    """Recompute printed-PPL agreement and every diagnostic score."""

    token_summary = recompute_token_nll_summary(token_nll)
    if (
        abs(raw_trials.perplexity.perplexity - token_summary.computed_perplexity)
        > MEASUREMENT_PRINTED_PERPLEXITY_ABSOLUTE_TOLERANCE
    ):
        raise MeasurementRawEvidenceError("printed perplexity differs from exp(mean token NLL)")
    suite_summaries: list[MeasurementDiagnosticSuiteSummary] = []
    for suite in MEASUREMENT_QUALITY_SUITE_ORDER:
        items = tuple(item for item in raw_trials.diagnostics if item.suite == suite)
        if len(items) != 8:
            raise MeasurementRawEvidenceError(
                f"diagnostic suite {suite} does not contain eight items"
            )
        correct = sum(int(item.score) for item in items)
        suite_summaries.append(
            MeasurementDiagnosticSuiteSummary(
                suite=suite,
                item_count=8,
                correct_items=correct,
                accuracy=correct / 8.0,
            )
        )
    correct_items = sum(item.correct_items for item in suite_summaries)
    return MeasurementSubjectQualitySummary(
        subject=raw_trials.bindings.subject,
        token_nll=token_summary,
        printed_perplexity=raw_trials.perplexity.perplexity,
        printed_perplexity_uncertainty=raw_trials.perplexity.uncertainty,
        printed_perplexity_absolute_tolerance=(MEASUREMENT_PRINTED_PERPLEXITY_ABSOLUTE_TOLERANCE),
        diagnostic_items=64,
        correct_items=correct_items,
        overall_accuracy=correct_items / 64.0,
        suites=cast(
            "tuple[MeasurementDiagnosticSuiteSummary, "
            "MeasurementDiagnosticSuiteSummary, "
            "MeasurementDiagnosticSuiteSummary, "
            "MeasurementDiagnosticSuiteSummary, "
            "MeasurementDiagnosticSuiteSummary, "
            "MeasurementDiagnosticSuiteSummary, "
            "MeasurementDiagnosticSuiteSummary, "
            "MeasurementDiagnosticSuiteSummary]",
            tuple(suite_summaries),
        ),
    )


def validate_measurement_diagnostic_evidence(
    diagnostic_items: tuple[DiagnosticItem, ...],
    *,
    prompt_template: str,
    prompt_interface: MeasurementPromptInterface,
    raw_trials: MeasurementRawTrialsEvidence,
) -> MeasurementRawTrialsEvidence:
    """Bind raw diagnostic results to the exact checked dataset and scorers."""

    if type(prompt_template) is not str or not prompt_template:
        raise MeasurementRawEvidenceBindingError(
            "checked diagnostic prompt template must be non-empty text"
        )
    if len(diagnostic_items) != MEASUREMENT_DIAGNOSTIC_ITEM_COUNT:
        raise MeasurementRawEvidenceBindingError(
            "checked diagnostic dataset must contain exactly 64 items"
        )

    for checked, observed in zip(
        diagnostic_items,
        raw_trials.diagnostics,
        strict=True,
    ):
        fixture = build_diagnostic_fixture_bytes(checked.fixture)
        render = render_measurement_diagnostic_prompt(
            prompt_template=prompt_template,
            item_prompt=checked.prompt,
            prompt_interface=prompt_interface,
            has_media=fixture is not None,
        )
        prompt = render.prompt_text.encode()
        expected_fixture_sha256 = None if fixture is None else hashlib.sha256(fixture).hexdigest()
        expected_fixture_size_bytes = None if fixture is None else len(fixture)
        request_prompt: object = render.prompt_text
        if fixture is not None:
            request_prompt = {
                "prompt_string": render.prompt_string,
                "multimodal_data": [base64.b64encode(fixture).decode("ascii")],
            }
        expected_request_body_sha256 = hashlib.sha256(
            canonical_measurement_raw_json_bytes(
                {
                    "prompt": request_prompt,
                    "seed": checked.seed,
                    "temperature": checked.temperature,
                    "n_predict": checked.max_new_tokens,
                    "stream": False,
                    "cache_prompt": False,
                    "return_tokens": True,
                    "timings_per_token": True,
                }
            )
        ).hexdigest()
        if (
            observed.item_id != checked.item_id
            or observed.suite != checked.suite
            or observed.modality != checked.modality
            or observed.request_body_sha256 != expected_request_body_sha256
            or observed.prompt_sha256 != hashlib.sha256(prompt).hexdigest()
            or observed.fixture_sha256 != expected_fixture_sha256
            or observed.fixture_size_bytes != expected_fixture_size_bytes
            or observed.seed != checked.seed
            or observed.temperature != checked.temperature
            or observed.max_new_tokens != checked.max_new_tokens
            or observed.scorer_kind != checked.scorer.kind
        ):
            raise MeasurementRawEvidenceBindingError(
                f"diagnostic item {checked.item_id} differs from the checked dataset"
            )

        expected_normalized_sha256 = diagnostic_expected_normalized_sha256(
            checked.scorer.kind,
            checked.scorer.expected,
        )
        trial = observed.trials[0]
        expected_score = (
            trial.normalization_succeeded and trial.normalized_sha256 == expected_normalized_sha256
        )
        if trial.score != expected_score:
            raise MeasurementRawEvidenceBindingError(
                f"diagnostic item {checked.item_id} score differs from its checked target"
            )

    return raw_trials


def recompute_bench_case_summary(
    case: MeasurementLlamaBenchCase,
) -> MeasurementBenchCaseSummary:
    """Compute mean, median, and sample deviation from five printed samples."""

    samples = case.sample_tokens_per_second
    return MeasurementBenchCaseSummary(
        case=case.case,
        sample_count=5,
        average_tokens_per_second=statistics.fmean(samples),
        median_tokens_per_second=statistics.median(samples),
        standard_deviation_tokens_per_second=statistics.stdev(samples),
    )


def recompute_subject_performance_summary(
    raw_trials: MeasurementRawTrialsEvidence,
) -> MeasurementSubjectPerformanceSummary:
    """Recompute bench and server rollups from retained measured trials."""

    text_checkpoint_size_bytes = sum(
        artifact.size_bytes for artifact in raw_trials.staging.artifacts[:49]
    )
    projectors = tuple(
        artifact
        for artifact in raw_trials.staging.artifacts
        if artifact.source_path.endswith("/mmproj/mmproj-BF16.gguf")
    )
    if len(projectors) != 1:
        raise MeasurementRawEvidenceError(
            "performance size summary requires exactly one multimodal projector"
        )
    multimodal_projector_size_bytes = projectors[0].size_bytes
    bench = tuple(recompute_bench_case_summary(case) for case in raw_trials.llama_bench.cases)
    cells = tuple(
        MeasurementServerCellSummary(
            concurrency=cell.concurrency,
            measured_batches=5,
            measured_requests=cell.measured_request_count,
            mean_ttft_seconds=cell.mean_ttft_seconds,
            mean_prompt_tokens_per_second=cell.mean_prompt_tokens_per_second,
            mean_decode_tokens_per_second=cell.mean_decode_tokens_per_second,
            mean_aggregate_decode_tokens_per_second=(cell.mean_aggregate_decode_tokens_per_second),
            batch_metrics=recompute_server_cell_batch_metrics(cell),
            inter_token_latency_p50_seconds=(cell.inter_token_latency_p50_seconds),
            inter_token_latency_p95_seconds=(cell.inter_token_latency_p95_seconds),
            inter_token_latency_p99_seconds=(cell.inter_token_latency_p99_seconds),
            resource_sample_summary=cell.resource_sample_summary,
        )
        for cell in raw_trials.server.concurrency
    )
    return MeasurementSubjectPerformanceSummary(
        subject=raw_trials.bindings.subject,
        text_checkpoint_size_bytes=text_checkpoint_size_bytes,
        multimodal_projector_size_bytes=multimodal_projector_size_bytes,
        executable_gguf_bundle_size_bytes=(
            text_checkpoint_size_bytes + multimodal_projector_size_bytes
        ),
        load_pair_repetitions=raw_trials.server.load_pair_repetitions,
        workload_load_pair_trial_index=(raw_trials.server.workload_load_pair_trial_index),
        cold_server_load_trials=raw_trials.server.cold_server_load_trials,
        warm_server_load_trials=raw_trials.server.warm_server_load_trials,
        cold_server_process_load_seconds=(
            raw_trials.server.cold_load.cold_server_process_load_seconds
        ),
        warm_server_process_load_seconds=(raw_trials.server.warm_server_process_load_seconds),
        bench_cases=cast(
            "tuple[MeasurementBenchCaseSummary, "
            "MeasurementBenchCaseSummary, "
            "MeasurementBenchCaseSummary]",
            bench,
        ),
        server_cells=cast(
            "tuple[MeasurementServerCellSummary, "
            "MeasurementServerCellSummary, "
            "MeasurementServerCellSummary]",
            cells,
        ),
    )


def recompute_telemetry_window(
    telemetry: MeasurementResourceTelemetryEvidence,
    *,
    workload: MeasurementRawWorkload,
    process_id: int,
    started_monotonic_seconds: float,
    finished_monotonic_seconds: float,
) -> MeasurementResourceSampleSummary:
    """Recompute sampled resource maxima for one declared measured window."""

    if (
        not math.isfinite(started_monotonic_seconds)
        or not math.isfinite(finished_monotonic_seconds)
        or started_monotonic_seconds >= finished_monotonic_seconds
    ):
        raise MeasurementRawEvidenceError(
            "telemetry window boundaries must be finite and increasing"
        )
    samples = tuple(
        row
        for row in telemetry.rows
        if row.workload == workload
        and row.process_id == process_id
        and started_monotonic_seconds
        <= row.sampled_at_monotonic_seconds
        <= finished_monotonic_seconds
    )
    if not samples:
        raise MeasurementRawEvidenceError("declared measured window contains no telemetry sample")
    return MeasurementResourceSampleSummary(
        window_started_monotonic_seconds=started_monotonic_seconds,
        window_finished_monotonic_seconds=finished_monotonic_seconds,
        sample_count=len(samples),
        max_sampled_host_rss_bytes=max(row.host_rss_bytes for row in samples),
        max_sampled_per_gpu_memory_bytes=cast(
            "tuple[int, int, int, int, int, int, int, int]",
            tuple(
                max(row.gpus[ordinal].memory_used_mib for row in samples) * 1024 * 1024
                for ordinal in range(8)
            ),
        ),
        max_sampled_per_gpu_utilization_percent=cast(
            "tuple[float, float, float, float, float, float, float, float]",
            tuple(
                float(max(row.gpus[ordinal].utilization_percent for row in samples))
                for ordinal in range(8)
            ),
        ),
    )


class MeasurementRawEvidenceLinks(_StrictRawModel):
    """Validated cross-blob bindings exposed to compact evidence builders."""

    run_id: StrictStr = Field(pattern=_RUN_ID_PATTERN)
    subject: MeasurementRawSubject
    hardware_identity_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    perplexity_process_id: StrictInt = Field(gt=0)
    cold_server_process_id: StrictInt = Field(gt=0)
    warm_server_process_id: StrictInt = Field(gt=0)
    llama_bench_process_id: StrictInt = Field(gt=0)


def validate_measurement_raw_evidence_links(
    token_nll: MeasurementTokenNllEvidence,
    raw_trials: MeasurementRawTrialsEvidence,
    telemetry: MeasurementResourceTelemetryEvidence,
    backend_audit: MeasurementBackendAuditEvidence,
) -> MeasurementRawEvidenceLinks:
    """Cross-check bindings, processes, logs, hardware, and telemetry rollups."""

    bindings = raw_trials.bindings
    token_nll_payload = b"".join(
        canonical_measurement_raw_json_bytes(row.model_dump(mode="json")) for row in token_nll.rows
    )
    if raw_trials.perplexity.token_nll_sha256 != hashlib.sha256(
        token_nll_payload
    ).hexdigest() or raw_trials.perplexity.token_nll_size_bytes != len(token_nll_payload):
        raise MeasurementRawEvidenceBindingError(
            "token-NLL blob differs from the perplexity trial binding"
        )
    if telemetry.rows[0].bindings != bindings or backend_audit.bindings != bindings:
        raise MeasurementRawEvidenceBindingError(
            "raw blobs do not share one exact accepted attempt"
        )
    process_by_workload = {
        "perplexity": raw_trials.perplexity.process_id,
        "server_quality_and_performance": raw_trials.server.process_id,
        "llama_bench": raw_trials.llama_bench.process_id,
    }
    for audit in backend_audit.workloads:
        expected_pid = process_by_workload[audit.workload]
        if audit.process_id != expected_pid:
            raise MeasurementRawEvidenceBindingError(
                f"{audit.workload} audit belongs to another process"
            )
        expected_command = {
            "perplexity": raw_trials.perplexity.command,
            "server_quality_and_performance": raw_trials.server.command,
            "llama_bench": raw_trials.llama_bench.command,
        }[audit.workload]
        if audit.command != expected_command:
            raise MeasurementRawEvidenceBindingError(
                f"{audit.workload} audit command differs from raw trials"
            )
        if audit.workload == "server_quality_and_performance":
            if (
                audit.log_sha256 != raw_trials.server.log_sha256
                or audit.log_size_bytes != raw_trials.server.log_size_bytes
            ):
                raise MeasurementRawEvidenceBindingError(
                    "server full log differs from raw-trials log binding"
                )
        else:
            stdout, stderr = audit.log.split(
                CAPTURED_TOOL_LOG_DELIMITER,
                maxsplit=1,
            )
            stdout_bytes = stdout.encode("utf-8")
            stderr_bytes = stderr.encode("utf-8")
            trial = (
                raw_trials.perplexity if audit.workload == "perplexity" else raw_trials.llama_bench
            )
            if (
                hashlib.sha256(stdout_bytes).hexdigest() != trial.stdout_sha256
                or len(stdout_bytes) != trial.stdout_size_bytes
                or hashlib.sha256(stderr_bytes).hexdigest() != trial.stderr_sha256
                or len(stderr_bytes) != trial.stderr_size_bytes
            ):
                raise MeasurementRawEvidenceBindingError(
                    f"{audit.workload} full log differs from stdout/stderr bindings"
                )
    telemetry_processes = {
        workload: {row.process_id for row in telemetry.rows if row.workload == workload}
        for workload in MEASUREMENT_RAW_WORKLOAD_ORDER
    }
    if any(
        values != {process_by_workload[workload]}
        for workload, values in telemetry_processes.items()
    ):
        raise MeasurementRawEvidenceBindingError("telemetry process scope differs from raw trials")
    hardware_uuids = tuple(gpu.uuid.lower() for gpu in raw_trials.hardware_identity.gpus)
    if any(tuple(gpu.uuid.lower() for gpu in row.gpus) != hardware_uuids for row in telemetry.rows):
        raise MeasurementRawEvidenceBindingError(
            "telemetry GPU inventory differs from hardware identity"
        )
    for cell in raw_trials.server.concurrency:
        observed = recompute_telemetry_window(
            telemetry,
            workload="server_quality_and_performance",
            process_id=raw_trials.server.process_id,
            started_monotonic_seconds=(
                cell.resource_sample_summary.window_started_monotonic_seconds
            ),
            finished_monotonic_seconds=(
                cell.resource_sample_summary.window_finished_monotonic_seconds
            ),
        )
        if observed != cell.resource_sample_summary:
            raise MeasurementRawEvidenceBindingError(
                f"c{cell.concurrency} resource sample summary differs from raw telemetry"
            )
    return MeasurementRawEvidenceLinks(
        run_id=bindings.run_id,
        subject=bindings.subject,
        hardware_identity_sha256=raw_trials.hardware_identity_sha256,
        perplexity_process_id=raw_trials.perplexity.process_id,
        cold_server_process_id=raw_trials.server.cold_load.process_id,
        warm_server_process_id=raw_trials.server.process_id,
        llama_bench_process_id=raw_trials.llama_bench.process_id,
    )


def _pairing_binding_projection(bindings: MeasurementAttemptBindings) -> dict[str, str]:
    return {
        "run_id": bindings.run_id,
        "reviewed_config_file_sha256": bindings.reviewed_config_file_sha256,
        "resolved_config_sha256": bindings.resolved_config_sha256,
        "protocol_sha256": bindings.protocol_sha256,
        "workload_sha256": bindings.workload_sha256,
        "launch_intent_sha256": bindings.launch_intent_sha256,
        "post_spawn_acceptance_sha256": bindings.post_spawn_acceptance_sha256,
        "call_id": bindings.call_id,
        "attempt_claim_sha256": bindings.attempt_claim_sha256,
    }


def _normalized_subject_command(
    command: tuple[str, ...],
    *,
    model_path: str,
    projector_path: str | None = None,
    normalize_server_port: bool = False,
) -> list[str]:
    replacements = {model_path: "<SUBJECT_MODEL>"}
    if projector_path is not None:
        replacements[projector_path] = "<SHARED_PROJECTOR>"
    normalized = [replacements.get(argument, argument) for argument in command]
    if normalize_server_port:
        indexes = tuple(index for index, argument in enumerate(normalized) if argument == "--port")
        if len(indexes) != 1 or indexes[0] + 1 >= len(normalized):
            raise ValueError("server command must contain one port for pairing")
        normalized[indexes[0] + 1] = "<LOOPBACK_PORT>"
    return normalized


def _staged_command_paths(
    raw_trials: MeasurementRawTrialsEvidence,
) -> tuple[str, str]:
    model_path = raw_trials.staging.artifacts[0].staged_path
    projectors = tuple(
        artifact.staged_path
        for artifact in raw_trials.staging.artifacts
        if artifact.source_path.endswith("/mmproj/mmproj-BF16.gguf")
    )
    if len(projectors) != 1:
        raise ValueError("staging must contain one projector for command pairing")
    return model_path, projectors[0]


def token_nll_pairing_projection_sha256(
    evidence: MeasurementTokenNllEvidence,
    raw_trials: MeasurementRawTrialsEvidence,
) -> str:
    """Hash subject-independent token positions and IDs; exclude measured NLL."""

    model_path, _ = _staged_command_paths(raw_trials)
    projection = {
        "command": _normalized_subject_command(
            raw_trials.perplexity.command,
            model_path=model_path,
        ),
        "corpus_reference_sha256": raw_trials.perplexity.corpus_reference_sha256,
        "corpus_sha256": raw_trials.perplexity.corpus_sha256,
        "corpus_size_bytes": raw_trials.perplexity.corpus_size_bytes,
        "tokens": [[row.chunk_index, row.token_index, row.token_id] for row in evidence.rows],
    }
    return _sha256_canonical(
        b"inkling-measurement-token-nll-pairing-v1\0",
        projection,
    )


def diagnostic_pairing_projection_sha256(
    raw_trials: MeasurementRawTrialsEvidence,
) -> str:
    """Hash the exact diagnostic inputs and decoding protocol, not outcomes."""

    model_path, projector_path = _staged_command_paths(raw_trials)
    projection = {
        "bindings": _pairing_binding_projection(raw_trials.bindings),
        "server_command": _normalized_subject_command(
            raw_trials.server.command,
            model_path=model_path,
            projector_path=projector_path,
            normalize_server_port=True,
        ),
        "diagnostics": [
            {
                "item_id": item.item_id,
                "suite": item.suite,
                "modality": item.modality,
                "request_body_sha256": item.request_body_sha256,
                "prompt_sha256": item.prompt_sha256,
                "fixture_sha256": item.fixture_sha256,
                "fixture_size_bytes": item.fixture_size_bytes,
                "seed": item.seed,
                "temperature": item.temperature,
                "max_new_tokens": item.max_new_tokens,
                "scorer_kind": item.scorer_kind,
                "trial_index": item.trials[0].trial_index,
            }
            for item in raw_trials.diagnostics
        ],
    }
    return _sha256_canonical(
        b"inkling-measurement-diagnostic-pairing-v1\0",
        projection,
    )


def performance_pairing_projection_sha256(
    raw_trials: MeasurementRawTrialsEvidence,
) -> str:
    """Hash workload identity, sample ordinals, and exact server prompt tokens."""

    model_path, projector_path = _staged_command_paths(raw_trials)
    projection = {
        "bindings": _pairing_binding_projection(raw_trials.bindings),
        "llama_bench_command": _normalized_subject_command(
            raw_trials.llama_bench.command,
            model_path=model_path,
        ),
        "server_command": _normalized_subject_command(
            raw_trials.server.command,
            model_path=model_path,
            projector_path=projector_path,
            normalize_server_port=True,
        ),
        "model_parameter_count": raw_trials.llama_bench.cases[0].model_parameter_count,
        "bench_cases": [
            {
                "case": case.case,
                "prompt_tokens": case.prompt_tokens,
                "generated_tokens": case.generated_tokens,
                "sample_indexes": list(range(1, 6)),
            }
            for case in raw_trials.llama_bench.cases
        ],
        "server": {
            "load_pair_repetitions": raw_trials.server.load_pair_repetitions,
            "load_pair_trial_scope": raw_trials.server.load_pair_trial_scope,
            "load_pair_trial_indexes": list(range(1, raw_trials.server.load_pair_repetitions + 1)),
            "workload_load_pair_trial_index": (raw_trials.server.workload_load_pair_trial_index),
            "cold_cache_conditioning": (raw_trials.server.cold_cache_conditioning.method),
            "cold_cache_advice": raw_trials.server.cold_cache_conditioning.advice,
            "conditioned_artifact_count": (
                raw_trials.server.cold_cache_conditioning.artifact_count
            ),
            "global_cache_flush_claimed": (
                raw_trials.server.cold_cache_conditioning.global_cache_flush_claimed
            ),
            "cold_load_readiness_only": raw_trials.server.cold_load.readiness_only,
            "cold_load_generation_requests_executed": (
                raw_trials.server.cold_load.generation_requests_executed
            ),
            "cold_load_command": _normalized_subject_command(
                raw_trials.server.cold_load.command,
                model_path=model_path,
                projector_path=projector_path,
                normalize_server_port=True,
            ),
            "warm_load_protocol": (
                "second_same_artifact_process_after_cold_termination_without_requested_cache_conditioning_or_eviction"
            ),
            "warm_load_is_next_model_load_after_cold": (
                raw_trials.server.warm_load_is_next_model_load_after_cold
            ),
            "explicit_cache_conditioning_or_eviction_requested_between_server_loads": (
                raw_trials.server.explicit_cache_conditioning_or_eviction_requested_between_server_loads
            ),
            "prompt_token_ids": list(raw_trials.server.prompt_token_ids),
            "prompt_token_ids_sha256": raw_trials.server.prompt_token_ids_sha256,
            "output_tokens": raw_trials.server.output_tokens,
            "seed": raw_trials.server.seed,
            "temperature": raw_trials.server.temperature,
            "streaming": raw_trials.server.streaming,
            "cache_prompt": raw_trials.server.cache_prompt,
            "return_tokens": raw_trials.server.return_tokens,
            "ignore_eos": raw_trials.server.ignore_eos,
            "request_body_sha256": raw_trials.server.request_body_sha256,
            "single_request_warmup_indexes": [1, 2],
            "cells": [
                {
                    "concurrency": cell.concurrency,
                    "concurrent_warmup_batch_index": (cell.concurrent_batch_warmup.batch_index),
                    "measured_batch_indexes": [
                        batch.batch_index for batch in cell.measured_batches
                    ],
                    "request_indexes": [
                        [request.request_index for request in batch.requests]
                        for batch in cell.measured_batches
                    ],
                }
                for cell in raw_trials.server.concurrency
            ],
        },
    }
    return _sha256_canonical(
        b"inkling-measurement-performance-pairing-v1\0",
        projection,
    )


def recompute_pairing_projection_hashes(
    token_nll: MeasurementTokenNllEvidence,
    raw_trials: MeasurementRawTrialsEvidence,
) -> MeasurementPairingProjectionHashes:
    """Return all three subject-independent pairing projection hashes."""

    return MeasurementPairingProjectionHashes(
        token_nll_pairing_sha256=token_nll_pairing_projection_sha256(
            token_nll,
            raw_trials,
        ),
        diagnostic_pairing_sha256=diagnostic_pairing_projection_sha256(raw_trials),
        performance_pairing_sha256=performance_pairing_projection_sha256(raw_trials),
    )


def validate_pairing_projection_hashes(
    baseline: MeasurementPairingProjectionHashes,
    candidate: MeasurementPairingProjectionHashes,
) -> None:
    """Fail if BF16 and Q3 did not use exactly equivalent paired workloads."""

    if baseline != candidate:
        differing = tuple(
            field
            for field in (
                "token_nll_pairing_sha256",
                "diagnostic_pairing_sha256",
                "performance_pairing_sha256",
            )
            if getattr(baseline, field) != getattr(candidate, field)
        )
        raise MeasurementRawEvidenceBindingError(
            "paired workload projections differ: " + ", ".join(differing)
        )


__all__ = [
    "CAPTURED_TOOL_LOG_DELIMITER",
    "MEASUREMENT_BENCH_CASE_ORDER",
    "MEASUREMENT_DIAGNOSTIC_ITEM_COUNT",
    "MEASUREMENT_QUALITY_SUITE_ORDER",
    "MEASUREMENT_RAW_EVIDENCE_KIND_ORDER",
    "MEASUREMENT_RAW_EVIDENCE_MAX_BYTES",
    "MEASUREMENT_RAW_WORKLOAD_ORDER",
    "MEASUREMENT_REMOTE_CORPUS_PATH",
    "MEASUREMENT_SERVER_CONCURRENCY_ORDER",
    "MEASUREMENT_TOKEN_NLL_RECORD_COUNT",
    "MeasurementAttemptBindings",
    "MeasurementBackendAuditEvidence",
    "MeasurementBenchCaseSummary",
    "MeasurementColdCacheConditioning",
    "MeasurementColdServerLoad",
    "MeasurementCudaRuntimeDeviceProbe",
    "MeasurementCudaRuntimePreflight",
    "MeasurementFiveBatchMetricSummary",
    "MeasurementFiveBatchNonnegativeMetricSummary",
    "MeasurementHardwareIdentity",
    "MeasurementPairingProjectionHashes",
    "MeasurementParsedRawEvidence",
    "MeasurementRawEvidenceBindingError",
    "MeasurementRawEvidenceCanonicalError",
    "MeasurementRawEvidenceError",
    "MeasurementRawEvidenceLinks",
    "MeasurementRawEvidenceSizeError",
    "MeasurementRawTrialsEvidence",
    "MeasurementRepeatedLoadDurations",
    "MeasurementResourceSampleSummary",
    "MeasurementResourceTelemetryEvidence",
    "MeasurementServerCellBatchMetrics",
    "MeasurementServerCellSummary",
    "MeasurementServerLoadObservation",
    "MeasurementServerLoadPairTrial",
    "MeasurementServerTrials",
    "MeasurementSubjectPerformanceSummary",
    "MeasurementSubjectQualitySummary",
    "MeasurementTokenNllEvidence",
    "MeasurementTokenNllSummary",
    "canonical_measurement_raw_json_bytes",
    "diagnostic_pairing_projection_sha256",
    "parse_backend_audit_evidence",
    "parse_measurement_raw_evidence",
    "parse_raw_trials_evidence",
    "parse_resource_telemetry_evidence",
    "parse_token_nll_raw_evidence",
    "performance_pairing_projection_sha256",
    "recompute_bench_case_summary",
    "recompute_pairing_projection_hashes",
    "recompute_server_cell_batch_metrics",
    "recompute_subject_performance_summary",
    "recompute_subject_quality_summary",
    "recompute_telemetry_window",
    "recompute_token_nll_summary",
    "token_nll_pairing_projection_sha256",
    "validate_measurement_diagnostic_evidence",
    "validate_measurement_raw_evidence_links",
    "validate_pairing_projection_hashes",
]
