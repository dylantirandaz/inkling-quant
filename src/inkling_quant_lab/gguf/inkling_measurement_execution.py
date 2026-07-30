"""Host-side contracts for matched Inkling measurements on Modal CUDA GPUs.

This module builds pinned llama.cpp commands and validates their machine-readable
results. It does not execute a model, launch Modal, or provide a CPU substitute.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePosixPath
from typing import Final, Literal, TypeAlias, cast

PINNED_LLAMA_CPP_COMMIT: Final = "a015409e6c27b84f60d688823d4c0126a11571fd"
# The pinned build uses ``git rev-parse --short HEAD`` in its depth-one checkout.
PINNED_LLAMA_CPP_BUILD_COMMIT: Final = PINNED_LLAMA_CPP_COMMIT[:7]
PINNED_LLAMA_BENCH_BINARY: Final = "/opt/llama.cpp/build/bin/llama-bench"
PINNED_LLAMA_PERPLEXITY_BINARY: Final = "/opt/llama.cpp/build/bin/llama-perplexity"
PINNED_LLAMA_SERVER_BINARY: Final = "/opt/llama.cpp/build/bin/llama-server"

EXACT_CUDA_DEVICE_NAMES: Final = tuple(f"CUDA{ordinal}" for ordinal in range(8))
EXACT_CUDA_TENSOR_SPLIT: Final = (1,) * len(EXACT_CUDA_DEVICE_NAMES)
PINNED_LLAMA_BENCH_CASES: Final = (
    ("pp512", 512, 0),
    ("pp2048", 2048, 0),
    ("tg128", 0, 128),
)

_MAX_BENCH_OUTPUT_CHARACTERS: Final = 4 * 1024 * 1024
_MAX_PERPLEXITY_OUTPUT_CHARACTERS: Final = 16 * 1024 * 1024
_UINT64_MASK: Final = (1 << 64) - 1
_CHOICES: Final = frozenset({"A", "B", "C", "D"})
_STRICT_INTEGER_RE: Final = re.compile(r"(?:0|-[1-9][0-9]*|[1-9][0-9]*)\Z")
_FINITE_NUMBER_PATTERN: Final = r"(?:[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"
_PERPLEXITY_FINAL_RE: Final = re.compile(
    rf"^Final estimate: PPL = ({_FINITE_NUMBER_PATTERN}) "
    rf"\+/- ({_FINITE_NUMBER_PATTERN})$",
    re.MULTILINE,
)
_PERPLEXITY_FAILURE_RE: Final = re.compile(
    r"(?im)^.*(?:"
    r"\berror:|\bfailed to\b|\bfatal:|\bexception\b|\btraceback\b|"
    r"\bsegmentation fault\b|\bout of memory\b|\bno usable GPU\b|"
    r"\bCUDA error\b"
    r").*$"
)
_BENCH_KEYS: Final = frozenset(
    {
        "build_commit",
        "build_number",
        "cpu_info",
        "gpu_info",
        "backends",
        "model_filename",
        "model_type",
        "model_size",
        "model_n_params",
        "n_batch",
        "n_ubatch",
        "n_threads",
        "cpu_mask",
        "cpu_strict",
        "poll",
        "type_k",
        "type_v",
        "n_gpu_layers",
        "n_cpu_moe",
        "split_mode",
        "main_gpu",
        "no_kv_offload",
        "flash_attn",
        "devices",
        "tensor_split",
        "tensor_buft_overrides",
        "use_mmap",
        "use_direct_io",
        "embeddings",
        "no_op_offload",
        "no_host",
        "fit_target",
        "fit_min_ctx",
        "n_prompt",
        "n_gen",
        "n_depth",
        "test_time",
        "avg_ns",
        "stddev_ns",
        "avg_ts",
        "stddev_ts",
        "samples_ns",
        "samples_ts",
    }
)

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
DiagnosticScorerKind: TypeAlias = Literal[
    "choice",
    "integer",
    "exact_text",
    "json_exact",
]

_DIAGNOSTIC_NORMALIZED_HASH_DOMAIN: Final = b"inkling-quant-lab/diagnostic-normalized-value/v1\0"


class BenchOutputError(ValueError):
    """Raised when llama-bench output violates the pinned result contract."""


class PerplexityOutputError(ValueError):
    """Raised when llama-perplexity output lacks one valid final estimate."""


class StrictJsonError(ValueError):
    """Raised when JSON is ambiguous or outside the JSON data model."""


def _require_exact_int(
    value: object,
    *,
    label: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    integer = value
    if minimum is not None and integer < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    if maximum is not None and integer > maximum:
        raise ValueError(f"{label} must be at most {maximum}")
    return integer


def _require_canonical_absolute_posix_path(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or "\x00" in value
        or "\\" in value
        or value.startswith("//")
    ):
        raise ValueError(f"{label} must be a canonical absolute POSIX path")
    path = PurePosixPath(value)
    if value == "/" or not path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError(f"{label} must be a canonical absolute POSIX path")
    return value


def _require_finite_number(
    value: object,
    *,
    label: str,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be a number")
    number = float(cast("int | float", value))
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    if positive and number <= 0.0:
        raise ValueError(f"{label} must be positive")
    if nonnegative and number < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    return number


def _require_nonempty_string(value: object, *, label: str) -> str:
    if type(value) is not str or not value or "\x00" in value:
        raise ValueError(f"{label} must be non-empty text")
    return value


@dataclass(frozen=True, slots=True)
class ExactCudaTopology:
    """The exact ordered llama.cpp device and split arguments for B300:8."""

    device_names: tuple[str, ...]
    tensor_split: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            type(self.device_names) is not tuple
            or any(type(name) is not str for name in self.device_names)
            or self.device_names != EXACT_CUDA_DEVICE_NAMES
        ):
            raise ValueError("CUDA devices must be exactly CUDA0 through CUDA7 in ordinal order")
        if (
            type(self.tensor_split) is not tuple
            or any(type(part) is not int for part in self.tensor_split)
            or self.tensor_split != EXACT_CUDA_TENSOR_SPLIT
        ):
            raise ValueError("CUDA tensor split must assign one equal part to every exact device")

    @property
    def bench_device_argument(self) -> str:
        """Return llama-bench's slash-delimited device argument."""

        return "/".join(self.device_names)

    @property
    def bench_tensor_split_argument(self) -> str:
        """Return llama-bench's slash-delimited tensor split."""

        return "/".join(str(part) for part in self.tensor_split)

    @property
    def common_device_argument(self) -> str:
        """Return the comma-delimited device argument used by common CLI parsing."""

        return ",".join(self.device_names)

    @property
    def common_tensor_split_argument(self) -> str:
        """Return the comma-delimited tensor split used by common CLI parsing."""

        return ",".join(str(part) for part in self.tensor_split)


def bind_exact_cuda_topology(
    observed_device_names: Sequence[str],
    tensor_split: Sequence[int],
) -> ExactCudaTopology:
    """Bind observed arguments to the exact ordered B300:8 llama.cpp topology."""

    if isinstance(observed_device_names, (str, bytes)):
        raise ValueError("observed CUDA devices must be a sequence of device names")
    if isinstance(tensor_split, (str, bytes)):
        raise ValueError("CUDA tensor split must be a sequence of integers")
    return ExactCudaTopology(tuple(observed_device_names), tuple(tensor_split))


@dataclass(frozen=True, slots=True)
class LlamaBenchCommandSpec:
    """One pinned process that measures all three text-throughput cases."""

    model_path: str
    repetitions: int
    batch_size: int
    ubatch_size: int
    threads: int
    topology: ExactCudaTopology

    def __post_init__(self) -> None:
        _require_canonical_absolute_posix_path(self.model_path, label="model_path")
        _require_exact_int(self.repetitions, label="repetitions", minimum=2)
        batch_size = _require_exact_int(self.batch_size, label="batch_size", minimum=1)
        ubatch_size = _require_exact_int(self.ubatch_size, label="ubatch_size", minimum=1)
        if ubatch_size > batch_size:
            raise ValueError("ubatch_size cannot exceed batch_size")
        _require_exact_int(self.threads, label="threads", minimum=1)
        if not isinstance(self.topology, ExactCudaTopology):
            raise ValueError("topology must be an ExactCudaTopology")


def build_llama_bench_command(spec: LlamaBenchCommandSpec) -> tuple[str, ...]:
    """Build the exact pinned three-case llama-bench command.

    The pinned benchmark parser accepts integer GPU-layer values and slash-delimited
    device lists. ``-p 0 -n 0`` disables its implicit default cases; each repeated
    ``-pg`` then adds one checked case without reloading the model.
    """

    return (
        PINNED_LLAMA_BENCH_BINARY,
        "-m",
        spec.model_path,
        "-p",
        "0",
        "-n",
        "0",
        "-pg",
        "512,0",
        "-pg",
        "2048,0",
        "-pg",
        "0,128",
        "-r",
        str(spec.repetitions),
        "-o",
        "jsonl",
        "-ngl",
        "-2",
        "-ncmoe",
        "0",
        "-sm",
        "layer",
        "-dev",
        spec.topology.bench_device_argument,
        "-ts",
        spec.topology.bench_tensor_split_argument,
        "-fa",
        "on",
        "-nkvo",
        "0",
        "-mmp",
        "1",
        "--no-host",
        "1",
        "-b",
        str(spec.batch_size),
        "-ub",
        str(spec.ubatch_size),
        "-t",
        str(spec.threads),
    )


@dataclass(frozen=True, slots=True)
class LlamaPerplexityCommandSpec:
    """One pinned text-perplexity command."""

    model_path: str
    corpus_path: str
    context_size: int
    batch_size: int
    ubatch_size: int
    chunks: int
    topology: ExactCudaTopology

    def __post_init__(self) -> None:
        _require_canonical_absolute_posix_path(self.model_path, label="model_path")
        _require_canonical_absolute_posix_path(self.corpus_path, label="corpus_path")
        _require_exact_int(self.context_size, label="context_size", minimum=2)
        batch_size = _require_exact_int(self.batch_size, label="batch_size", minimum=1)
        ubatch_size = _require_exact_int(self.ubatch_size, label="ubatch_size", minimum=1)
        if ubatch_size > batch_size:
            raise ValueError("ubatch_size cannot exceed batch_size")
        _require_exact_int(self.chunks, label="chunks", minimum=1)
        if not isinstance(self.topology, ExactCudaTopology):
            raise ValueError("topology must be an ExactCudaTopology")


def build_llama_perplexity_command(
    spec: LlamaPerplexityCommandSpec,
) -> tuple[str, ...]:
    """Build the exact pinned full-offload llama-perplexity command."""

    return (
        PINNED_LLAMA_PERPLEXITY_BINARY,
        "-m",
        spec.model_path,
        "-f",
        spec.corpus_path,
        "-c",
        str(spec.context_size),
        "-b",
        str(spec.batch_size),
        "-ub",
        str(spec.ubatch_size),
        "--chunks",
        str(spec.chunks),
        "--ppl-stride",
        "0",
        "--ppl-output-type",
        "1",
        "-ngl",
        "all",
        "-ncmoe",
        "0",
        "-sm",
        "layer",
        "-dev",
        spec.topology.common_device_argument,
        "-ts",
        spec.topology.common_tensor_split_argument,
        "-fa",
        "on",
        "-kvo",
        "--mmap",
        "-fit",
        "off",
        "--no-host",
    )


@dataclass(frozen=True, slots=True)
class LlamaServerCommandSpec:
    """One pinned multimodal server command for measured requests."""

    model_path: str
    projector_path: str
    context_size: int
    batch_size: int
    ubatch_size: int
    parallel_slots: int
    port: int
    topology: ExactCudaTopology

    def __post_init__(self) -> None:
        _require_canonical_absolute_posix_path(self.model_path, label="model_path")
        _require_canonical_absolute_posix_path(self.projector_path, label="projector_path")
        _require_exact_int(self.context_size, label="context_size", minimum=2)
        batch_size = _require_exact_int(self.batch_size, label="batch_size", minimum=1)
        ubatch_size = _require_exact_int(self.ubatch_size, label="ubatch_size", minimum=1)
        if ubatch_size > batch_size:
            raise ValueError("ubatch_size cannot exceed batch_size")
        _require_exact_int(self.parallel_slots, label="parallel_slots", minimum=1)
        _require_exact_int(self.port, label="port", minimum=1, maximum=65_535)
        if not isinstance(self.topology, ExactCudaTopology):
            raise ValueError("topology must be an ExactCudaTopology")


def build_llama_server_command(spec: LlamaServerCommandSpec) -> tuple[str, ...]:
    """Build the exact pinned full-offload llama-server command."""

    return (
        PINNED_LLAMA_SERVER_BINARY,
        "--log-verbosity",
        "4",
        "-m",
        spec.model_path,
        "-mm",
        spec.projector_path,
        "-c",
        str(spec.context_size),
        "-b",
        str(spec.batch_size),
        "-ub",
        str(spec.ubatch_size),
        "-ngl",
        "all",
        "-ncmoe",
        "0",
        "-sm",
        "layer",
        "-dev",
        spec.topology.common_device_argument,
        "-ts",
        spec.topology.common_tensor_split_argument,
        "-fa",
        "on",
        "-kvo",
        "--mmap",
        "--mmproj-offload",
        "--op-offload",
        "--no-warmup",
        "-fit",
        "off",
        "--no-host",
        "-np",
        str(spec.parallel_slots),
        "-cb",
        "--threads",
        "16",
        "--threads-batch",
        "16",
        "--host",
        "127.0.0.1",
        "--port",
        str(spec.port),
        "--metrics",
        "--slots",
        "--no-ui",
    )


@dataclass(frozen=True, slots=True)
class LlamaBenchResult:
    """Validated measurements from one pinned llama-bench JSONL record."""

    build_commit: str
    test_time_utc: str
    model_path: str
    model_type: str
    model_size_bytes: int
    model_parameter_count: int
    prompt_tokens: int
    generated_tokens: int
    sample_nanoseconds: tuple[int, ...]
    sample_tokens_per_second: tuple[float, ...]
    average_nanoseconds: int
    standard_deviation_nanoseconds: int
    average_tokens_per_second: float
    standard_deviation_tokens_per_second: float
    gpu_info: str
    backends: str


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise StrictJsonError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise StrictJsonError(f"non-finite JSON number: {value}")


def _parse_json_object(line: str, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            line,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except json.JSONDecodeError as error:
        raise BenchOutputError(f"{label} is not valid JSON") from error
    except StrictJsonError as error:
        raise BenchOutputError(f"{label} is not strict JSON: {error}") from error
    if type(value) is not dict:
        raise BenchOutputError(f"{label} must be one JSON object")
    return value


def _record_exact_int(
    record: dict[str, object],
    field: str,
    *,
    minimum: int | None = None,
) -> int:
    try:
        return _require_exact_int(record[field], label=field, minimum=minimum)
    except ValueError as error:
        raise BenchOutputError(str(error)) from error


def _record_number(
    record: dict[str, object],
    field: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    try:
        return _require_finite_number(
            record[field],
            label=field,
            positive=positive,
            nonnegative=nonnegative,
        )
    except ValueError as error:
        raise BenchOutputError(str(error)) from error


def _record_string(record: dict[str, object], field: str) -> str:
    try:
        return _require_nonempty_string(record[field], label=field)
    except ValueError as error:
        raise BenchOutputError(str(error)) from error


def _record_exact_bool(record: dict[str, object], field: str) -> bool:
    value = record[field]
    if type(value) is not bool:
        raise BenchOutputError(f"{field} must be a Boolean")
    return value


def _record_number_tuple(
    record: dict[str, object],
    field: str,
    *,
    integer: bool,
) -> tuple[int, ...] | tuple[float, ...]:
    value = record[field]
    if type(value) is not list:
        raise BenchOutputError(f"{field} must be a JSON array")
    if integer:
        try:
            return tuple(
                _require_exact_int(item, label=f"{field}[{index}]", minimum=1)
                for index, item in enumerate(value)
            )
        except ValueError as error:
            raise BenchOutputError(str(error)) from error
    try:
        return tuple(
            _require_finite_number(item, label=f"{field}[{index}]", positive=True)
            for index, item in enumerate(value)
        )
    except ValueError as error:
        raise BenchOutputError(str(error)) from error


def _cpp_uint64_average(values: tuple[int, ...]) -> int:
    total = 0
    for value in values:
        total = (total + value) & _UINT64_MASK
    return total // len(values)


def _cpp_uint64_sample_standard_deviation(values: tuple[int, ...]) -> int:
    """Reproduce the pinned benchmark's unsigned-integer ``stdev`` template."""

    if len(values) <= 1:
        return 0
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


def _cpp_double_average(values: tuple[float, ...]) -> float:
    total = 0.0
    for value in values:
        total += value
    return total / len(values)


def _cpp_double_sample_standard_deviation(values: tuple[float, ...]) -> float:
    """Reproduce the pinned benchmark's double-precision ``stdev`` template."""

    if len(values) <= 1:
        return 0.0
    mean = _cpp_double_average(values)
    square_sum = 0.0
    for value in values:
        square_sum += value * value
    divisor = len(values) - 1
    variance = square_sum / divisor - mean * mean * len(values) / divisor
    if variance < 0.0:
        raise BenchOutputError("llama-bench sample variance is negative")
    return math.sqrt(variance)


def _serialized_bench_sample(value: float) -> float:
    """Return a number after the JSONL sample's six-significant-digit rounding."""

    return float(format(value, ".6g"))


def _serialized_bench_summary(value: float) -> float:
    """Return a number after ``std::to_string(double)`` six-decimal rounding."""

    return float(format(value, ".6f"))


def _require_bench_record_binding(
    record: dict[str, object],
    *,
    spec: LlamaBenchCommandSpec,
    prompt_tokens: int,
    generated_tokens: int,
) -> None:
    expected_scalar_values: dict[str, object] = {
        "build_commit": PINNED_LLAMA_CPP_BUILD_COMMIT,
        "model_filename": spec.model_path,
        "n_batch": spec.batch_size,
        "n_ubatch": spec.ubatch_size,
        "n_threads": spec.threads,
        "cpu_mask": "0x0",
        "cpu_strict": False,
        "poll": 50,
        "n_gpu_layers": -2,
        "n_cpu_moe": 0,
        "split_mode": "layer",
        "main_gpu": 0,
        "no_kv_offload": False,
        "flash_attn": 1,
        "devices": spec.topology.bench_device_argument,
        "tensor_split": "/".join("1.00" for _ in spec.topology.tensor_split),
        "tensor_buft_overrides": "none",
        "use_mmap": True,
        "use_direct_io": False,
        "embeddings": False,
        "no_op_offload": False,
        "no_host": True,
        "fit_target": 0,
        "fit_min_ctx": 0,
        "n_prompt": prompt_tokens,
        "n_gen": generated_tokens,
        "n_depth": 0,
    }
    for field, expected in expected_scalar_values.items():
        if type(record[field]) is not type(expected) or record[field] != expected:
            raise BenchOutputError(f"llama-bench field {field} differs from its command")

    if _record_string(record, "backends") != "CUDA":
        raise BenchOutputError("llama-bench did not report the pinned CUDA backend")
    for field in ("cpu_info", "gpu_info", "model_type", "type_k", "type_v"):
        _record_string(record, field)
    _record_exact_int(record, "build_number", minimum=0)
    _record_exact_int(record, "model_size", minimum=1)
    _record_exact_int(record, "model_n_params", minimum=1)
    try:
        datetime.strptime(_record_string(record, "test_time"), "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise BenchOutputError("test_time must use llama-bench's UTC timestamp format") from error


def _parse_llama_bench_record(
    line: str,
    *,
    spec: LlamaBenchCommandSpec,
    prompt_tokens: int,
    generated_tokens: int,
) -> LlamaBenchResult:
    record = _parse_json_object(line, label="llama-bench record")
    if set(record) != _BENCH_KEYS:
        missing = sorted(_BENCH_KEYS - set(record))
        extra = sorted(set(record) - _BENCH_KEYS)
        raise BenchOutputError(f"llama-bench JSONL fields differ: missing={missing}, extra={extra}")

    _require_bench_record_binding(
        record,
        spec=spec,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
    )
    sample_ns = _record_number_tuple(record, "samples_ns", integer=True)
    sample_ts = _record_number_tuple(record, "samples_ts", integer=False)
    if len(sample_ns) != spec.repetitions or len(sample_ts) != spec.repetitions:
        raise BenchOutputError("llama-bench sample count differs from repetitions")

    typed_sample_ns = tuple(int(value) for value in sample_ns)
    typed_sample_ts = tuple(float(value) for value in sample_ts)
    token_count = prompt_tokens + generated_tokens
    derived_sample_ts = tuple(
        1_000_000_000.0 * token_count / nanoseconds for nanoseconds in typed_sample_ns
    )
    for index, (tokens_per_second, derived) in enumerate(
        zip(typed_sample_ts, derived_sample_ts, strict=True)
    ):
        if tokens_per_second != _serialized_bench_sample(derived):
            raise BenchOutputError(f"samples_ts[{index}] differs from samples_ns")

    average_ns = _record_exact_int(record, "avg_ns", minimum=1)
    standard_deviation_ns = _record_exact_int(record, "stddev_ns", minimum=0)
    average_ts = _record_number(record, "avg_ts", positive=True)
    standard_deviation_ts = _record_number(record, "stddev_ts", nonnegative=True)
    if average_ns != _cpp_uint64_average(typed_sample_ns):
        raise BenchOutputError("avg_ns differs from the sample mean")
    if standard_deviation_ns != _cpp_uint64_sample_standard_deviation(typed_sample_ns):
        raise BenchOutputError("stddev_ns differs from the samples")
    expected_average_ts = _cpp_double_average(derived_sample_ts)
    if average_ts != _serialized_bench_summary(expected_average_ts):
        raise BenchOutputError("avg_ts differs from the sample mean")
    expected_standard_deviation = _cpp_double_sample_standard_deviation(derived_sample_ts)
    if standard_deviation_ts != _serialized_bench_summary(expected_standard_deviation):
        raise BenchOutputError("stddev_ts differs from the sample standard deviation")

    return LlamaBenchResult(
        build_commit=_record_string(record, "build_commit"),
        test_time_utc=_record_string(record, "test_time"),
        model_path=_record_string(record, "model_filename"),
        model_type=_record_string(record, "model_type"),
        model_size_bytes=_record_exact_int(record, "model_size", minimum=1),
        model_parameter_count=_record_exact_int(record, "model_n_params", minimum=1),
        prompt_tokens=_record_exact_int(record, "n_prompt", minimum=0),
        generated_tokens=_record_exact_int(record, "n_gen", minimum=0),
        sample_nanoseconds=typed_sample_ns,
        sample_tokens_per_second=typed_sample_ts,
        average_nanoseconds=average_ns,
        standard_deviation_nanoseconds=standard_deviation_ns,
        average_tokens_per_second=average_ts,
        standard_deviation_tokens_per_second=standard_deviation_ts,
        gpu_info=_record_string(record, "gpu_info"),
        backends=_record_string(record, "backends"),
    )


def parse_llama_bench_jsonl(
    output: str,
    *,
    spec: LlamaBenchCommandSpec,
) -> tuple[LlamaBenchResult, LlamaBenchResult, LlamaBenchResult]:
    """Parse the exact ordered pp512, pp2048, and tg128 JSONL records."""

    if type(output) is not str or len(output) > _MAX_BENCH_OUTPUT_CHARACTERS:
        raise BenchOutputError("llama-bench output is not bounded text")
    lines = output.splitlines()
    if len(lines) != len(PINNED_LLAMA_BENCH_CASES) or any(not line for line in lines):
        raise BenchOutputError(
            "llama-bench must emit exactly three non-empty ordered JSONL records"
        )
    results = tuple(
        _parse_llama_bench_record(
            line,
            spec=spec,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
        )
        for line, (_, prompt_tokens, generated_tokens) in zip(
            lines,
            PINNED_LLAMA_BENCH_CASES,
            strict=True,
        )
    )
    return cast(
        "tuple[LlamaBenchResult, LlamaBenchResult, LlamaBenchResult]",
        results,
    )


@dataclass(frozen=True, slots=True)
class PerplexityResult:
    """One finite final estimate from llama-perplexity."""

    perplexity: float
    uncertainty: float

    def __post_init__(self) -> None:
        _require_finite_number(self.perplexity, label="perplexity", positive=True)
        _require_finite_number(self.uncertainty, label="uncertainty", nonnegative=True)


def parse_llama_perplexity_final(output: str) -> PerplexityResult:
    """Parse exactly one finite final estimate and reject failure diagnostics."""

    if type(output) is not str or len(output) > _MAX_PERPLEXITY_OUTPUT_CHARACTERS:
        raise PerplexityOutputError("llama-perplexity output is not bounded text")
    failure = _PERPLEXITY_FAILURE_RE.search(output)
    if failure is not None:
        raise PerplexityOutputError(
            f"llama-perplexity reported a failure diagnostic: {failure.group(0).strip()}"
        )
    matches = tuple(_PERPLEXITY_FINAL_RE.finditer(output))
    if len(matches) != 1:
        raise PerplexityOutputError(
            f"llama-perplexity must emit one final estimate; observed {len(matches)}"
        )
    perplexity = float(matches[0].group(1))
    uncertainty = float(matches[0].group(2))
    if not math.isfinite(perplexity) or perplexity <= 0.0:
        raise PerplexityOutputError("llama-perplexity final estimate is not finite and positive")
    if not math.isfinite(uncertainty) or uncertainty < 0.0:
        raise PerplexityOutputError("llama-perplexity uncertainty is not finite and nonnegative")
    return PerplexityResult(perplexity=perplexity, uncertainty=uncertainty)


def score_choice(response: str, expected: str) -> bool:
    """Score exactly one ASCII A-D choice after stripping outer whitespace."""

    if type(response) is not str:
        raise TypeError("response must be text")
    if type(expected) is not str or expected not in _CHOICES:
        raise ValueError("expected choice must be exactly A, B, C, or D")
    normalized = response.strip()
    return normalized in _CHOICES and normalized == expected


def score_strict_decimal_integer(response: str, expected: int) -> bool:
    """Score one stripped canonical integer without plus or leading-zero aliases."""

    if type(response) is not str:
        raise TypeError("response must be text")
    if type(expected) is not int:
        raise ValueError("expected integer must be an integer")
    normalized = response.strip()
    if _STRICT_INTEGER_RE.fullmatch(normalized) is None:
        return False
    return int(normalized, 10) == expected


def normalize_exact_text(value: str) -> str:
    """Strip outer whitespace while preserving case and internal text exactly."""

    if type(value) is not str:
        raise TypeError("exact text value must be text")
    return value.strip()


def score_exact_text(response: str, expected: str) -> bool:
    """Score normalized exact text."""

    return normalize_exact_text(response) == normalize_exact_text(expected)


def _validate_json_value(value: object, *, label: str) -> JsonValue:
    if value is None or type(value) in (bool, int, str):
        return cast("JsonScalar", value)
    if type(value) is float:
        if not math.isfinite(value):
            raise StrictJsonError(f"{label} contains a non-finite number")
        return value
    if type(value) is list:
        return [
            _validate_json_value(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    if type(value) is dict:
        validated: dict[str, JsonValue] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise StrictJsonError(f"{label} contains a non-string object key")
            validated[key] = _validate_json_value(item, label=f"{label}.{key}")
        return validated
    raise StrictJsonError(f"{label} contains a value outside the JSON data model")


def parse_strict_json(value: str) -> JsonValue:
    """Parse JSON while rejecting duplicate object keys and non-finite numbers."""

    if type(value) is not str:
        raise TypeError("JSON response must be text")
    try:
        parsed = json.loads(
            value,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except json.JSONDecodeError as error:
        raise StrictJsonError("response is not valid JSON") from error
    return _validate_json_value(parsed, label="response")


def _json_values_equal(left: JsonValue, right: JsonValue) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is list:
        right_list = cast("list[JsonValue]", right)
        if len(left) != len(right_list):
            return False
        return all(
            _json_values_equal(left_item, right_item)
            for left_item, right_item in zip(left, right_list, strict=True)
        )
    if type(left) is dict:
        right_dict = cast("dict[str, JsonValue]", right)
        if left.keys() != right_dict.keys():
            return False
        return all(_json_values_equal(left[key], right_dict[key]) for key in left)
    return left == right


def score_json_exact(response: str, expected: JsonValue) -> bool:
    """Score a strict JSON response with type-sensitive structural equality."""

    validated_expected = _validate_json_value(expected, label="expected")
    try:
        parsed_response = parse_strict_json(response)
    except StrictJsonError:
        return False
    return _json_values_equal(parsed_response, validated_expected)


def _diagnostic_normalized_sha256(
    scorer_kind: DiagnosticScorerKind,
    normalized_value: JsonValue,
) -> str:
    payload = json.dumps(
        {
            "scorer_kind": scorer_kind,
            "normalized_value": normalized_value,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(_DIAGNOSTIC_NORMALIZED_HASH_DOMAIN + payload).hexdigest()


def diagnostic_expected_normalized_sha256(
    scorer_kind: DiagnosticScorerKind,
    expected: object,
) -> str:
    """Hash one reviewed scorer target without retaining response text."""

    normalized: JsonValue
    if scorer_kind == "choice":
        if type(expected) is not str or expected not in _CHOICES:
            raise ValueError("expected choice must be exactly A, B, C, or D")
        normalized = expected
    elif scorer_kind == "integer":
        if type(expected) is not int:
            raise ValueError("expected integer must be an integer")
        normalized = expected
    elif scorer_kind == "exact_text":
        if type(expected) is not str:
            raise ValueError("expected exact text must be text")
        normalized = normalize_exact_text(expected)
    elif scorer_kind == "json_exact":
        normalized = _validate_json_value(expected, label="expected")
    else:
        raise ValueError(f"unsupported diagnostic scorer kind: {scorer_kind}")
    return _diagnostic_normalized_sha256(scorer_kind, normalized)


@dataclass(frozen=True, slots=True)
class DiagnosticScoreEvidence:
    """Content-free evidence sufficient to independently recompute one score."""

    normalization_succeeded: bool
    normalized_sha256: str | None
    score: bool

    def __post_init__(self) -> None:
        if type(self.normalization_succeeded) is not bool or type(self.score) is not bool:
            raise TypeError("diagnostic score flags must be booleans")
        if self.normalization_succeeded != (self.normalized_sha256 is not None):
            raise ValueError("normalized hash must appear exactly when normalization succeeds")
        if (
            self.normalized_sha256 is not None
            and re.fullmatch(
                r"[0-9a-f]{64}",
                self.normalized_sha256,
            )
            is None
        ):
            raise ValueError("normalized diagnostic hash must be lowercase SHA-256")
        if not self.normalization_succeeded and self.score:
            raise ValueError("a response that failed normalization cannot pass its scorer")


def evaluate_diagnostic_response(
    response: str,
    *,
    scorer_kind: DiagnosticScorerKind,
    expected: object,
) -> DiagnosticScoreEvidence:
    """Normalize and score one response while retaining no response content."""

    if type(response) is not str:
        raise TypeError("response must be text")

    normalized: JsonValue
    if scorer_kind == "choice":
        stripped = response.strip()
        if stripped not in _CHOICES:
            return DiagnosticScoreEvidence(
                normalization_succeeded=False,
                normalized_sha256=None,
                score=False,
            )
        normalized = stripped
    elif scorer_kind == "integer":
        stripped = response.strip()
        if _STRICT_INTEGER_RE.fullmatch(stripped) is None:
            return DiagnosticScoreEvidence(
                normalization_succeeded=False,
                normalized_sha256=None,
                score=False,
            )
        normalized = int(stripped, 10)
    elif scorer_kind == "exact_text":
        normalized = normalize_exact_text(response)
    elif scorer_kind == "json_exact":
        try:
            normalized = parse_strict_json(response)
        except StrictJsonError:
            return DiagnosticScoreEvidence(
                normalization_succeeded=False,
                normalized_sha256=None,
                score=False,
            )
    else:
        raise ValueError(f"unsupported diagnostic scorer kind: {scorer_kind}")

    normalized_sha256 = _diagnostic_normalized_sha256(scorer_kind, normalized)
    expected_sha256 = diagnostic_expected_normalized_sha256(scorer_kind, expected)
    return DiagnosticScoreEvidence(
        normalization_succeeded=True,
        normalized_sha256=normalized_sha256,
        score=normalized_sha256 == expected_sha256,
    )


def _validated_positive_samples(values: Sequence[float], *, label: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be a sequence of numbers")
    try:
        samples = tuple(
            _require_finite_number(value, label=f"{label}[{index}]", positive=True)
            for index, value in enumerate(values)
        )
    except TypeError as error:
        raise ValueError(f"{label} must be a finite sequence of numbers") from error
    if not samples:
        raise ValueError(f"{label} must not be empty")
    return samples


def latency_percentile_ms(values_ms: Sequence[float], percentile: float) -> float:
    """Return a linearly interpolated latency percentile (R-7 method)."""

    samples = tuple(sorted(_validated_positive_samples(values_ms, label="latency_ms")))
    percentile_value = _require_finite_number(percentile, label="percentile")
    if not 0.0 <= percentile_value <= 100.0:
        raise ValueError("percentile must be from 0 through 100")
    position = (len(samples) - 1) * percentile_value / 100.0
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return samples[lower_index]
    fraction = position - lower_index
    return samples[lower_index] + fraction * (samples[upper_index] - samples[lower_index])


@dataclass(frozen=True, slots=True)
class LatencyStatistics:
    """Summary of one non-empty set of positive latency samples in milliseconds."""

    sample_count: int
    minimum_ms: float
    maximum_ms: float
    mean_ms: float
    population_standard_deviation_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float

    def __post_init__(self) -> None:
        _require_exact_int(self.sample_count, label="sample_count", minimum=1)
        for label, value in (
            ("minimum_ms", self.minimum_ms),
            ("maximum_ms", self.maximum_ms),
            ("mean_ms", self.mean_ms),
            ("p50_ms", self.p50_ms),
            ("p95_ms", self.p95_ms),
            ("p99_ms", self.p99_ms),
        ):
            _require_finite_number(value, label=label, positive=True)
        _require_finite_number(
            self.population_standard_deviation_ms,
            label="population_standard_deviation_ms",
            nonnegative=True,
        )
        if not (self.minimum_ms <= self.p50_ms <= self.p95_ms <= self.p99_ms <= self.maximum_ms):
            raise ValueError("latency percentiles are not ordered inside the sample range")


def summarize_latency_ms(values_ms: Sequence[float]) -> LatencyStatistics:
    """Validate and summarize positive latency samples in milliseconds."""

    samples = _validated_positive_samples(values_ms, label="latency_ms")
    return LatencyStatistics(
        sample_count=len(samples),
        minimum_ms=min(samples),
        maximum_ms=max(samples),
        mean_ms=statistics.fmean(samples),
        population_standard_deviation_ms=statistics.pstdev(samples),
        p50_ms=latency_percentile_ms(samples, 50.0),
        p95_ms=latency_percentile_ms(samples, 95.0),
        p99_ms=latency_percentile_ms(samples, 99.0),
    )


__all__ = [
    "EXACT_CUDA_DEVICE_NAMES",
    "EXACT_CUDA_TENSOR_SPLIT",
    "PINNED_LLAMA_BENCH_BINARY",
    "PINNED_LLAMA_BENCH_CASES",
    "PINNED_LLAMA_CPP_BUILD_COMMIT",
    "PINNED_LLAMA_CPP_COMMIT",
    "PINNED_LLAMA_PERPLEXITY_BINARY",
    "PINNED_LLAMA_SERVER_BINARY",
    "BenchOutputError",
    "DiagnosticScoreEvidence",
    "DiagnosticScorerKind",
    "ExactCudaTopology",
    "JsonValue",
    "LatencyStatistics",
    "LlamaBenchCommandSpec",
    "LlamaBenchResult",
    "LlamaPerplexityCommandSpec",
    "LlamaServerCommandSpec",
    "PerplexityOutputError",
    "PerplexityResult",
    "StrictJsonError",
    "bind_exact_cuda_topology",
    "build_llama_bench_command",
    "build_llama_perplexity_command",
    "build_llama_server_command",
    "diagnostic_expected_normalized_sha256",
    "evaluate_diagnostic_response",
    "latency_percentile_ms",
    "normalize_exact_text",
    "parse_llama_bench_jsonl",
    "parse_llama_perplexity_final",
    "parse_strict_json",
    "score_choice",
    "score_exact_text",
    "score_json_exact",
    "score_strict_decimal_integer",
    "summarize_latency_ms",
]
