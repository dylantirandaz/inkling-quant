"""Host-only contracts for matched Inkling measurement commands and parsers."""

from __future__ import annotations

import json
from typing import Any

import pytest

from inkling_quant_lab.gguf.inkling_measurement_execution import (
    EXACT_CUDA_DEVICE_NAMES,
    EXACT_CUDA_TENSOR_SPLIT,
    PINNED_LLAMA_BENCH_BINARY,
    PINNED_LLAMA_CPP_BUILD_COMMIT,
    PINNED_LLAMA_PERPLEXITY_BINARY,
    PINNED_LLAMA_SERVER_BINARY,
    BenchOutputError,
    DiagnosticScoreEvidence,
    ExactCudaTopology,
    LlamaBenchCommandSpec,
    LlamaPerplexityCommandSpec,
    LlamaServerCommandSpec,
    PerplexityOutputError,
    StrictJsonError,
    bind_exact_cuda_topology,
    build_llama_bench_command,
    build_llama_perplexity_command,
    build_llama_server_command,
    diagnostic_expected_normalized_sha256,
    evaluate_diagnostic_response,
    extract_llama_perplexity_machine_failure,
    normalize_exact_text,
    parse_llama_bench_jsonl,
    parse_llama_perplexity_final,
    parse_strict_json,
    score_choice,
    score_exact_text,
    score_json_exact,
    score_strict_decimal_integer,
)

pytestmark = pytest.mark.unit

MODEL_PATH = "/models/inkling/model-00001-of-00049.gguf"
PROJECTOR_PATH = "/models/inkling/mmproj.gguf"
CORPUS_PATH = "/data/wiki.test.raw"


def _topology() -> ExactCudaTopology:
    return bind_exact_cuda_topology(
        EXACT_CUDA_DEVICE_NAMES,
        EXACT_CUDA_TENSOR_SPLIT,
    )


def _bench_spec() -> LlamaBenchCommandSpec:
    return LlamaBenchCommandSpec(
        model_path=MODEL_PATH,
        repetitions=5,
        batch_size=2048,
        ubatch_size=512,
        threads=16,
        topology=_topology(),
    )


def _bench_record(
    prompt_tokens: int,
    generated_tokens: int,
    *,
    mutate: dict[str, object] | None = None,
    remove: str | None = None,
) -> dict[str, object]:
    token_count = prompt_tokens + generated_tokens
    sample_nanoseconds = token_count * 1_000_000
    record: dict[str, object] = {
        "build_commit": PINNED_LLAMA_CPP_BUILD_COMMIT,
        "build_number": 0,
        "cpu_info": "host orchestration",
        "gpu_info": "NVIDIA B300 x8",
        "backends": "CUDA",
        "model_filename": MODEL_PATH,
        "model_type": "Inkling",
        "model_size": 1,
        "model_n_params": 1,
        "n_batch": 2048,
        "n_ubatch": 512,
        "n_threads": 16,
        "cpu_mask": "0x0",
        "cpu_strict": False,
        "poll": 50,
        "type_k": "f16",
        "type_v": "f16",
        "n_gpu_layers": -2,
        "n_cpu_moe": 0,
        "split_mode": "layer",
        "main_gpu": 0,
        "no_kv_offload": False,
        "flash_attn": 1,
        "devices": "/".join(EXACT_CUDA_DEVICE_NAMES),
        "tensor_split": "/".join("1.00" for _ in EXACT_CUDA_TENSOR_SPLIT),
        "tensor_buft_overrides": "none",
        "use_mmap": True,
        "use_direct_io": False,
        "embeddings": False,
        "no_op_offload": 0,
        "no_host": True,
        "fit_target": 0,
        "fit_min_ctx": 0,
        "n_prompt": prompt_tokens,
        "n_gen": generated_tokens,
        "n_depth": 0,
        "test_time": "2026-07-30T00:00:00Z",
        "avg_ns": sample_nanoseconds,
        "stddev_ns": 0,
        "avg_ts": 1000.0,
        "stddev_ts": 0.0,
        "samples_ns": [sample_nanoseconds] * 5,
        "samples_ts": [1000.0] * 5,
    }
    if mutate is not None:
        record.update(mutate)
    if remove is not None:
        del record[remove]
    return record


def _bench_output(
    *,
    mutation_index: int | None = None,
    mutate: dict[str, object] | None = None,
    remove: str | None = None,
) -> str:
    cases = ((512, 0), (2048, 0), (0, 128))
    records = [
        _bench_record(
            prompt_tokens,
            generated_tokens,
            mutate=mutate if index == mutation_index else None,
            remove=remove if index == mutation_index else None,
        )
        for index, (prompt_tokens, generated_tokens) in enumerate(cases)
    ]
    return "\n".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) for record in records
    )


def test_exact_command_builders_pin_all_gpu_execution_arguments() -> None:
    topology = _topology()
    bench = build_llama_bench_command(_bench_spec())
    perplexity = build_llama_perplexity_command(
        LlamaPerplexityCommandSpec(
            model_path=MODEL_PATH,
            corpus_path=CORPUS_PATH,
            context_size=512,
            batch_size=2048,
            ubatch_size=512,
            chunks=64,
            topology=topology,
        )
    )
    server = build_llama_server_command(
        LlamaServerCommandSpec(
            model_path=MODEL_PATH,
            projector_path=PROJECTOR_PATH,
            context_size=8192,
            batch_size=2048,
            ubatch_size=512,
            parallel_slots=4,
            port=8080,
            topology=topology,
        )
    )

    assert bench[:2] == (PINNED_LLAMA_BENCH_BINARY, "-v")
    assert bench.count("-v") == 1
    assert bench[bench.index("-ngl") : bench.index("-ngl") + 2] == ("-ngl", "-2")
    assert bench[bench.index("-dev") : bench.index("-dev") + 2] == (
        "-dev",
        topology.bench_device_argument,
    )
    assert bench[bench.index("-ts") : bench.index("-ts") + 2] == (
        "-ts",
        topology.bench_tensor_split_argument,
    )
    assert "--no-host" in bench

    assert perplexity[:3] == (
        PINNED_LLAMA_PERPLEXITY_BINARY,
        "--log-verbosity",
        "4",
    )
    assert perplexity.count("--log-verbosity") == 1
    assert perplexity[perplexity.index("-ngl") : perplexity.index("-ngl") + 2] == ("-ngl", "all")
    assert perplexity[perplexity.index("-dev") : perplexity.index("-dev") + 2] == (
        "-dev",
        topology.common_device_argument,
    )
    assert "--no-host" in perplexity

    assert server[0] == PINNED_LLAMA_SERVER_BINARY
    assert server[server.index("-np") : server.index("-np") + 2] == ("-np", "4")
    assert server[server.index("-dev") : server.index("-dev") + 2] == (
        "-dev",
        topology.common_device_argument,
    )
    assert "--no-host" in server


@pytest.mark.parametrize(
    ("factory", "match"),
    (
        (
            lambda: bind_exact_cuda_topology(EXACT_CUDA_DEVICE_NAMES[:-1], EXACT_CUDA_TENSOR_SPLIT),
            "CUDA devices",
        ),
        (
            lambda: bind_exact_cuda_topology(EXACT_CUDA_DEVICE_NAMES, EXACT_CUDA_TENSOR_SPLIT[:-1]),
            "tensor split",
        ),
        (
            lambda: LlamaBenchCommandSpec(
                model_path="relative.gguf",
                repetitions=5,
                batch_size=2048,
                ubatch_size=512,
                threads=16,
                topology=_topology(),
            ),
            "absolute POSIX",
        ),
        (
            lambda: LlamaPerplexityCommandSpec(
                model_path=MODEL_PATH,
                corpus_path=CORPUS_PATH,
                context_size=512,
                batch_size=512,
                ubatch_size=1024,
                chunks=64,
                topology=_topology(),
            ),
            "cannot exceed",
        ),
        (
            lambda: LlamaServerCommandSpec(
                model_path=MODEL_PATH,
                projector_path=PROJECTOR_PATH,
                context_size=8192,
                batch_size=2048,
                ubatch_size=512,
                parallel_slots=4,
                port=65_536,
                topology=_topology(),
            ),
            "at most",
        ),
    ),
)
def test_command_specs_reject_scope_drift(
    factory: Any,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        factory()


def test_llama_bench_parser_accepts_exact_three_record_output() -> None:
    results = parse_llama_bench_jsonl(_bench_output(), spec=_bench_spec())

    assert [(result.prompt_tokens, result.generated_tokens) for result in results] == [
        (512, 0),
        (2048, 0),
        (0, 128),
    ]
    assert all(result.backends == "CUDA" for result in results)
    assert all(result.average_tokens_per_second == 1000.0 for result in results)
    assert all(result.sample_nanoseconds == (result.average_nanoseconds,) * 5 for result in results)


@pytest.mark.parametrize(
    ("output", "match"),
    (
        (_bench_output() + "\n{}", "exactly three"),
        (_bench_output(mutation_index=0, remove="gpu_info"), "missing="),
        (_bench_output(mutation_index=0, mutate={"unexpected": 1}), "extra="),
        (
            _bench_output(mutation_index=0, mutate={"test_time": "2026-07-30"}),
            "timestamp",
        ),
        (
            _bench_output(mutation_index=0, mutate={"backends": "CPU"}),
            "CUDA backend",
        ),
        (
            _bench_output(mutation_index=0, mutate={"no_op_offload": False}),
            "field no_op_offload",
        ),
        (
            _bench_output(mutation_index=0, mutate={"no_op_offload": 1}),
            "field no_op_offload",
        ),
        (
            _bench_output(mutation_index=0, mutate={"samples_ts": [999.0] * 5}),
            r"samples_ts\[0\]",
        ),
        (
            _bench_output(mutation_index=0, mutate={"samples_ns": [512_000_000] * 4}),
            "sample count",
        ),
        ("{}\n{}\n{}", "fields differ"),
    ),
)
def test_llama_bench_parser_rejects_contract_drift(output: str, match: str) -> None:
    with pytest.raises(BenchOutputError, match=match):
        parse_llama_bench_jsonl(output, spec=_bench_spec())


def test_llama_bench_parser_rejects_duplicate_and_nonfinite_json() -> None:
    first, second, third = _bench_output().splitlines()
    duplicate = first[:-1] + ',"build_commit":"other"}'
    nonfinite = first.replace('"avg_ts":1000.0', '"avg_ts":NaN')

    with pytest.raises(BenchOutputError, match="duplicate"):
        parse_llama_bench_jsonl("\n".join((duplicate, second, third)), spec=_bench_spec())
    with pytest.raises(BenchOutputError, match="non-finite"):
        parse_llama_bench_jsonl("\n".join((nonfinite, second, third)), spec=_bench_spec())


def test_perplexity_parser_accepts_one_finite_final_estimate() -> None:
    result = parse_llama_perplexity_final(
        "loading model\nFinal estimate: PPL = 5.125 +/- 0.03125\n"
    )

    assert result.perplexity == 5.125
    assert result.uncertainty == 0.03125


@pytest.mark.parametrize(
    ("stdout", "stderr", "expected"),
    (
        (
            "",
            "IQL_MEASUREMENT_PERPLEXITY_ERROR_V1 code=invalid_statistics\n",
            "IQL_MEASUREMENT_PERPLEXITY_ERROR_V1 code=invalid_statistics",
        ),
        (
            "IQL_MEASUREMENT_PERPLEXITY_ERROR_V1 code=invalid_statistics\n",
            "IQL_MEASUREMENT_PERPLEXITY_ERROR_V1 code=measurement_failed\n",
            "IQL_MEASUREMENT_PERPLEXITY_ERROR_V1 code=invalid_statistics",
        ),
        (
            "IQL_MEASUREMENT_PERPLEXITY_ERROR_V1 code=measurement_failed\n",
            "",
            "IQL_MEASUREMENT_PERPLEXITY_ERROR_V1 code=measurement_failed",
        ),
        ("ordinary failure output", "", None),
        ("IQL_MEASUREMENT_PERPLEXITY_ERROR_V1 code=unknown\n", "", None),
        (
            "IQL_MEASUREMENT_PERPLEXITY_ERROR_V1 code=open_output\n",
            "IQL_MEASUREMENT_PERPLEXITY_ERROR_V1 code=write_output\n",
            None,
        ),
        ("prefix IQL_MEASUREMENT_PERPLEXITY_ERROR_V1 code=open_output", "", None),
    ),
)
def test_perplexity_machine_failure_extractor_is_bounded_to_safe_codes(
    stdout: str,
    stderr: str,
    expected: str | None,
) -> None:
    assert extract_llama_perplexity_machine_failure(stdout, stderr) == expected


@pytest.mark.parametrize(
    ("output", "match"),
    (
        ("loading model", "observed 0"),
        (
            "Final estimate: PPL = 5 +/- 0.1\nFinal estimate: PPL = 6 +/- 0.2",
            "observed 2",
        ),
        ("fatal: CUDA initialization failed\nFinal estimate: PPL = 5 +/- 0.1", "failure"),
        ("Unexpected negative standard deviation of log(prob)", "failure"),
        ("IQL_MEASUREMENT_PERPLEXITY_ERROR_V1 code=invalid_statistics", "failure"),
        ("Final estimate: PPL = inf +/- inf", "malformed"),
        (
            "Final estimate: PPL = 5 +/- 0.1\nFinal estimate: PPL = inf +/- inf",
            "malformed",
        ),
        ("Final estimate: PPL = 0 +/- 0.1", "positive"),
    ),
)
def test_perplexity_parser_rejects_missing_ambiguous_or_failed_output(
    output: str,
    match: str,
) -> None:
    with pytest.raises(PerplexityOutputError, match=match):
        parse_llama_perplexity_final(output)


def test_diagnostic_scorers_are_strict_and_content_free() -> None:
    assert score_choice(" A\n", "A")
    assert not score_choice("A.", "A")
    assert score_strict_decimal_integer("-12", -12)
    assert not score_strict_decimal_integer("012", 12)
    assert normalize_exact_text("  Keep Case  ") == "Keep Case"
    assert score_exact_text(" answer\n", "answer")
    assert score_json_exact('{"a":[1,true,null]}', {"a": [1, True, None]})
    assert not score_json_exact('{"a":1,"a":1}', {"a": 1})

    passing = evaluate_diagnostic_response(
        '{"answer":42}',
        scorer_kind="json_exact",
        expected={"answer": 42},
    )
    malformed = evaluate_diagnostic_response(
        '{"answer":NaN}',
        scorer_kind="json_exact",
        expected={"answer": 42},
    )

    assert passing.normalization_succeeded
    assert passing.score
    assert passing.normalized_sha256 == diagnostic_expected_normalized_sha256(
        "json_exact",
        {"answer": 42},
    )
    assert malformed == DiagnosticScoreEvidence(
        normalization_succeeded=False,
        normalized_sha256=None,
        score=False,
    )


def test_strict_json_rejects_ambiguous_or_non_json_values() -> None:
    assert parse_strict_json('{"a":[1,2.5]}') == {"a": [1, 2.5]}

    with pytest.raises(StrictJsonError, match="duplicate"):
        parse_strict_json('{"a":1,"a":2}')
    with pytest.raises(StrictJsonError, match="non-finite"):
        parse_strict_json('{"a":NaN}')
    with pytest.raises(TypeError, match="text"):
        parse_strict_json(1)  # type: ignore[arg-type]
