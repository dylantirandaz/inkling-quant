# Inkling Quant Lab

Inkling Quant Lab quantizes and tests large mixture-of-experts models.

A mixture-of-experts (MoE) model selects a small group of expert networks for each token.
This toolkit measures the effect of quantization on the model and its expert routing.

The toolkit records each experiment input and result.
It also records the model revision, software versions, hardware, random seed, and file checksums.
These records help another user repeat and inspect an experiment.

## Verified Inkling result

The exact Inkling export is complete.
The workflow used the official model and the pinned software revisions below.

| Item | Verified value |
|---|---|
| Source model | `thinkingmachines/Inkling` |
| Source revision | `86b4d430ab871652a707666b89203a866888c5e5` |
| Converter source | `danielhanchen/llama.cpp` |
| Converter revision | `a015409e6c27b84f60d688823d4c0126a11571fd` |
| Text quantization | Stock `Q3_K_M` |
| Text output | 49 GGUF split files |
| Text output size | 451,035,400,288 bytes |
| Projector output | BF16 GGUF file |
| Projector size | 183,264,288 bytes |
| Final manifest SHA-256 | `23db1314d521210bab5d53df20ed432f784774c59d98e8db3de9004702e1ac7a` |
| Final verification receipt SHA-256 | `08b4928333720962e1192ef0af12672c8155c70ddc03813376cbd431c2409291` |

GGUF is a file format for models that use the `llama.cpp` runtime.
BF16 is the 16-bit brain floating-point data type.

The workflow omitted the multi-token prediction (MTP) tensors.
The pinned `llama.cpp` converter does not support these Inkling tensors.
The workflow records this omission and does not hide it.

The final verification checked the export structure, file set, sizes, and checksums.
It did not measure the quality of the quantized Inkling model.
The matched measurement below did this work later.

An external immutable receipt records a controlled smoke-test pass for the final Q3 export.
The test loaded all 49 text files and the BF16 projector.
It ran the text, image, and audio probes two times.
The token output was repeatable, and all checked unpadded logits were finite.
The backend audit found no CPU model graph work.

The receipt has file SHA-256
`9b951a46cde9fbd30a825e606ce7f84c8c2a4ec9fb5e61971d8bb8b386279871`
and internal canonical seal
`2eccb51e6d4519a47930ca3a89cebe22f09f883b82868343b16eefc5a9f8dad7`.
The receipt did not capture the host CPU model.
The repository therefore does not publish this observation as a supported compatibility cell.
The observed pass applies only to the recorded model, files, runtime, software, and two B300 GPUs.
That smoke pass does not prove quality retention or deployment speed.

The project also records the retained 49-file BF16 control.
The BF16 text files contain 1,894,278,547,552 bytes.
These files are on a mutable Modal volume.
The matched workflow must hash every BF16 and Q3 file again before it uses either subject.

The [checked matched-cell file](configs/experiments/inkling_q3_k_m_matched_cell_modal.yaml)
is the execution record for one matched smoke-test cell.
It binds the BF16 files, Q3 files, shared projector, tokenizer files, and exact runtime.
The matched Modal runner runs BF16 first and Q3 second on one eight-B300 allocation.
It hashes every input again, checks the exact CUDA device and peer topology, and runs
the same text, image, and audio probes for both subjects.
The manager keeps local review, deployment, and launch as separate steps.
Deployment does not start compute.
Launch can start one call only after an exact confirmation.

The accepted Q3 smoke receipt keeps its exact two-GPU audit.
The repository now has a
[separate validator for backend marker logs from the matched eight-GPU cell](src/inkling_quant_lab/gguf/inkling_matched_execution.py).
The validator requires one text graph to use CUDA0 through CUDA7.
It requires each vision and audio graph to use CUDA0 only.
It rejects CPU model graph work and unexpected CUDA identities.
CPU-only unit tests cover the validator.
The repository has not run this separate matched smoke-test cell.
The runner and its unit tests do not prove that Inkling can run on eight GPUs.
A valid immutable rollup receipt from the confirmed run is the required evidence.

The repository now includes a
[separate matched measurement plan](configs/experiments/inkling_q3_k_m_measurement_modal.yaml).
It compares the exact BF16 and Q3 subjects with the same runtime, hardware, data,
prompts, tokenization, and decoding settings.
It measures token negative log-likelihood, perplexity, diagnostic accuracy, load time, latency,
throughput, memory, GPU use, and file size.

The confirmed measurement completed on eight NVIDIA B300 GPUs.
The [immutable terminal receipt](docs/experiments/inkling-q3-k-m-measurement-8xb300.json)
contains the full recorded result and the exact experiment scope.
The Git file is byte-for-byte equal to the downloaded receipt.
Its file SHA-256 is
`c2350ba71c5211e404d4b3701118496f1321ce8ab7e9af37d00a91048190dc6d`.
Its domain-separated content SHA-256 is
`ea11084d2cb80b5d97cc9c4412dfbf73be361ac5d9824439cbf39d63cde7f4b3`.

| Measure | BF16 | Q3 | Result |
|---|---:|---:|---|
| Mean token NLL | 1.225250 | 1.324502 | Q3 delta was 0.099252. The limit was less than 0.1. |
| Perplexity | 3.4050 | 3.7603 | Recorded for the same 16,320 paired token positions. |
| Diagnostic accuracy | 5 of 64 | 7 of 64 | The Q3 comparison gate passed. The BF16 absolute floor failed. |
| Text file size | 1,894,278,547,552 bytes | 451,035,400,288 bytes | Q3 used 23.81% of the BF16 bytes. |
| Cold readiness median | 1,309.68 seconds | 127.34 seconds | Three trials per subject. This used file-level cache advice, not a global cache flush. |
| `llama-bench` generation | 21.083 tokens/s | 24.845 tokens/s | Q3 was 1.178 times faster for the `tg128` case. |
| `llama-bench` prompt processing | 697.788 tokens/s | 199.327 tokens/s | Q3 was slower for the `pp512` case. |

The Q3-versus-BF16 NLL and accuracy comparison gates passed.
The final quality-retention result is false because the BF16 control failed the required
absolute accuracy floors.
This diagnostic set therefore cannot support a quality-retention claim.
Q3 decode speed was higher, but prompt processing and time to first token were slower.
The receipt supports only the recorded metric-specific comparisons.
It does not support a general speedup or a single-run causation claim.

The receipt binds three lower-level subject and comparison records by content hash and size.
Those records remain in Modal storage and are not yet copied into this repository.
The receipt records no prompt text or model output text.

The Git repository does not contain the model files.
The files are too large for Git, and the project does not upload model weights by default.

## Main functions

The toolkit provides these functions:

- It validates experiment configuration files.
- It creates deterministic run identifiers.
- It records immutable run artifacts.
- It supports CPU-only development tests.
- It provides reference INT8 and INT4 quantizers.
- It provides optional native and hardware-specific quantizers.
- It records MoE routes and routing drift.
- It measures quality, latency, throughput, memory, and serialized size.
- It compares a baseline model with one or more quantized models.
- It creates JSON, CSV, Markdown, and SVG reports.
- It resumes an incomplete run without rewriting successful stages.

Reference quantizers test the experiment contracts.
They are not optimized inference kernels.
Each optional backend must pass its capability checks before the toolkit uses it.

## Requirements

Use Python 3.11 or a later version.
Use `uv` to create the environment and install dependencies.

Install the base environment:

```console
uv sync
```

Install the development tools:

```console
uv sync --extra dev
```

Optional dependencies are separate.
Install only the dependencies that your experiment needs.

```console
uv sync --extra hf
uv sync --extra mlx
uv sync --extra gptq
uv sync --extra awq
uv sync --extra fp8
uv sync --extra modal
```

The `modal` extra conflicts with the `gptq` and `awq` extras.
Use separate environments for these workflows.

## Local quick start

Inspect the local environment:

```console
uv run iql doctor
```

Validate a small CPU experiment:

```console
uv run iql validate configs/experiments/tiny_moe_int8.yaml
```

Run the experiment:

```console
uv run iql run configs/experiments/tiny_moe_int8.yaml
```

Run the baseline and the INT4 candidate:

```console
uv run iql run configs/experiments/tiny_moe_baseline.yaml
uv run iql run configs/experiments/tiny_moe_int4.yaml
```

The command prints the run directory.
Use that directory in the next commands.

```console
uv run iql resume artifacts/<run-id>
uv run iql compare artifacts/<baseline-run-id> artifacts/<candidate-run-id>
uv run iql report artifacts/<run-id-or-comparison>
```

Use `--json` when a command must return one JSON document.

## Inkling workflow

The Inkling workflow uses Modal for remote storage and compute.
The workflow accepts only the pinned Inkling model and stock `Q3_K_M` quantization.
It rejects a different model, model revision, converter revision, or quantization type.

Install the Modal environment:

```console
uv sync --extra modal
```

Run the metadata preflight first:

```console
uv run python scripts/preflight_inkling_gguf.py \
  --config configs/experiments/inkling_q3_k_m_modal.yaml
```

The preflight does not download model weights.
It does not start a remote Modal function.

Validate the checked BF16/Q3 matched plan:

```console
uv run python scripts/manage_inkling_matched.py preflight --json
```

This command reads local control files only.
It does not read the model files, inspect a remote volume, probe hardware, write a receipt, or start compute.
Only the `verify_subject_references` stage can pass.
Every remote stage remains `not_executed`.

Inspect the execution-ready matched controls:

```console
uv run python scripts/manage_inkling_matched_modal.py inspect --json
```

This command is local-only.
It checks the files and prints the exact run identity.

Inspect the matched quality and performance controls:

```console
uv run python scripts/manage_inkling_measurement_modal.py inspect --json
```

This command is also local-only.
It does not deploy a Modal function or start GPU compute.
The measurement manager keeps deployment and launch as separate confirmed actions.

Inspect the controlled manager commands:

```console
uv run python scripts/manage_inkling_modal.py --help
uv run python scripts/manage_inkling_matched_modal.py --help
uv run python scripts/manage_inkling_measurement_modal.py --help
```

Use the manager for each remote stage.
Do not call the stage module with `modal run`.
The manager checks the reviewed Git state, deployment identity, launch confirmation,
and immutable control records.

These tracked files define the workflow:

- [Inkling experiment configuration](configs/experiments/inkling_q3_k_m_modal.yaml)
- [Inkling metadata preflight](scripts/preflight_inkling_gguf.py)
- [Inkling local manager](scripts/manage_inkling_modal.py)
- [Inkling remote stages](scripts/quantize_inkling_modal.py)
- [Source adoption record](configs/experiments/inkling_q3_k_m_source_adoption.json)
- [BF16 subject reference](configs/experiments/inkling_bf16_subject_reference.json)
- [Matched-cell configuration](configs/experiments/inkling_q3_k_m_matched_cell_modal.yaml)
- [Matched-cell local preflight](scripts/manage_inkling_matched.py)
- [Matched-cell Modal manager](scripts/manage_inkling_matched_modal.py)
- [Matched-cell Modal runner](scripts/run_inkling_matched_modal.py)
- [Matched measurement configuration](configs/experiments/inkling_q3_k_m_measurement_modal.yaml)
- [Matched measurement Modal manager](scripts/manage_inkling_measurement_modal.py)
- [Matched measurement Modal runner](scripts/run_inkling_measurement_modal.py)
- [Matched measurement result](docs/experiments/inkling-q3-k-m-measurement-8xb300.json)

## Configuration

Each experiment starts from a checked-in YAML file.
An experiment can extend model, quantization, and evaluation fragments.

```yaml
extends:
  - ../models/tiny_moe.yaml
  - ../quantization/dynamic_int8.yaml
  - ../evaluations/tiny_cpu.yaml
schema_version: "1.0"
name: tiny-moe-int8
seed: 17
```

The command line can override a value with `--set KEY=VALUE`.
The toolkit saves the resolved configuration in the run directory.
The toolkit also calculates a hash for that configuration.

The configuration does not select a hidden precision or device.
It does not enable remote model code by default.

## Run artifacts

Each run uses this directory structure:

```text
artifacts/<run-id>/
├── manifest.json
├── resolved_config.yaml
├── environment.json
├── events.jsonl
├── status.json
├── metrics/
├── routing/
├── checkpoints/
├── reports/
└── completion.json
```

The manifest is the main run record.
The event log uses JSON Lines format.
Each line is one structured event.

The toolkit treats successful stage outputs as append-only data.
A resume operation checks the existing output before it skips a stage.
Use `--force-stage` only when you must replace one successful stage and its dependent stages.
The toolkit archives the old outputs before it runs the forced stage.

## Comparison rules

The toolkit compares runs only when their important inputs match.
These inputs include the model revision, dataset, seed, prompt, and decoding settings.
They also include the runtime, hardware, and benchmark method.

The report keeps failed and unavailable measurements.
It does not change an unavailable value to zero.
An unsafe override stays visible in the comparison record.

## Security rules

The toolkit uses these default rules:

- It disables remote model code.
- It rejects pickle-based model files.
- It prefers `safetensors` model files.
- It reads secrets from environment variables.
- It removes secret values from logs.
- It keeps artifact paths under the configured artifact root.
- It disables external uploads.
- It disables prompt and model-output logs.
- It passes subprocess arguments without shell interpolation.

The `run`, `resume`, and `inspect-model` commands need `--allow-remote-code` for remote model code.
Use that option only after you inspect and pin the remote repository.

## Development checks

Run these checks before you push a change:

```console
uv sync --extra dev --extra hf --extra modal --frozen
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest -m "not network and not gpu and not slow and not large_model"
```

Network, GPU, large-model, and optional-backend tests need explicit selection.
These tests must use fixed external revisions.

## Public repository contents

The public repository contains code, configuration files, tests, small checked evaluation inputs,
and machine-readable evidence.
It does not contain downloaded models, full external datasets, caches, logs, or run artifacts.

The public repository publishes only this Markdown file.
The `.gitignore` file excludes `SPEC.md`, `SDD.md`, `TDD.md`, and agent instruction files.

The main paths are:

- [Python package](src/inkling_quant_lab/)
- [Configuration files](configs/)
- [Operational scripts](scripts/)
- [Tests](tests/)
- [Machine-readable experiment records](docs/experiments/)
- [Package metadata](pyproject.toml)

## Limits

The final Inkling export does not include MTP tensors.
The completed measurement applies only to the exact recorded model, runtime, software,
hardware, data, and protocol.
It did not pass the final quality-retention gate because the BF16 control failed the
absolute accuracy floors.
Most hardware-specific backends support only the tested model and software matrix.
The CPU reference quantizers prove behavior and file contracts, not deployment performance.

Read the machine-readable record before use.
Do not apply a result to a different model, dataset, runtime, software, hardware, or protocol.

## License

The project uses the Apache License 2.0.
See [LICENSE](LICENSE) for the license terms.
