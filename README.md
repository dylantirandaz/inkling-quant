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
It does not prove quality retention or deployment speed.

The project also records the retained 49-file BF16 control.
The BF16 text files contain 1,894,278,547,552 bytes.
These files are on a mutable Modal volume.
The matched workflow must hash every BF16 and Q3 file again before it uses either subject.

The checked matched-cell file is a planning record.
It cannot start remote work.
It binds the BF16 files, Q3 files, shared projector, tokenizer files, and exact runtime.
It plans to run BF16 first and Q3 second on one eight-B300 allocation.
The local preflight checks the capacity calculation with the minimum values in the config.
It does not probe hardware or run the aggregate capacity screen.
The calculation does not prove per-device placement, runtime workspace fit, or inference.
The current backend audit supports two GPUs only.
A later reviewed change must add an eight-GPU audit and runner.
The plan does not allow quality or speed measurement until both subject smoke tests pass.

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

Inspect the controlled manager commands:

```console
uv run python scripts/manage_inkling_modal.py --help
```

Use the manager for each remote stage.
Do not call the stage module with `modal run`.
The manager checks the stage order, deployment identity, and immutable receipts.

These tracked files define the workflow:

- [Inkling experiment configuration](configs/experiments/inkling_q3_k_m_modal.yaml)
- [Inkling metadata preflight](scripts/preflight_inkling_gguf.py)
- [Inkling local manager](scripts/manage_inkling_modal.py)
- [Inkling remote stages](scripts/quantize_inkling_modal.py)
- [Source adoption record](configs/experiments/inkling_q3_k_m_source_adoption.json)
- [BF16 subject reference](configs/experiments/inkling_bf16_subject_reference.json)
- [Matched-cell configuration](configs/experiments/inkling_q3_k_m_matched_cell_modal.yaml)
- [Matched-cell local preflight](scripts/manage_inkling_matched.py)

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

The public repository contains code, configuration files, tests, and machine-readable evidence.
It does not contain downloaded models, datasets, caches, logs, or run artifacts.

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
The project has not measured final Inkling quality or inference speed.
Most hardware-specific backends support only the tested model and software matrix.
The CPU reference quantizers prove behavior and file contracts, not deployment performance.

Read each machine-readable experiment record before you use its result.
Do not apply a result to a different model, dataset, runtime, or hardware system.

## License

The project uses the Apache License 2.0.
See [LICENSE](LICENSE) for the license terms.
