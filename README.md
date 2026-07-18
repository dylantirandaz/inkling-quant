# Inkling Quant Lab

Inkling Quant Lab is a CPU-first research toolkit for measuring how uniform and expert-aware
quantization affects model quality, MoE routing, latency, throughput, memory, serialized size, and
behavior learned during post-training.

The primary deliverable is a reproducible experiment system rather than only a quantized checkpoint.
Runs preserve resolved configuration, environment, logs, stage state, metrics, routing evidence,
candidate metadata, checksums, and reports so measured facts can be audited separately from
interpretation.

## What is available

The default path uses deterministic local dense, top-2 MoE, and multimodal contract fixtures. It
does not need a network connection, download model weights, execute remote model code, require a
GPU, or upload artifacts.

| Area | Current status |
|---|---|
| Local CPU model fixtures | Implemented |
| Pinned public native Mixtral adapter | Implemented; opt-in CPU/network smoke |
| No-op quantizer | Implemented |
| Dynamic INT8 CPU reference | Implemented; dequantizes before matmul |
| Packed weight-only INT4 CPU reference | Implemented; unpacks on forward |
| Mixed INT8/INT4 expert-aware CPU reference | Implemented |
| Native dynamic INT8 CPU | Implemented; capability-probed, prepacked native operator |
| Native KleidiAI INT4 CPU | Implemented; capability-probed opaque W4A8 operator |
| Public-model native linear quality | Executed on 32 TinyStories samples; attention linears only, fused experts FP32 |
| Public-model full MLX q4/q8 quality | Executed frozen direct-MLX 32-story slice on Apple M3; all eligible leaves including fused experts |
| Registered MLX public-MoE pipeline | Exact Stories15M q4/q8 path executed on Apple Metal with safe export/reload, loss, routing, and reports; pipeline benchmark disabled |
| Learned-router post-training and q4/q8 overlay | Ten-step Metal training executed but failed two held-out accuracy gates; failure sealed, no overlay/q4/q8 retention claim |
| Routing events, aggregate/sample/full modes, and drift metrics | Implemented for validated adapters |
| Deterministic evaluation, benchmark records, comparison, Pareto, Markdown/CSV/SVG | Implemented |
| Isolated benchmark-stage peak host RSS | Executed on tiny MoE and pinned Stories15M/native INT8 with distinct spawned workers; includes stage construction, not steady-state-only memory |
| Source-weight-free candidate subject-artifact peak RSS | Executed pinned Stories15M governed export in a fresh worker with 81 strict-assigned tensors, 24 packed native-INT8 wrappers, and no float source weights; cached config/tokenizer still required |
| Local two-process gloo collective smoke | Opt-in; collectives only, not model sharding |
| Local two-process CPU DTensor tensor parallel | Executed tiny-MLP parameter-sharded forward |
| Public-MoE expert tensor parallel | Executed all six Stories15M expert blocks with complementary rank-local slices totaling half of expert state per rank; not an end-to-end transformer run or peak-residency result |
| GPTQModel GPTQ | Exact pinned Stories15M INT4-attention CPU path executed with strict project-owned export reload; other matrices pending |
| Exact Inkling GGUF Q3_K_M | Official revision and 1.90-TB index preflight verified; pinned converter and guarded Modal workflow implemented |
| GPTQModel AWQ | Adapter/API boundary implemented; no governed promoted CPU or CUDA experiment |
| Fine-grained FP8 | Transformers adapter/API boundary and CUDA CC>=9.0 probe implemented; no composed experiment yet |
| Hugging Face/Accelerate large-model path | Native single-CPU Mixtral validated; sharding unverified |
| Apple MPS eager runtime | Validated for single-device local MoE execution/routing on Apple M3 |
| CUDA eager runtime | Implemented fail-closed boundary; compatible-hardware run pending |
| SGLang MLX serving | One real Apple Metal MoE request; external smoke only, no pipeline adapter |
| vLLM Apple CPU | Patched direct public-MoE inference executed with exact Transformers token match; external only |
| Sharding and Inkling-compatible checkpoints | Public Stories15M expert blocks validated on two CPU ranks; full-transformer and Inkling checkpoint sharding unverified |
| HTML report | Optional escaped-Markdown view for configured run reports; disabled by default |

Reference quantizers establish correctness, metadata, storage, policy, and comparison contracts;
they are not optimized kernels. The separately named native CPU backends avoid per-forward weight
reconstruction, but support and performance are established only by an execution probe and a
benchmark on the declared PyTorch build, CPU, shape, and thread count.

## Install

Python 3.11 or newer and [`uv`](https://docs.astral.sh/uv/) are required.

```console
uv sync
```

For the test/lint/type-check toolchain:

```console
uv sync --extra dev
```

The `hf` extra installs Transformers/Accelerate for the native Mixtral adapter. The platform-marked
`mlx` extra pins the exact Apple-Silicon matrix used by the registered Stories15M path. The `awq`
and `gptq` extras install the exact GPTQModel 5.8.0 integration matrix; `fp8` installs the bounded
Transformers fine-grained-FP8 matrix. Installing an extra makes the adapter importable, but support
still fails closed unless the immutable model, runtime, explicit device, policy, calibration, and
hardware/software gates pass. One exact GPTQ/CPU matrix has executed; AWQ, general GPTQ/CUDA, and
FP8 conversion evidence remain open. The `cuda`, `vllm`, `sglang`, and `multimodal` groups remain
separate integration boundaries.

```console
uv sync --extra awq   # GPTQModel AWQ conversion dependencies
uv sync --extra gptq  # GPTQModel GPTQ conversion dependencies
uv sync --extra fp8   # Transformers fine-grained FP8 dependencies
uv sync --extra mlx   # exact Darwin-arm64 Stories15M MLX path
uv sync --extra modal # exact manual-large Inkling Modal control plane
```

The Modal extra supports the separate exact-Inkling manual-large workflow. It accepts only the
official pinned checkpoint and stock `Q3_K_M`; it cannot silently substitute another model or claim
the unreproducible `UD-Q3_K_XL` recipe. Start with the metadata-only preflight, which downloads no
weights and starts no remote Function:

```console
uv run python scripts/preflight_inkling_gguf.py \
  --config configs/experiments/inkling_q3_k_m_modal.yaml
```

Deployment and execution remain gated on explicit operator confirmation. The supported path deploys
once under a control-hash-specific App name, seals its newest
unique tagged history row and all five concrete `fu-*` Function-ID/exact-name pairs, and invokes
every ordered stage through the checked local manager; direct `modal run` is intentionally
unsupported. Live Modal 1.5 inspection returned an empty `definition_id` from both deployed
Function lookup and App layout, so the receipt intentionally makes no definition-ID claim. Before
each launch the manager rehydrates the Function-ID/name pair, revalidates the newest unique tagged
history, and the remote entrypoint validates its mounted script/package bytes against the sealed
control-plane manifest. This stays compatible with Modal Starter and does not require the Team-only
numeric-version lookup feature. Starter lookup is not an atomic version pin: the implementation-
addressed App name must have one exclusive deployer for the run, while the manager detects ordinary
redeployment before and after call acceptance and requests cancellation on drift. The manager reads
committed Volume markers to reject completed or out-of-order stages before container startup and
blocks locally recorded pending duplicates or predecessors, including a predecessor whose success
marker is already visible. The checked
[experiment configuration](configs/experiments/inkling_q3_k_m_modal.yaml),
[local manager](scripts/manage_inkling_modal.py), and
[remote stages](scripts/quantize_inkling_modal.py) define the commands, crash-safe receipt checks,
separate BF16 projector, explicit MTP omission, and non-claims. The launcher enforces configured
execution windows and deletion lag. Crash-rescheduled containers that reach the ledger consume a
new slot through required Modal task identity. The implementation-addressed App must be stopped at
the cutoff and source/work Volumes deleted promptly after final verification.

A deploy-only `--accept-short-initial-window-risk` exception can waive the initial full-sequence
fit check when its explicit confirmation and minimum-time checks pass. It does not weaken launch,
remote, continuation, or per-stage cutoffs, or guarantee completion.
Modal 1.5.0 and GPTQModel 5.8.0 require incompatible protobuf major versions, so `uv` explicitly
marks the `modal` extra as mutually exclusive with the `awq` and `gptq` extras. Use separate
environments for those independently pinned workflows.

The AWQ, CUDA-GPTQ, and FP8 YAML files remain capability fragments rather than runnable experiments.
On a compatible CUDA host, the opt-in test below probes dependencies, versions, device capability,
immutable identity, and backend configuration only. It does not load or convert a model:

```console
uv sync --extra dev --extra awq  # substitute gptq or fp8 as needed
IQL_RUN_OPTIONAL_BACKEND_PROBES=1 uv run pytest -q \
  tests/system/test_optional_backend_capability_probes.py -m backend_awq
```

The exception is the exact offline Stories15M CPU GPTQ pilot. It requires the immutable model
revision to be present in the local Hub cache and the exact dependency matrix pinned in
[`uv.lock`](uv.lock):

```console
uv sync --extra dev --extra gptq
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run iql run \
  configs/experiments/hf_stories15m_gptq_cpu_pilot.yaml
```

That checked experiment explicitly selects `hf_causal_lm_linear_mixtral`, the exact Stories15M
Defuser-linear adapter supplied by the GPTQ extra. Ordinary `hf_causal_lm` remains native-fused and
is not implicitly patched. The experiment quantizes only the 24 attention projections to symmetric
INT4/group-32; all expert, router, embedding, normalization, rotary, and output-head modules remain
float32.
Candidate evaluation reconstructs the published safetensors through the project-owned strict loader
and rejects state, policy, dependency, source, export-config, or global-binding drift. The canonical
execution also binds its runnable project bytes through `environment.project_source`, a 94-file SHA-256
manifest covering `pyproject.toml`, `uv.lock`, and `src/inkling_quant_lab`. The recorded four-sample,
17-token pilot observed worse candidate loss; it is conversion/reload evidence, not representative
quality, checkpoint-compression, routing, performance, memory, or energy evidence. See the
[curated evidence record](docs/experiments/stories15m-gptq-cpu-pilot.json).

The opt-in offline TC-QUANT-007 representation test has also passed under the exact dependency
matrix. It proves that all 117 raw checkpoint tensors, including 72 expert tensors, map without
value loss into both the 57-tensor native fused state and the 117-tensor Defuser-linear state:

```console
IQL_RUN_STORIES15M_REPRESENTATION_EQUIVALENCE=1 \
  uv run pytest -q tests/system/test_stories15m_representation_equivalence.py
```

## Quick start

The local CPU fixture workflow has been exercised end to end in this workspace. This workspace has
no `.git` metadata, so those runs are not clean-checkout release evidence. Each environment record
still includes a content-addressed manifest of the runnable package source and dependency lock;
that preserves source-byte identity without inventing Git ancestry or dirty state. Repeat the
sequence from a clean checkout and retain its output when producing a release artifact.

Inspect local devices and registered optional components:

```console
uv run iql doctor
uv run iql doctor --json
```

Validate a composed experiment before loading weights:

```console
uv run iql validate configs/experiments/tiny_moe_int8.yaml
```

`iql validate` checks configuration resolution and schema invariants only. Its output explicitly
reports that runtime, model, and backend capabilities were not probed. Likewise, `iql doctor`
lists optional registry entries as `not probed`; neither command is conversion evidence.

Run the requested CPU MoE INT8 experiment:

```console
uv run iql run configs/experiments/tiny_moe_int8.yaml
```

Run a compatible baseline and other local candidate:

```console
uv run iql run configs/experiments/tiny_moe_baseline.yaml
uv run iql run configs/experiments/tiny_moe_int4.yaml
```

Exercise the native CPU paths when their real kernel probes pass:

```console
uv run iql validate configs/experiments/tiny_moe_native_int8.yaml
uv run iql run configs/experiments/tiny_moe_native_int8.yaml
uv run iql validate configs/experiments/tiny_moe_native_int4.yaml
uv run iql run configs/experiments/tiny_moe_native_int4.yaml
```

The controlled native-versus-reference linear benchmark is reproducible separately from the
end-to-end pipeline:

```console
uv run python scripts/benchmark_native_cpu_quant.py \
  --hardware-label "<CPU, core count, memory>" \
  --host-isolation "<isolation/background-load conditions>"
```

The checked-in Apple M3/PyTorch 2.13.0 record measured 0.2706 ms native versus 1.8139 ms reference
for dynamic INT8 (6.704x), and 0.07189 ms native versus 2.8291 ms reference for INT4 (39.354x), at
the exact 8-by-1024 input, 1024-by-1024 weight, and one-thread protocol. INT8 state size was
identical; opaque native INT4 state was 0.77% larger. The native INT4 kernel is W4A8 while the
reference is W4A16, and tiny-MoE end-to-end runs can still be slower when fixed kernel-call costs
dominate. The host was shared and background load was not controlled. See the
[benchmark record](docs/experiments/native-cpu-kernels-m3-torch-2.13.json).

Use the run directory printed by `iql run` for resume, comparison, and report generation:

```console
uv run iql resume artifacts/<run-id>
uv run iql compare artifacts/<baseline-run-id> artifacts/<candidate-run-id>
uv run iql report artifacts/<run-id-or-comparison>
```

Inspect a model configuration without silently enabling remote code:

```console
uv run iql inspect-model configs/models/tiny_moe.yaml
```

Run the fast, pinned public Mixtral contract (about 3 MB including tokenizer files):

```console
uv sync --extra dev --extra hf
uv run iql inspect-model configs/models/hf_mixtral_tiny_random.yaml
uv run pytest -q tests/system/test_public_hf_moe.py -k tiny_random \
  -m "network and model_public"
```

The stronger trained-weight target downloads a 72,744,704-byte safetensors file:

```console
uv run iql inspect-model configs/models/hf_stories15m_moe.yaml
uv run pytest -q tests/system/test_public_hf_moe.py -k stories \
  -m "network and model_public"
```

Both configs pin immutable Hub commits, force native Transformers code and safetensors, and keep
remote code disabled. The fast model is random contract data. The trained target repeats
TinyStories-trained expert weights, but its router was randomly initialized, so it is not evidence
of learned routing specialization.

Measure the trained target's baseline causal quality on the pinned official TinyStories validation
file after verifying that file's SHA-256:

```console
uv run python scripts/evaluate_public_moe_quality.py \
  configs/experiments/hf_stories15m_tinystories_quality.yaml \
  /path/to/TinyStories-valid.txt \
  --expected-dataset-sha256 94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4
```

The recorded 32-story CPU run evaluated 5,826 causal-loss tokens at mean NLL 1.621984 and
perplexity 5.063128. This is baseline-only, subset-specific evidence; it is not a quantization
comparison. See the
[quality record](docs/experiments/stories15m-moe-tinystories-quality.json) for exact identities and
limitations.
This purpose-built evaluator rejects any other model/data revision, file digest, quantizer, runtime,
remote-code setting, or prompt template before model execution.

A separate aligned public-model slice quantized exactly the 24 attention `q/k/v/o` linears while
leaving every fused expert, router, embedding, normalization, and output-head tensor in float32:

| Candidate | Perplexity | Relative change | Greedy exact match | Route agreement | Tensor-state reduction |
|---|---:|---:|---:|---:|---:|
| Native dynamic INT8 | 5.077201 | +0.278% | 4/4 | 98.14% | 4.09% |
| Native KleidiAI INT4 | 6.300933 | +24.45% | 0/4 | 83.74% | 4.73% |

Three independent executions reproduced all scientific fields exactly. This is partial native-linear
evidence on one subset, not expert-weight quantization, full-dataset quality, peak-memory evidence,
or preservation of learned routing—the checkpoint router is publisher-randomized. See the
[compact record](docs/experiments/stories15m-moe-native-linear-quality.json) and
[curated full evidence](docs/experiments/stories15m-moe-native-linear-quality-full-v2.json).

The later confirmatory native-INT8 result remains valid for its native fused, attention-only
scope. An exact audit reconstructed Transformers 5.12's 117-source-tensor to 57-fused-tensor
conversion and reproduced the state checksum from both confirmatory attempts. The current GPTQ
source checksum differs because its Defuser model uses unfused expert names/layout, not because the
native runs omitted experts. The four-sample/17-token GPTQ pilot does not broaden that confirmatory
result's claim.

The same pinned Stories15M/native-INT8 policy has also completed one governed 14-stage CPU run with
fresh isolated benchmark workers. The baseline worker (PID 12534) reached 948,174,848 bytes peak
RSS; the candidate worker (PID 12648) reached 1,199,210,496 bytes, a neutral delta of
+251,035,648 bytes (+26.48%). The candidate safe bundle was 139,505,845 bytes versus
145,441,121 bytes for the baseline representation, a 4.08% reduction. Its median end-to-end
latency was 185.03 ms versus 160.57 ms, so this run supplies neither a speedup nor a live-memory
reduction claim. Candidate reconstruction can hold source and candidate state together, and the
three-prompt/17-loss-token local fixtures are contract data rather than representative quality
evidence. See the
[checked run record](docs/experiments/stories15m-native-int8-isolated-peak-m3.json).

After caching that exact revision, reproduce the governed run without network access:

```console
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run iql run \
  configs/experiments/hf_stories15m_native_int8_isolated_peak.yaml --json
```

A later governed run closes that run's candidate-reconstruction and export-not-executed caveats
without rewriting its evidence. Candidate evaluation, routing, and benchmarking loaded the
checksummed exported bundle through a metadata-derived empty Mixtral shell: provenance records
`source_weights_loaded=false`, strict assignment of 81 tensors, 24 packed qnnpack INT8 attention
wrappers, and no remaining meta tensors. The load still requires four exact cached config/tokenizer
files, so it is source-weight-free rather than fully source-free or self-contained.

At the distinct subject-artifact process boundary, baseline peak RSS was 630,423,552 bytes and
candidate peak RSS was 552,615,936 bytes (-77,807,616, or -12.3421%). That scope includes imports,
integrity validation, config/tokenizer and architecture construction, one subject's weights,
prepacking, warm-up, and trials; it excludes cleanup, result IPC, and exit. It is not steady-state,
tensor-attributable, or final-through-exit deployment residency. The candidate bundle was 4.0809%
smaller, but median end-to-end latency was slower (134.366 versus 115.651 ms) and median throughput
lower (89.308 versus 103.760 output tokens/s), so there is no speedup claim. Load operation kinds
differ and their delta is unavailable.

Full routing aligned 126 token-layer events (0.952381 route agreement, 0.976190 top-k overlap,
0.000647304 Jensen-Shannon divergence). The 17-token loss and three-prompt generation fixtures are
contract data; both exact-match scores were zero while output hashes differed, so this is not
representative quality preservation. Fused experts remained float32. See the
[checked record](docs/experiments/stories15m-native-int8-source-weight-free-peak-m3.json).

```console
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run iql run \
  configs/experiments/hf_stories15m_native_int8_source_weight_free_peak.yaml --json

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
IQL_RUN_PUBLIC_NATIVE_SOURCE_WEIGHT_FREE_PEAK=1 \
uv run pytest -q tests/integration/test_public_native_source_weight_free_peak_pipeline.py
```

A separate external MLX-LM slice converted every eligible leaf, including all 18 fused expert
projections, and safely reloaded q4 and q8 safetensors checkpoints on Apple Metal:

| Candidate | Perplexity | Relative change | Greedy exact match | Unordered route agreement | Tensor-byte reduction |
|---|---:|---:|---:|---:|---:|
| MLX affine q8 g32 | 5.062541 | -0.01346% | 4/4 | 98.9473% | 68.7429% |
| MLX affine q4 g32 | 6.245815 | +23.3565% | 0/4 | 85.2367% | 81.2416% |

This is a uniform all-eligible-leaf policy: embedding, output head, routers, attention, and experts
all change, so no delta is attributed specifically to expert weights. The q8 value is an observed
single-run subset difference, not an improvement. The publisher-randomized router prevents learned
specialization claims. This frozen record remains direct MLX evidence rather than SGLang, CUDA, or
sharding evidence; the later registered pipeline promoted only its exact model/software path. See
the [exact checked record](docs/experiments/stories15m-moe-mlx-full-quantization-m3.json).

The registered exact path composes `mlx_lm_mixtral`, `mlx_metal`, and `mlx_affine` through ordinary
pipeline stages. It requires the absolute audited local snapshot, float32 control execution, q4 or
q8 affine group-size-32 uniform conversion, and Apple Metal. Both checked configs completed loss,
six-layer full routing, reports, atomic safetensors export, byte audit, reload, and a packed-kernel
forward. Pipeline benchmarks are rejected because that loop is still PyTorch-specific; use the
checked direct-MLX record above for bounded timing observations.

```console
uv sync --extra dev --extra mlx
uv run --extra mlx iql run configs/experiments/mlx_stories15m_moe_q4.yaml \
  --set model.local_snapshot_path=/absolute/path/to/snapshots/b6dd737497465570b5f5e962dbc9d9454ed1e0eb
uv run --extra mlx iql run configs/experiments/mlx_stories15m_moe_q8.yaml \
  --set model.local_snapshot_path=/absolute/path/to/snapshots/b6dd737497465570b5f5e962dbc9d9454ed1e0eb
```

The default evaluation/routing inputs are tiny contract fixtures, not representative quality data.
The uniform policy changes embedding, head, routers, attention, and experts together, and the
publisher-randomized router prevents learned-specialization claims. See the
[registered-pipeline record](docs/experiments/stories15m-moe-registered-mlx-pipeline-m3.json).

The learned-router slice is separate. Its CPU-only contracts restrict an accepted learned-float
artifact to the six `(4, 288)` gate weights, then represent each gate as affine
group-size-32 q4 (`uint32[4,36]` packed weight) or q8 (`uint32[4,72]`) plus `float32[4,9]` scales and
affine biases. Each overlay has exactly 18 tensors and 5,184 q4 or 8,640 q8 raw tensor bytes. Its
manifest binds every tensor fact to the learned-float lineage and rejects substitution,
missing/extra keys, or byte tampering before safetensors-only reconstruction.

The unchanged ten-step contract did execute on Apple M3 Metal. It reduced held-out cross-entropy by
`1.91856` and improved exact-top-2 accuracy from `0.14616` to `0.48177`, but failed the predeclared
overall `0.60` and Alice `0.50` accuracy gates (Alice observed `0.35156`). Seven other overall and
per-domain gates passed. The runner sealed this as a negative result and left the configured success
path absent: no overlay, lineage, q4/q8 descendant, or retention claim was produced. See the
[checked negative-result record](docs/experiments/stories15m-router-domain-pair-10step-failed-m3.json).

The existing uniform MLX results above use the publisher-randomized router and cannot fill the
learned-behavior evidence gap. Even a future passing learned run supports only its predeclared
label-conditioned routing statement; packing error and reload success are not retention
measurements. See the reviewed
[preflight configuration](configs/post_training/stories15m_router_10step.yaml).

A separate two-rank CPU/Gloo slice executed every expert block in the pinned Stories15M checkpoint.
Each rank loaded exactly one half of every fused expert projection, all 144 local source-slice
reconstructions matched their float32 source hashes, every expert received top-2 traffic, and the
all-reduced outputs matched the single-process reference within the declared tolerance. This proves
expert-block tensor parallelism only: attention, embeddings, norms, residuals, routers, and the LM
head were not executed as a sharded end-to-end transformer, and the slice makes no performance,
peak-memory, energy, training, export, multi-host, or Inkling-checkpoint claim. See the
[exact checked record](docs/experiments/stories15m-moe-expert-tensor-parallel-m3.json).

SGLang 0.5.15.post1 also served this MoE through MLX on Apple Metal and returned one real HTTP 200
generation. That is a serving smoke, not a latency distribution or an Inkling Quant Lab runtime.
The upstream MLX path did not forward SGLang's `--revision` to its weight loader; the observed Hub
ref happened to resolve to the requested commit. Future runs must inspect the snapshot for custom
Python and pass its absolute immutable path. After starting the pinned server, collect repeated
content-redacted trials with an untracked prompt file. Capture a redacted server-environment JSON
from the serving process and host; the probe requires OS, hardware, Python, backend, and
runtime-dependent package versions so its timing observations are not detached from provenance:

```json
{
  "captured_at_utc": "2026-07-16T04:00:00Z",
  "platform": {"system": "Darwin", "release": "24.3.0", "machine": "arm64"},
  "hardware": {"description": "Apple M3, 10 GPU cores", "logical_cpu_count": 8},
  "python": {"version": "3.12.7", "implementation": "CPython"},
  "software": {
    "sglang": "0.5.15.post1", "torch": "2.11.0", "transformers": "5.12.1",
    "mlx": "0.32.0", "mlx-lm": "0.31.3"
  },
  "capture_method": "versions queried inside the serving virtual environment"
}
```

```console
uv run python scripts/probe_serving_endpoint.py \
  --base-url http://127.0.0.1:43440 \
  --prompt-file /path/to/untracked-prompt.txt \
  --model-id ggml-org/stories15M_MOE \
  --endpoint-model-path /absolute/path/to/snapshots/b6dd737497465570b5f5e962dbc9d9454ed1e0eb \
  --model-revision b6dd737497465570b5f5e962dbc9d9454ed1e0eb \
  --model-checksum 93e36334ff1be21096ca5f59c6b4d8bdfb212c8854b815583110755df75d6ed9 \
  --backend sglang --backend-version 0.5.15.post1 \
  --backend-revision 0b3bb0cbe31873994c9f989fddfe2f87ca839fdd \
  --runtime mlx_on_apple_metal \
  --server-environment-file /path/to/server-environment.json \
  --seed 17
```

See the [one-request record](docs/experiments/sglang-mlx-stories15m-smoke.json) for the exact install
transformation, cache/blob evidence, safety caveats, and failed synthetic-model probe. The initial
vLLM 0.23.0 Apple-CPU source build imported its native extensions, but the managed environment
denied the local socket bind required before weight loading; that historical
[blocker record](docs/experiments/vllm-0.23.0-apple-cpu-build-blocker.json) remains accurate for
the restricted environment. A subsequent offline run where single-rank Gloo TCP was permitted used
an exact checked compatibility patch, loaded the pinned Stories15M MoE safetensors, and produced
three identical four-token greedy completions whose token-ID hashes exactly matched native
Transformers. See the
[execution record](docs/experiments/vllm-0.23.0-apple-cpu-stories15m-inference.json). This is
external direct-inference evidence, not unpatched upstream compatibility, a project runtime,
serving, or a performance result.

Automation-sensitive commands expose `--json` where supported. Failures return nonzero and include a
stable error code, component/stage context, and remediation when known.

## Experiments and configuration

Experiments compose checked-in model, quantizer, and evaluation fragments:

```yaml
extends:
  - ../models/tiny_moe.yaml
  - ../quantization/dynamic_int8.yaml
  - ../evaluations/tiny_cpu.yaml
schema_version: "1.0"
name: tiny-moe-int8
seed: 17
```

Resolution order is component bases, experiment YAML, CLI overrides, then in-memory secret
references. The complete resolved config is validated, serialized, saved, and hashed. There are no
hidden environment-dependent defaults for device, dtype, remote code, or checkpoint safety.

See the validated [configuration models](src/inkling_quant_lab/config.py) and
[checked-in configuration fragments](configs/) for fields, policy precedence,
calibration/evaluation separation, routing modes, and overrides.

## Run artifacts

The run contract is:

```text
artifacts/<run_id>/
├── manifest.json
├── resolved_config.yaml
├── environment.json
├── events.jsonl
├── status.json
├── metrics/
├── routing/
├── checkpoints/
├── reports/
├── failures/                   # structured stage failures, when present
├── failure_reports/evaluation/ # required-evaluation failure reports, when present
├── archive/forced/             # forced-stage transaction evidence, when present
└── completion.json             # successful finalization only
```

Required secrets resolve before directory creation, so a missing required secret leaves no partial
run. Once a run directory is created, the five top-level provenance/control files and the four
category roots are materialized before governed stages execute. `environment.json` is explicitly
marked with a pending capability probe until `probe_runtime` atomically replaces it with measured
capabilities; the initial manifest carries the same provisional environment so even an early
governed failure remains attributable. Its presence alone is not proof that probing succeeded.

- `manifest.json` is the authoritative run/stage ledger with timestamps, fingerprints, checksums,
  model/config identity, environment, warnings, and explicit status.
- `resolved_config.yaml` is the exact effective experiment input; secret values are never written.
- `environment.json` records Python, packages, platform, hardware, Git provenance where available,
  and a content-addressed `project_source` manifest covering `pyproject.toml`, `uv.lock`, and
  `src/inkling_quant_lab`.
- `events.jsonl` is an append-only redacted structured log; console logs remain human-readable.
- `status.json` makes current stage transitions and failures externally visible.
- `metrics/` retains evaluation and benchmark facts with exact dataset digests, selected samples,
  prompt/decode/seed workload identity, evaluator/protocol metadata, and memory collector scope.
- `routing/` separates full-denominator aggregates from sampled or full raw traces.
- `checkpoints/` contains safe exports or reconstruction recipes and quantization metadata.
- `failures/` retains the redacted structured error for every failed stage attempt.
- `metrics/evaluation_failures/` and `failure_reports/evaluation/` retain typed results and
  attempt-scoped reports for required evaluator failures outside the canonical successful report
  stage, so retries never overwrite or discard them.
- `archive/forced/` contains the transaction ledger and checksummed prior outputs when
  `iql resume <run_dir> --force-stage <stage>` deliberately invalidates successful evidence.
- `reports/` contains machine-readable report data, Markdown, CSV tables, and SVG plots with
  adjacent CSV source data. When `reporting.html=true`, a configured run also emits a minimal
  escaped-Markdown `report.html`; it is not a rich HTML renderer.

Successful stage output is append-only. Resume verifies already-successful output and fingerprints
before topology recovery or publication adoption, then skips a stage only after those checks pass.
A forced stage archives prior output before invalidating that stage and its dependents.
Unsupported, skipped-not-required, failed, and successful are distinct states.

This current workspace does not contain `.git` metadata. Its manifests must therefore record Git
commit and dirty state as unavailable. The separate `project_source` record binds runnable source
bytes through a deterministic per-file and aggregate SHA-256 manifest, but it does not establish a
commit or clean checkout. Runs made from a real Git checkout record exact Git provenance; do not
invent a commit for this workspace.

## Comparison integrity

Runs compare only when model/revision, exact evaluation dataset digests, seed sets, prompt templates,
decoding, stable evaluation/routing samples, routing capture evidence, benchmark protocol, and
hardware/runtime placement are compatible. Benchmark comparison also requires the same exact
dataset/sample/prompt/decode/seed/execution-mode workload and memory collector kind/scope. Selected
unsafe overrides are recorded by dimension in the comparison and report; they do not make unlike
experiments scientifically equivalent.

Evaluation provenance is retained per canonically ordered suite, including duplicate evaluator
types, so swapping datasets, samples, prompt hashes, or decode settings between suites cannot hide
behind the same aggregate sets. Evaluation measurements use stable suite-scoped metric keys and
carry the complete evaluator/dataset/sample/seed provenance into JSON, CSV, and Markdown; ambiguous
duplicate producers cannot overwrite one another. Compatibility aliases choose their source from
the full suite contract before outcomes are known; a failed preferred suite stays unavailable
instead of falling back to a different metric. The headline `quality` metric deterministically
prefers exact match and then perplexity. Deltas and Pareto extraction refuse mixed metric sources.
Governed run and comparison reports must retain their respective
manifests; only an explicitly standalone `report_data.json` may regenerate without one.

Aggregate routing stays aggregate and retains no raw traces. Its distribution drift remains
available, while token agreement/top-k overlap are explicitly unavailable. Sampled/full-trace runs
persist capture mode, alignment-key count, and alignment-key SHA-256 with their normalized summary.

Reports retain failures and unavailable/unsupported measurements instead of coercing them to zero.
Behavioral retention is reported only from explicit base/fine-tuned rubric evidence and the
measured candidate score; the checked-in CPU fixture carries per-sample B/F evidence. Missing or
inconsistent evidence is shown as unavailable, never replaced with an assumed zero base score.
Optional evaluator failures remain typed inputs to a normal report. Required evaluator failures
retain attempt-scoped evaluation JSON and a failure report while the manifest stays failed.
Normalized benchmark summaries retain median and p90 values alongside mean, population standard
deviation, and normal-approximation 95% intervals for latency and throughput when at least two
measured repetitions are available.
The portable CPU runtime samples current RSS at a pre-warm-up baseline and after each
warm-up/measured trial on macOS/Linux. Reports retain the interval, count, maximum observed RSS, and
non-negative delta as neutral metrics. `peak_host_memory_bytes` remains unavailable because boundary
sampling is not a continuous peak collector. The opt-in
`benchmark.host_memory_mode=isolated_stage_worker_peak_rss` instead gives baseline and candidate
their own fresh spawned process and records its OS high-water RSS and PID immediately after the
measured trials. That exact high-water mark through the final benchmark boundary includes
interpreter startup, imports, model load or candidate reconstruction, size accounting, warm-up,
and trials; it excludes the parent, prior stages, and post-read cleanup/result transport. It is not
a final through-exit process peak, steady-state-only memory, or candidate-attributable deployment
memory. A runnable example is
`configs/experiments/tiny_moe_native_int8_isolated_peak.yaml`; the pinned offline public-model
example is `configs/experiments/hf_stories15m_native_int8_isolated_peak.yaml` and requires the exact
revision in the local Hugging Face cache. Its checked M3 run observed a higher candidate-worker
peak because reconstruction can retain source and candidate state simultaneously; do not interpret
that worker-history value as source-free deployed residency.
The later `isolated_subject_artifact_peak_rss` mode gives each worker exactly one persisted subject.
Its candidate verifies the governed export and loads it without float source weights, while still
requiring exact cached config/tokenizer metadata. The measurement kind is
`benchmark_subject_artifact_worker_process_peak_rss`; use
`configs/experiments/hf_stories15m_native_int8_source_weight_free_peak.yaml` and retain its full
scope rather than relabeling it as steady-state or tensor-attributable deployment memory.
Pareto membership means non-dominated under the configured objective directions and tolerances, not
that a candidate is universally best or deployable.

## Security defaults

- Remote model code is disabled. Programmatic configuration requires both
  `model.trust_remote_code=true` and `security.allow_remote_code=true`. CLI `run` and
  `inspect-model` additionally require `--allow-remote-code`; that flag sets both resolved fields
  for the execution, while fields supplied without the flag are rejected. `resume` requires the
  flag again for a persisted remote-code run.
- Pickle-based checkpoint formats are rejected unless explicitly allowed.
- External uploads, prompt logging, and model-output logging are disabled.
- Secrets are environment-variable references and are redacted from logs.
- Artifact paths must remain under the configured root.
- Subprocesses use argument arrays with no shell interpolation.

Review the [security defaults implementation](src/inkling_quant_lab/security.py) before adding
external models, datasets, or backends.

## Development commands

The default CPU-only quality-gate usage is:

```console
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest -m "not network and not gpu and not slow and not large_model"
```

GPU, network, public-model, backend, and large-model tests are opt-in and must pin their external
revisions and document hardware/software requirements.

## Repository map

- [Source package](src/inkling_quant_lab/)
- [Experiment configurations](configs/)
- [Operational scripts](scripts/)
- [Tests](tests/)
- [Curated experiment records](docs/experiments/)
