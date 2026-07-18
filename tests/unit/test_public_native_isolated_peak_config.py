from __future__ import annotations

from pathlib import Path

from inkling_quant_lab.config import load_config
from inkling_quant_lab.data import load_local_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT_ROOT / "configs/experiments/hf_stories15m_native_int8_isolated_peak.yaml"


def test_public_native_isolated_peak_config_is_offline_safe_and_exact() -> None:
    config = load_config(CONFIG)

    assert config.model.model_id == "ggml-org/stories15M_MOE"
    assert config.model.revision == "b6dd737497465570b5f5e962dbc9d9454ed1e0eb"
    assert config.model.adapter == "hf_causal_lm"
    assert config.model.local_files_only is True
    assert config.model.checkpoint_format == "safetensors"
    assert config.model.trust_remote_code is False
    assert config.security.allow_remote_code is False
    assert config.security.allow_uploads is False
    assert config.security.log_prompts is False
    assert config.security.log_model_outputs is False
    assert config.runtime.backend == "torch_eager_cpu"
    assert config.runtime.device == "cpu"
    assert config.runtime.dtype == "float32"
    assert config.runtime.device_map == "single"
    assert config.runtime.sharding is None


def test_public_native_isolated_peak_config_selects_only_addressable_int8_linears() -> None:
    config = load_config(CONFIG)
    policy = config.quantization.policy

    assert config.quantization.backend == "torch_native_dynamic_int8"
    assert config.quantization.method == "native_dynamic_w8a8"
    assert policy.type == "uniform"
    assert policy.default_precision == "float32"
    assert len(policy.module_class_rules) == 1
    assert policy.module_class_rules[0].name == "quantize-addressable-linear"
    assert policy.module_class_rules[0].pattern == "torch.nn.modules.linear.Linear"
    assert policy.module_class_rules[0].precision == "int8"
    assert policy.preserve_router_precision is True
    assert policy.router_precision == "float32"
    assert policy.preserve_output_head is True
    assert policy.output_head_precision == "float32"
    assert policy.preserve_embeddings is True
    assert policy.embedding_precision == "float32"
    assert config.quantization.export.enabled is True
    assert config.quantization.export.format == "safetensors"


def test_public_native_isolated_peak_config_uses_distinct_workers_and_checked_fixtures() -> None:
    config = load_config(CONFIG)

    assert config.benchmark.enabled is True
    assert config.benchmark.protocol_version == "public-moe-cpu-isolated-stage-peak-v1"
    assert config.benchmark.host_memory_mode == "isolated_stage_worker_peak_rss"
    assert config.benchmark.warmup_iterations == 1
    assert config.benchmark.repetitions == 3
    assert config.benchmark.measure_energy is False
    assert config.routing.mode == "full_trace"
    assert config.routing.required is True
    assert config.routing.capture_router_logits is False
    assert config.routing.dataset == "local://fixtures/routing-prompts"
    assert [suite.type for suite in config.evaluation.suites] == [
        "perplexity",
        "generation_regression",
    ]
    for suite in config.evaluation.suites:
        dataset = load_local_dataset(suite.dataset, suite.revision, suite.split)
        assert dataset.sha256
        assert dataset.samples
