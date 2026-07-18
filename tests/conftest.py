from __future__ import annotations

from collections.abc import Callable

import pytest

from inkling_quant_lab.config import ExperimentConfig


@pytest.fixture
def config_factory() -> Callable[..., ExperimentConfig]:
    def build(
        *,
        model_id: str = "local://fixtures/tiny-moe",
        backend: str = "noop",
        precision: str = "float32",
        routing_mode: str = "aggregate",
    ) -> ExperimentConfig:
        return ExperimentConfig.model_validate(
            {
                "schema_version": "1.0",
                "name": "test-experiment",
                "seed": 17,
                "model": {
                    "model_id": model_id,
                    "revision": "fixture-v1",
                    "adapter": "local_fixture",
                    "dtype": "float32",
                    "trust_remote_code": False,
                    "local_files_only": True,
                    "checkpoint_format": "fixture",
                },
                "quantization": {
                    "backend": backend,
                    "method": "none" if backend == "noop" else "weight_only",
                    "policy": {
                        "type": "uniform",
                        "default_precision": precision,
                        "module_class_rules": [
                            {
                                "name": "preserve_normalization",
                                "pattern": "*LayerNorm",
                                "precision": "float32",
                            }
                        ],
                        "preserve_router_precision": True,
                        "preserve_output_head": True,
                    },
                    "export": {"format": "recipe_json", "destination": "candidate"},
                },
                "evaluation": {
                    "suites": [
                        {
                            "type": "perplexity",
                            "dataset": "local://fixtures/tiny-corpus",
                            "revision": "fixture-data-v1",
                            "split": "evaluation",
                        },
                        {
                            "type": "generation_regression",
                            "dataset": "local://fixtures/generation-prompts",
                            "revision": "fixture-data-v1",
                            "split": "evaluation",
                        },
                    ]
                },
                "routing": {"mode": routing_mode, "required": False},
                "benchmark": {"warmup_iterations": 1, "repetitions": 2},
                "runtime": {
                    "backend": "torch_eager_cpu",
                    "device": "cpu",
                    "dtype": "float32",
                    "device_map": "single",
                },
                "output": {"root": "artifacts", "run_prefix": "test"},
                "security": {
                    "allow_remote_code": False,
                    "allow_pickle_weights": False,
                    "allow_uploads": False,
                    "log_prompts": False,
                    "log_model_outputs": False,
                },
                "reporting": {"markdown": True, "html": False, "plots": True},
            }
        )

    return build
