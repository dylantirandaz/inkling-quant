from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from inkling_quant_lab.config import ExperimentConfig
from inkling_quant_lab.pipeline.runner import run_experiment
from inkling_quant_lab.registry import (
    EVALUATORS,
    MODEL_ADAPTERS,
    QUANTIZERS,
    REPORTERS,
    RUNTIMES,
    Registry,
)

pytestmark = pytest.mark.integration


def _track_creations(
    monkeypatch: pytest.MonkeyPatch,
    registry: Registry[Any],
    calls: list[str],
) -> None:
    original = registry.create

    def tracked(name: str, *args: Any, **kwargs: Any) -> Any:
        calls.append(name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(registry, "create", tracked)


def test_pipeline_resolves_every_selected_component_through_lazy_registries(
    config_factory: Callable[..., ExperimentConfig],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = config_factory(
        model_id="local://fixtures/tiny-dense",
        backend="noop",
        precision="float32",
        routing_mode="off",
    ).model_copy(
        update={
            "benchmark": config_factory().benchmark.model_copy(update={"enabled": False}),
            "output": config_factory().output.model_copy(update={"root": str(tmp_path)}),
            "reporting": config_factory().reporting.model_copy(update={"plots": False}),
        }
    )
    calls: dict[str, list[str]] = {
        "model": [],
        "quantizer": [],
        "runtime": [],
        "evaluator": [],
        "reporter": [],
    }
    _track_creations(monkeypatch, MODEL_ADAPTERS, calls["model"])
    _track_creations(monkeypatch, QUANTIZERS, calls["quantizer"])
    _track_creations(monkeypatch, RUNTIMES, calls["runtime"])
    _track_creations(monkeypatch, EVALUATORS, calls["evaluator"])
    _track_creations(monkeypatch, REPORTERS, calls["reporter"])

    run_experiment(config, project_root=tmp_path, run_id="registry-execution")

    assert calls["model"] and set(calls["model"]) == {"local_fixture"}
    assert calls["quantizer"] and set(calls["quantizer"]) == {"noop"}
    assert calls["runtime"] and set(calls["runtime"]) == {"torch_eager_cpu"}
    assert calls["evaluator"] == [
        "perplexity",
        "generation_regression",
        "perplexity",
        "generation_regression",
    ]
    assert calls["reporter"] == ["markdown"]
