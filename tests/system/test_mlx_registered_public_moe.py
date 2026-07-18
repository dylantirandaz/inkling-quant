"""Opt-in end-to-end contracts for the exact registered MLX public-MoE path."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from inkling_quant_lab.config import load_config
from inkling_quant_lab.manifests import RunStatus, load_manifest
from inkling_quant_lab.mlx_contract import MODEL_REVISION
from inkling_quant_lab.pipeline.runner import run_experiment

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
    pytest.mark.model_public,
    pytest.mark.slow,
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RUN_REGISTERED_MLX = os.environ.get("IQL_RUN_MLX_REGISTERED") == "1"
_EXTERNAL_ONLY = pytest.mark.skipif(
    not _RUN_REGISTERED_MLX,
    reason="set IQL_RUN_MLX_REGISTERED=1 and INKLING_QUANT_STORIES15M_SNAPSHOT",
)


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("bits", [4, 8])
@_EXTERNAL_ONLY
def test_registered_mlx_pipeline_quantizes_evaluates_and_traces_public_moe(
    tmp_path: Path, bits: int
) -> None:
    raw_snapshot = os.environ.get("INKLING_QUANT_STORIES15M_SNAPSHOT")
    if raw_snapshot is None:
        pytest.fail("set INKLING_QUANT_STORIES15M_SNAPSHOT to the pinned local snapshot")
    snapshot = Path(raw_snapshot).expanduser().resolve(strict=True)
    if snapshot.name != MODEL_REVISION:
        pytest.fail("INKLING_QUANT_STORIES15M_SNAPSHOT does not name the pinned revision")

    config = load_config(
        PROJECT_ROOT / "configs" / "experiments" / f"mlx_stories15m_moe_q{bits}.yaml",
        (
            f"model.local_snapshot_path={json.dumps(str(snapshot))}",
            f"output.root={json.dumps(str(tmp_path / 'artifacts'))}",
        ),
    )
    run_directory = run_experiment(
        config,
        project_root=PROJECT_ROOT,
        run_id=f"registered-mlx-q{bits}",
    )

    manifest = load_manifest(run_directory)
    assert manifest.status is RunStatus.SUCCESS
    assert manifest.stages["benchmark_baseline"].required is False
    assert manifest.stages["benchmark_candidate"].required is False

    quantization = _read_json(run_directory / "checkpoints/candidate/quantization_manifest.json")
    assert isinstance(quantization, dict)
    assert quantization["backend"] == "mlx_affine"
    assert quantization["method"] == f"mlx_affine_q{bits}_g32"
    assert quantization["quantization_parameters"]["quantized_leaf_count"] == 50
    assert quantization["quantization_parameters"]["quantized_fused_expert_projection_count"] == 18

    export = run_directory / "checkpoints/candidate/candidate"
    assert (export / "model.safetensors").is_file()
    assert (export / "inkling_quant_manifest.json").is_file()

    for kind in ("baseline", "candidate"):
        evaluation = _read_json(run_directory / f"metrics/evaluation_{kind}/results.json")
        assert isinstance(evaluation, list) and len(evaluation) == 1
        assert evaluation[0]["status"] == "success"
        routing = _read_json(run_directory / f"routing/{kind}/aggregates.json")
        assert isinstance(routing, dict)
        assert len(routing["aggregates"]) == 6
        assert all(layer["assignment_count"] > 0 for layer in routing["aggregates"].values())

    comparison = _read_json(run_directory / "routing/comparison.json")
    assert isinstance(comparison, dict)
    assert comparison["macro"]["layer_count"] == 6
    assert comparison["macro"]["token_weight"] > 0
