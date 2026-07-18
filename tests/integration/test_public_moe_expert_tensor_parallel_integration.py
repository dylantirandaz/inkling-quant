"""Opt-in integration execution of the pinned public-MoE tensor-parallel contract."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from inkling_quant_lab.public_moe_tensor_parallel import (
    MODEL_REVISION,
    run_public_moe_expert_tensor_parallel,
)

pytestmark = [pytest.mark.integration, pytest.mark.model_public, pytest.mark.slow]


def test_public_stories15m_all_expert_blocks_execute_tensor_parallel() -> None:
    raw_snapshot = os.environ.get("INKLING_QUANT_STORIES15M_SNAPSHOT")
    if raw_snapshot is None:
        pytest.skip("set INKLING_QUANT_STORIES15M_SNAPSHOT to the pinned local snapshot")
    snapshot = Path(raw_snapshot).expanduser().resolve(strict=True)
    if snapshot.name != MODEL_REVISION:
        pytest.fail("INKLING_QUANT_STORIES15M_SNAPSHOT does not name the pinned revision")

    result = run_public_moe_expert_tensor_parallel(
        snapshot,
        hardware_label="operator-declared opt-in integration CPU",
    )

    assert result.public_checkpoint_executed is True
    assert result.all_public_moe_expert_blocks_tensor_parallel_validated is True
    assert result.end_to_end_transformer_forward_validated is False
    assert tuple(len(rank.shard_evidence) for rank in result.ranks) == (72, 72)
