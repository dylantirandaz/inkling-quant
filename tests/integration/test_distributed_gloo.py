"""Real local two-process collective and tiny tensor-parallel evidence."""

from __future__ import annotations

import pytest

from inkling_quant_lab.distributed import (
    probe_gloo_capability,
    run_local_cpu_tensor_parallel_smoke,
    run_local_gloo_smoke,
)

pytestmark = pytest.mark.integration


def test_gloo_probe_never_promotes_collectives_to_model_sharding() -> None:
    capability = probe_gloo_capability()

    assert capability.supports_model_sharding is False
    assert capability.supports_collectives is (capability.status == "available")
    assert capability.availability_kind == "compiled_backend"
    assert capability.smoke_tested is False
    assert "does not validate model sharding" in capability.scope


@pytest.mark.slow
def test_local_two_process_gloo_all_reduce_smoke_is_truthfully_scoped() -> None:
    capability = probe_gloo_capability()
    if capability.status != "available":
        pytest.skip(capability.reason or "gloo unavailable")

    result = run_local_gloo_smoke(timeout_seconds=20.0)

    assert result.world_size == 2
    assert result.rank_results == (3.0, 3.0)
    assert result.expected_sum == 3.0
    assert result.interface in {"lo", "lo0"}
    assert result.supports_model_sharding is False
    assert "does not validate model sharding" in result.scope


@pytest.mark.slow
def test_local_two_process_cpu_tensor_parallel_forward_is_parameter_sharded() -> None:
    capability = probe_gloo_capability()
    if capability.status != "available":
        pytest.skip(capability.reason or "gloo unavailable")

    result = run_local_cpu_tensor_parallel_smoke(
        hardware_label="operator-declared integration-test CPU host",
        timeout_seconds=30.0,
    )

    assert result.world_size == 2
    assert result.parameter_sharding_executed is True
    assert result.tiny_model_forward_validated is True
    assert result.public_model_sharding_validated is False
    assert result.distributed_training_validated is False
    assert result.performance_validated is False
    assert "does not validate public-model or MoE sharding" in result.scope
    rank_zero, rank_one = result.rank_results
    assert rank_zero.tensor_parallel_output_sha256 == rank_one.tensor_parallel_output_sha256
    assert rank_zero.max_abs_error <= result.absolute_tolerance
    assert rank_one.max_abs_error <= result.absolute_tolerance

    zero_up, zero_down = rank_zero.parameter_shards
    one_up, one_down = rank_one.parameter_shards
    assert zero_up.global_shape == one_up.global_shape == (12, 8)
    assert zero_up.local_shape == one_up.local_shape == (6, 8)
    assert zero_up.placement == one_up.placement == "Shard(dim=0)"
    assert zero_up.shard_range == (0, 6)
    assert one_up.shard_range == (6, 12)
    assert zero_down.global_shape == one_down.global_shape == (5, 12)
    assert zero_down.local_shape == one_down.local_shape == (5, 6)
    assert zero_down.placement == one_down.placement == "Shard(dim=1)"
    assert zero_down.shard_range == (0, 6)
    assert one_down.shard_range == (6, 12)
    for zero_shard, one_shard in ((zero_up, one_up), (zero_down, one_down)):
        assert zero_shard.local_sha256 != one_shard.local_sha256
        assert zero_shard.reconstructed_sha256 == one_shard.reconstructed_sha256
        assert zero_shard.reconstructed_sha256 == zero_shard.source_full_sha256
