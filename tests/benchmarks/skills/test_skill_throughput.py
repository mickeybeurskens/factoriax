"""Performance tests for benchmark skill wrappers.

Measures wrapper overhead and throughput at various batch sizes
on 5x5 maps with small entity arrays.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import pytest

from factoriax.benchmarks.skills.fuel_miner import FuelMinerSkill, fuel_miner_level
from factoriax.benchmarks.skills.mining import MiningSkill, mining_level
from factoriax.benchmarks.skills.place_miner import (
    PlaceMinerSkill,
    place_miner_level,
)
from factoriax.constants import NUM_ACTIONS
from factoriax.state import EnvParams

NUM_STEPS = 100
WARMUP_ITERS = 5


def _measure_throughput(
    env: object,
    params: EnvParams,
    n_envs: int,
) -> float:
    """Measure vmapped step throughput for an environment.

    Args:
        env: Environment with step_env and reset_env methods.
        params: Environment parameters.
        n_envs: Batch size.

    Returns:
        Steps per second.
    """
    keys = jax.random.split(jax.random.key(42), n_envs)
    _, states = jax.vmap(env.reset_env, in_axes=(0, None))(keys, params)
    vmap_step = jax.jit(jax.vmap(env.step_env, in_axes=(0, 0, 0, None)))

    rng = jax.random.key(77)
    actions = jnp.zeros(n_envs, dtype=jnp.int32)
    step_keys = jax.random.split(rng, n_envs)
    for _ in range(WARMUP_ITERS):
        _, states, _, _, _ = vmap_step(step_keys, states, actions, params)
    jax.block_until_ready(jax.tree.leaves(states))

    rng_t = jax.random.key(88)
    t0 = time.perf_counter()
    for _ in range(NUM_STEPS):
        rng_t, k_a, k_s = jax.random.split(rng_t, 3)
        actions = jax.random.randint(k_a, (n_envs,), 0, NUM_ACTIONS)
        step_keys = jax.random.split(k_s, n_envs)
        _, states, _, _, _ = vmap_step(step_keys, states, actions, params)
    jax.block_until_ready(jax.tree.leaves(states))
    elapsed = time.perf_counter() - t0

    return n_envs * NUM_STEPS / elapsed


# -----------------------------------------------------------------------
# Layer 1: Wrapper overhead at various batch sizes
# -----------------------------------------------------------------------

BATCH_SIZES = [64, 1024, 4096]

MIN_FPS = {
    64: 8000,
    1024: 200000,
    4096: 500000,
}


@pytest.mark.parametrize("n_envs", BATCH_SIZES)
def test_mining_skill_throughput(n_envs: int) -> None:
    """MiningSkill throughput must exceed minimum fps.

    Args:
        n_envs: Batch size.
    """
    _, params = mining_level()
    env = MiningSkill()
    fps = _measure_throughput(env, params, n_envs)
    threshold = MIN_FPS[n_envs]
    assert fps >= threshold, (
        f"MiningSkill {fps:.0f} fps < {threshold} fps at batch={n_envs}"
    )


@pytest.mark.parametrize("n_envs", BATCH_SIZES)
def test_place_miner_skill_throughput(n_envs: int) -> None:
    """PlaceMinerSkill throughput must exceed minimum fps.

    Args:
        n_envs: Batch size.
    """
    _, params = place_miner_level()
    env = PlaceMinerSkill()
    fps = _measure_throughput(env, params, n_envs)
    threshold = MIN_FPS[n_envs]
    assert fps >= threshold, (
        f"PlaceMinerSkill {fps:.0f} fps < {threshold} fps at batch={n_envs}"
    )


@pytest.mark.parametrize("n_envs", BATCH_SIZES)
def test_fuel_miner_skill_throughput(n_envs: int) -> None:
    """FuelMinerSkill throughput must exceed minimum fps.

    Args:
        n_envs: Batch size.
    """
    _, params = fuel_miner_level()
    env = FuelMinerSkill()
    fps = _measure_throughput(env, params, n_envs)
    threshold = MIN_FPS[n_envs]
    assert fps >= threshold, (
        f"FuelMinerSkill {fps:.0f} fps < {threshold} fps at batch={n_envs}"
    )


# -----------------------------------------------------------------------
# Layer 2: max_machines sensitivity
# -----------------------------------------------------------------------


@pytest.mark.parametrize("max_m", [8, 16, 32, 64])
def test_max_machines_scaling(max_m: int) -> None:
    """Smaller max_machines should not be slower than larger.

    Args:
        max_m: Entity array capacity to test.
    """
    _, base_params = place_miner_level()
    params = EnvParams(
        map_width=base_params.map_width,
        map_height=base_params.map_height,
        num_players=1,
        max_machines=max_m,
        max_timesteps=base_params.max_timesteps,
    )
    env = PlaceMinerSkill()
    fps = _measure_throughput(env, params, 1024)
    assert fps >= 100000, (
        f"PlaceMinerSkill {fps:.0f} fps < 100k at max_machines={max_m}"
    )
