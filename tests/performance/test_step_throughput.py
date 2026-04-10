"""Performance tests for environment stepping throughput.

Measures vmapped step_env throughput at various batch sizes and asserts
minimum fps thresholds to catch performance regressions.
"""

from __future__ import annotations

import time

import jax
import pytest

from factoriax.envs.factoriax_env import FactoriaXEnv
from factoriax.state import EnvParams

from .conftest import NUM_STEPS, make_batched_envs

BATCH_SIZES = [1, 8, 64, 512, 4096]

# Minimum fps thresholds per batch size. These are conservative floors
# that should pass on any modern GPU. Actual throughput is typically
# 2-5x higher.
MIN_FPS = {
    1: 50,
    8: 400,
    64: 3000,
    512: 10000,
    4096: 20000,
}


@pytest.mark.parametrize("n_envs", BATCH_SIZES)
def test_step_throughput(
    n_envs: int, perf_params: EnvParams, perf_log_dir: object
) -> None:
    """Vmapped step throughput must exceed minimum fps.

    Args:
        n_envs: Batch size.
        perf_params: Shared environment parameters.
        perf_log_dir: Log directory (unused here, available for extensions).
    """
    env = FactoriaXEnv()
    _, states = make_batched_envs(n_envs, perf_params)
    vmap_step = jax.jit(
        jax.vmap(env.step_env, in_axes=(0, 0, 0, None))
    )

    # Warmup.
    rng = jax.random.key(77)
    rng, k_a, k_s = jax.random.split(rng, 3)
    actions = jax.random.randint(k_a, (n_envs,), 0, perf_params.NUM_ACTIONS)
    step_keys = jax.random.split(k_s, n_envs)
    _, states, _, _, _ = vmap_step(step_keys, states, actions, perf_params)
    jax.block_until_ready(jax.tree.leaves(states))

    # Timed run.
    t0 = time.perf_counter()
    for _ in range(NUM_STEPS):
        rng, k_a, k_s = jax.random.split(rng, 3)
        actions = jax.random.randint(
            k_a, (n_envs,), 0, perf_params.NUM_ACTIONS
        )
        step_keys = jax.random.split(k_s, n_envs)
        _, states, _, _, _ = vmap_step(
            step_keys, states, actions, perf_params
        )
    jax.block_until_ready(jax.tree.leaves(states))
    elapsed = time.perf_counter() - t0

    fps = n_envs * NUM_STEPS / elapsed
    threshold = MIN_FPS.get(n_envs, 50)
    assert fps >= threshold, (
        f"Step throughput {fps:.0f} fps < {threshold} fps at batch={n_envs}"
    )
