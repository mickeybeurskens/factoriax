"""Micro-benchmark for environment rollout throughput.

Measures steps/second for random-action rollouts on a fixed 15x15 map.
Run with: python tests/bench_rollout.py
"""

import time

import jax
import jax.numpy as jnp
from jax import random

from factoriax.engine.envs.base import FactoriaxEnv
from factoriax.engine.constants import NUM_ACTIONS
from factoriax.engine.state import EnvParams


def bench_single_env(num_steps: int = 2000) -> float:
    """Benchmark single-environment rollout.

    Args:
        num_steps: Number of steps to run.

    Returns:
        Steps per second.
    """
    env = FactoriaxEnv()
    params = EnvParams(num_players=1)
    rng = random.PRNGKey(0)

    rng, reset_key = random.split(rng)
    obs, state = env.reset_env(reset_key, params)

    step_fn = jax.jit(env.step_env)

    # Warmup: compile the step function.
    rng, warmup_key, action_key = random.split(rng, 3)
    action = random.randint(action_key, (), 0, NUM_ACTIONS)
    obs, state, reward, done, info = step_fn(warmup_key, state, action, params)
    obs.block_until_ready()

    # Timed rollout.
    rng, *step_keys = random.split(rng, num_steps + 1)
    step_keys = jnp.array([k for k in step_keys])
    actions = random.randint(random.PRNGKey(1), (num_steps,), 0, NUM_ACTIONS)

    # Force compilation to finish before timing.
    jax.block_until_ready(state)

    t0 = time.perf_counter()
    for i in range(num_steps):
        obs, state, reward, done, info = step_fn(
            step_keys[i], state, actions[i], params
        )
    jax.block_until_ready(state)
    t1 = time.perf_counter()

    return num_steps / (t1 - t0)


def bench_batched_env(num_envs: int = 64, num_steps: int = 500) -> float:
    """Benchmark vmapped multi-environment rollout.

    Args:
        num_envs: Number of parallel environments.
        num_steps: Number of steps per environment.

    Returns:
        Total steps per second (num_envs * num_steps / elapsed).
    """
    env = FactoriaxEnv()
    params = EnvParams(num_players=1)
    rng = random.PRNGKey(0)

    # Reset all envs.
    reset_keys = random.split(rng, num_envs)
    vmap_reset = jax.vmap(env.reset_env, in_axes=(0, None))
    obs_batch, state_batch = vmap_reset(reset_keys, params)

    vmap_step = jax.jit(jax.vmap(env.step_env, in_axes=(0, 0, 0, None)))

    # Warmup.
    rng, warmup_key = random.split(rng)
    warmup_keys = random.split(warmup_key, num_envs)
    warmup_actions = jnp.zeros(num_envs, dtype=jnp.int32)
    obs_batch, state_batch, _, _, _ = vmap_step(
        warmup_keys, state_batch, warmup_actions, params
    )
    jax.block_until_ready(state_batch)

    # Timed rollout.
    rng, action_rng = random.split(rng)
    all_actions = random.randint(action_rng, (num_steps, num_envs), 0, NUM_ACTIONS)
    rng, *step_rngs = random.split(rng, num_steps + 1)
    all_keys = jnp.array([random.split(k, num_envs) for k in step_rngs])

    jax.block_until_ready(state_batch)

    t0 = time.perf_counter()
    for i in range(num_steps):
        obs_batch, state_batch, _, _, _ = vmap_step(
            all_keys[i], state_batch, all_actions[i], params
        )
    jax.block_until_ready(state_batch)
    t1 = time.perf_counter()

    total_steps = num_envs * num_steps
    return total_steps / (t1 - t0)


if __name__ == "__main__":
    print("Benchmarking single-env rollout (2000 steps)...")
    single_sps = bench_single_env(2000)
    print(f"  {single_sps:.0f} steps/sec")

    for n_envs in [64, 256, 1024, 4096, 8192, 16384]:
        steps = max(20, 500 // max(1, n_envs // 64))
        print(f"Benchmarking batched rollout ({n_envs} envs x {steps} steps)...")
        try:
            batched_sps = bench_batched_env(n_envs, steps)
            print(f"  {batched_sps:.0f} steps/sec")
        except Exception as e:
            print(f"  FAILED: {e}")
            break
