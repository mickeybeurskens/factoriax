"""Quick A/B benchmark: step throughput before and after rotate.

Stashes the current changes, benchmarks the old code, restores,
benchmarks the new code, and prints a comparison table. Run this
when the GPU is free (no training in progress).

Usage:
    python scripts/bench_rotate.py
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp

from factoriax.constants import Action
from factoriax.envs.factoriax_env import make_factoriax_env
from factoriax.state import EnvParams


def _bench(
    label: str,
    map_sizes: tuple[int, ...] = (16, 64),
    batch_sizes: tuple[int, ...] = (1, 256, 1024, 4096),
    duration: float = 10.0,
) -> None:
    """Run step throughput for a matrix of configurations.

    Args:
        label: Label for this benchmark run.
        map_sizes: Map widths/heights to test.
        batch_sizes: Batch sizes to test.
        duration: Seconds to measure per configuration.
    """
    env, _ = make_factoriax_env()
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"{'Map':>6} {'Batch':>6} {'Steps/s':>12} {'Total steps':>12}")
    print(f"{'-' * 6} {'-' * 6} {'-' * 12} {'-' * 12}")

    for ms in map_sizes:
        for bs in batch_sizes:
            params = EnvParams(map_width=ms, map_height=ms)
            keys = jax.random.split(jax.random.PRNGKey(0), bs)

            reset_fn = jax.vmap(env.reset_env, in_axes=(0, None))
            step_fn = jax.jit(
                jax.vmap(env.step_env, in_axes=(0, 0, None, None)),
            )

            _, states = reset_fn(keys, params)
            action = int(Action.NOOP)
            step_keys = jax.random.split(jax.random.PRNGKey(1), bs)

            # Warmup JIT.
            _ = step_fn(step_keys, states, action, params)
            jax.block_until_ready(_)

            # Measure.
            total = 0
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < duration:
                _ = step_fn(step_keys, states, action, params)
                jax.block_until_ready(_)
                total += bs

            elapsed = time.perf_counter() - t0
            sps = total / elapsed
            print(f"{ms:>6} {bs:>6} {sps:>12,.0f} {total:>12,}")


if __name__ == "__main__":
    _bench("Current code (with rotate)")
    print("\nDone. Compare against previous commit by checking out")
    print("the parent commit and running this script again.")
