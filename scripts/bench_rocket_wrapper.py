"""Raw vs. with-achievement_fn env throughput A/B for the rocket benchmark.

Measures the per-step cost of evaluating :func:`rocket_conditions` on
every tick by binding it as :class:`FactoriaXEnv`'s ``achievement_fn``
constructor argument. Runs an identical random-policy rollout through
both configurations (no fn vs fn), warms up JIT, then times a scan of
``rollout_steps`` ticks across ``num_envs`` parallel envs.

Usage::

    uv run python scripts/bench_rocket_wrapper.py
    uv run python scripts/bench_rocket_wrapper.py --num-envs 128 --rollout-steps 2000

Run this when the GPU is free (no training in progress).
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from factoriax.benchmarks.rocket import build_rocket_level, rocket_conditions
from factoriax.envs import FactoriaXEnv
from factoriax.levels import build_state
from factoriax.state import EnvParams

logger = logging.getLogger("bench_rocket_wrapper")


@dataclass
class BenchResult:
    """Outcome of one timed rollout."""

    label: str
    total_steps: int
    seconds: float

    @property
    def steps_per_sec(self) -> float:
        return self.total_steps / self.seconds if self.seconds > 0 else float("inf")


def _random_policy(key: jax.Array, num_actions: int = 20) -> jax.Array:
    """Sample a uniform random action."""
    return jax.random.randint(key, (), 0, num_actions)


def _time_rollout(
    label: str,
    env,
    state0,
    num_envs: int,
    rollout_steps: int,
    seed: int,
) -> BenchResult:
    """JIT a scan of *rollout_steps* ticks and return wall-clock time.

    Uses the same vmap-over-envs + lax.scan-over-time pattern as
    PureJaxRL training loops. The scan body calls the (possibly
    wrapped) ``env.step_env`` and a random policy.
    """
    vmap_step = jax.vmap(env.step_env, in_axes=(0, 0, 0, None))
    vmap_policy = jax.vmap(_random_policy)
    params = EnvParams(
        map_width=32,
        map_height=32,
        num_players=1,
        max_timesteps=rollout_steps,
    )

    # Broadcast initial state to (num_envs, ...).
    def _broadcast(x: jax.Array) -> jax.Array:
        a = jnp.asarray(x)
        return jnp.broadcast_to(a[None], (num_envs,) + a.shape)

    states0 = jax.tree.map(_broadcast, state0)
    rng0 = jax.random.PRNGKey(seed)

    @jax.jit
    def rollout(states, rng):
        def step(carry, _):
            st, rn = carry
            rn, act_key, step_key = jax.random.split(rn, 3)
            actions = vmap_policy(jax.random.split(act_key, num_envs))
            _, next_st, _, _, _ = vmap_step(
                jax.random.split(step_key, num_envs),
                st,
                actions,
                params,
            )
            return (next_st, rn), None

        (final_st, _), _ = jax.lax.scan(step, (states, rng), None, length=rollout_steps)
        return final_st

    # Warm up JIT.
    warmup = rollout(states0, rng0)
    jax.block_until_ready(warmup)

    start = time.perf_counter()
    final = rollout(states0, rng0)
    jax.block_until_ready(final)
    elapsed = time.perf_counter() - start

    total = num_envs * rollout_steps
    return BenchResult(label=label, total_steps=total, seconds=elapsed)


def bench(
    num_envs: int,
    rollout_steps: int,
    seed: int = 0,
) -> tuple[BenchResult, BenchResult]:
    """Run raw and wrapped rollouts back-to-back.

    Returns:
        ``(raw_result, wrapped_result)``.
    """
    level = build_rocket_level()
    params = EnvParams(
        map_width=32,
        map_height=32,
        num_players=1,
        max_timesteps=rollout_steps,
    )
    env_state0 = build_state(level, params)

    raw_env = FactoriaXEnv()
    raw_result = _time_rollout(
        "raw",
        raw_env,
        env_state0,
        num_envs,
        rollout_steps,
        seed,
    )

    wrapped_env = FactoriaXEnv(achievement_fn=rocket_conditions)
    wrapped_result = _time_rollout(
        "wrapped",
        wrapped_env,
        env_state0,
        num_envs,
        rollout_steps,
        seed,
    )

    return raw_result, wrapped_result


def _report(raw: BenchResult, wrapped: BenchResult) -> None:
    """Print a two-row table comparing raw vs. wrapped throughput."""
    overhead_ratio = wrapped.seconds / raw.seconds if raw.seconds > 0 else float("inf")
    overhead_per_step_us = (
        (wrapped.seconds - raw.seconds) / wrapped.total_steps * 1e6
        if wrapped.total_steps > 0
        else 0.0
    )
    print(f"\n{'=' * 72}")
    print(f"{'Config':<10}{'Steps':>14}{'Seconds':>12}{'Steps/sec':>18}")
    print(f"{'-' * 10}{'-' * 14}{'-' * 12}{'-' * 18}")
    for r in (raw, wrapped):
        print(
            f"{r.label:<10}{r.total_steps:>14,}{r.seconds:>12.3f}{r.steps_per_sec:>18,.0f}"
        )
    print(f"{'=' * 72}")
    print(f"Wrapped overhead: {(overhead_ratio - 1.0) * 100:+.1f}% wall-clock")
    print(f"Wrapped overhead per step: {overhead_per_step_us:+.3f} us")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--rollout-steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logger.info(
        "Benchmarking raw vs. wrapped: num_envs=%d, rollout_steps=%d",
        args.num_envs,
        args.rollout_steps,
    )
    raw, wrapped = bench(args.num_envs, args.rollout_steps, args.seed)
    _report(raw, wrapped)


if __name__ == "__main__":
    main()
