"""Raw vs. ActionMaskWrapper'd env throughput A/B.

Measures whether wrapping :class:`FactoriaXEnv` in
:class:`ActionMaskWrapper` adds measurable per-step cost. The wrapper
is a thin closure: a single ``mask[action]`` index + a ``where`` to
substitute ``NOOP``, all on the host of ``step_env``. Once JIT'd, the
expectation is "negligible overhead" — this script verifies it.

Run this at the F.1/F.2 checkpoint to confirm the per-level mask path
introduced in :class:`BenchmarkRunner` doesn't regress training-time
throughput. Mirrors ``scripts/bench_rocket_wrapper.py``'s pattern:
random policy, vmap-over-envs, lax.scan-over-time, warmup once before
timing.

Usage::

    uv run python scripts/bench_actionmask_overhead.py
    uv run python scripts/bench_actionmask_overhead.py \
        --num-envs 128 --rollout-steps 2000

Run when the GPU is free (no training in progress).
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from factoriax.constants import Action, BlockType
from factoriax.envs import FactoriaXEnv
from factoriax.envs.action_mask_wrapper import ActionMaskWrapper
from factoriax.levels import LevelBuilder, build_state
from factoriax.state import EnvParams

logger = logging.getLogger("bench_actionmask_overhead")


@dataclass
class BenchResult:
    """Outcome of one timed rollout."""

    label: str
    total_steps: int
    seconds: float

    @property
    def steps_per_sec(self) -> float:
        return self.total_steps / self.seconds if self.seconds > 0 else float("inf")


def _random_policy(key: jax.Array, num_actions: int = 79) -> jax.Array:
    """Sample a uniform random action across the full action space."""
    return jax.random.randint(key, (), 0, num_actions)


def _time_rollout(
    label: str,
    env,
    state0,
    params: EnvParams,
    num_envs: int,
    rollout_steps: int,
    seed: int,
) -> BenchResult:
    """JIT a vmapped scan of *rollout_steps* ticks and return wall-clock time."""
    vmap_step = jax.vmap(env.step_env, in_axes=(0, 0, 0, None))
    vmap_policy = jax.vmap(_random_policy)

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

    # Warm up JIT — first call traces + compiles.
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
) -> tuple[BenchResult, BenchResult, BenchResult]:
    """Run raw / wrapped-noop / wrapped-rewrite rollouts back-to-back.

    The third configuration uses a mask that *actually* hits the
    ``where`` rewrite path (blocks ``RIGHT``, which the random policy
    selects ~1/79 of the time). Useful to separate "always-NOOP cost"
    from "real-rewrite cost" — though once JIT'd they should be the
    same since both branches of the ``where`` execute regardless.

    Returns:
        ``(raw, wrapped_inert_mask, wrapped_active_mask)``.
    """
    map_size = 16
    level = (
        LevelBuilder(map_size, map_size)
        .fill_rect(2, 2, 4, 4, BlockType.COAL)
        .set_player_position(0, 0)
        .build("bench_actionmask")
    )
    params = EnvParams(
        map_width=map_size,
        map_height=map_size,
        num_players=1,
        max_timesteps=rollout_steps,
    )
    env_state0 = build_state(level, params)

    raw_env = FactoriaXEnv()
    raw_result = _time_rollout(
        "raw", raw_env, env_state0, params, num_envs, rollout_steps, seed
    )

    inert_env = ActionMaskWrapper(raw_env, (int(Action.NOOP),))
    inert_result = _time_rollout(
        "wrapped(inert)",
        inert_env,
        env_state0,
        params,
        num_envs,
        rollout_steps,
        seed,
    )

    active_env = ActionMaskWrapper(raw_env, (int(Action.RIGHT),))
    active_result = _time_rollout(
        "wrapped(active)",
        active_env,
        env_state0,
        params,
        num_envs,
        rollout_steps,
        seed,
    )

    return raw_result, inert_result, active_result


def _report(raw: BenchResult, inert: BenchResult, active: BenchResult) -> None:
    """Print a three-row table comparing raw vs. wrapped throughputs."""

    def _overhead_pct(other: BenchResult) -> float:
        return (other.seconds / raw.seconds - 1.0) * 100.0 if raw.seconds > 0 else 0.0

    print(f"\n{'=' * 78}")
    print(f"{'Config':<18}{'Steps':>14}{'Seconds':>12}{'Steps/sec':>20}")
    print(f"{'-' * 18}{'-' * 14}{'-' * 12}{'-' * 20}")
    for r in (raw, inert, active):
        print(
            f"{r.label:<18}{r.total_steps:>14,}{r.seconds:>12.3f}"
            f"{r.steps_per_sec:>20,.0f}"
        )
    print(f"{'=' * 78}")
    print(f"wrapped(inert)  overhead: {_overhead_pct(inert):+.1f}% wall-clock")
    print(f"wrapped(active) overhead: {_overhead_pct(active):+.1f}% wall-clock")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--rollout-steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logger.info(
        "Benchmarking raw vs. ActionMaskWrapper: num_envs=%d, rollout_steps=%d",
        args.num_envs,
        args.rollout_steps,
    )
    raw, inert, active = bench(args.num_envs, args.rollout_steps, args.seed)
    _report(raw, inert, active)


if __name__ == "__main__":
    main()
