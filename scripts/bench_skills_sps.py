"""Diagnose why ``baselines.skills.train_ppo`` reports ~13-16k SPS.

Compares pure step throughput (random policy, JIT-compiled scan) for:
 1. Raw ``FactoriaXEnv`` on the navigate skills level (5x5).
 2. ``SkillsRewardEnv`` on the same level (the wrapper trainer uses).
 3. ``ActionMaskWrapper`` over ``SkillsRewardEnv`` (full trainer stack).
 4. Raw ``FactoriaXEnv`` on the rocket level (32x32) for cross-reference.

Reports first-pass (with JIT compile) and steady-state (post-compile)
SPS. The training-loop SPS the trainer prints is computed from
``current_step / wall_clock`` cumulatively, so the gap between
first-pass and steady-state is the JIT-amortization effect that
dominates short training runs.

Run when the GPU is free.

Usage::

    uv run python scripts/bench_skills_sps.py
    uv run python scripts/bench_skills_sps.py --num-envs 128 --rollout-steps 128
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from factoriax.benchmarks.rocket import build_rocket_level, rocket_conditions
from factoriax.benchmarks.skills import SkillsBenchmark, skills_conditions
from factoriax.envs import FactoriaXEnv
from factoriax.envs.action_mask_wrapper import ActionMaskWrapper
from factoriax.levels import build_state
from factoriax.state import EnvParams

logger = logging.getLogger("bench_skills_sps")


@dataclass
class Sample:
    label: str
    num_envs: int
    rollout_steps: int
    compile_secs: float
    steady_secs: float

    @property
    def first_pass_sps(self) -> float:
        return self.num_envs * self.rollout_steps / self.compile_secs

    @property
    def steady_sps(self) -> float:
        return self.num_envs * self.rollout_steps / self.steady_secs


def _time_env(
    label: str,
    env: Any,
    state0: Any,
    env_params: EnvParams,
    num_envs: int,
    rollout_steps: int,
    seed: int,
) -> Sample:
    """Time first call (compile + run) and a second call (steady-state)."""
    vmap_step = jax.vmap(env.step_env, in_axes=(0, 0, 0, None))

    def _broadcast(x: jax.Array) -> jax.Array:
        a = jnp.asarray(x)
        return jnp.broadcast_to(a[None], (num_envs,) + a.shape)

    states0 = jax.tree.map(_broadcast, state0)
    rng0 = jax.random.PRNGKey(seed)

    @jax.jit
    def rollout(states: Any, rng: jax.Array) -> Any:
        def step(
            carry: tuple[Any, jax.Array], _: None
        ) -> tuple[tuple[Any, jax.Array], None]:
            st, rn = carry
            rn, act_key, step_key = jax.random.split(rn, 3)
            actions = jax.random.randint(act_key, (num_envs,), 0, 20)
            _, next_st, _, _, _ = vmap_step(
                jax.random.split(step_key, num_envs),
                st,
                actions,
                env_params,
            )
            return (next_st, rn), None

        (final_st, _), _ = jax.lax.scan(step, (states, rng), None, length=rollout_steps)
        return final_st

    # First pass — includes JIT compile.
    t0 = time.perf_counter()
    out = rollout(states0, rng0)
    jax.block_until_ready(out)
    compile_secs = time.perf_counter() - t0

    # Steady-state pass — already compiled.
    t0 = time.perf_counter()
    out = rollout(states0, rng0)
    jax.block_until_ready(out)
    steady_secs = time.perf_counter() - t0

    return Sample(
        label=label,
        num_envs=num_envs,
        rollout_steps=rollout_steps,
        compile_secs=compile_secs,
        steady_secs=steady_secs,
    )


def _build_skills_env_stack(
    target_bit: int,
    blocked: tuple[int, ...],
    env_params: EnvParams,
) -> Any:
    """Reproduce the trainer's wrapping: SkillsRewardEnv → ActionMaskWrapper.

    Imports the trainer's wrapper class directly so we measure the
    same code path. Only effect of ``target_bit`` here is that early
    termination becomes possible — for a random policy in 128 steps
    the probability is tiny so it doesn't bias the steady-state time.
    """
    from baselines.skills.train_ppo import SkillsRewardEnv  # noqa: PLC0415

    inner = SkillsRewardEnv(target_bit=target_bit)
    if not blocked:
        return inner
    return ActionMaskWrapper(inner, blocked)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    bench = SkillsBenchmark()
    levels = bench.levels()
    nav = next(b for b in levels if b.name == "navigate")
    nav_idx = levels.index(nav)

    # ---- skills: raw FactoriaXEnv on navigate ----
    skills_state0 = build_state(nav.level, nav.env_params)
    raw_skills = FactoriaXEnv(achievement_fn=skills_conditions)
    raw_sample = _time_env(
        "skills/raw",
        raw_skills,
        skills_state0,
        nav.env_params,
        args.num_envs,
        args.rollout_steps,
        args.seed,
    )

    # ---- skills: SkillsRewardEnv (trainer wrapper, no mask) ----
    rew_env = _build_skills_env_stack(
        target_bit=nav_idx, blocked=(), env_params=nav.env_params
    )
    rew_sample = _time_env(
        "skills/reward_env",
        rew_env,
        skills_state0,
        nav.env_params,
        args.num_envs,
        args.rollout_steps,
        args.seed,
    )

    # ---- skills: full trainer stack (mask + reward) ----
    blocked = tuple(nav.blocked_actions or ())
    full_env = _build_skills_env_stack(
        target_bit=nav_idx, blocked=blocked, env_params=nav.env_params
    )
    full_sample = _time_env(
        "skills/full_stack",
        full_env,
        skills_state0,
        nav.env_params,
        args.num_envs,
        args.rollout_steps,
        args.seed,
    )

    # ---- rocket: raw FactoriaXEnv on 32x32 ----
    rocket_level = build_rocket_level()
    rocket_params = EnvParams(
        map_width=32,
        map_height=32,
        num_players=1,
        max_timesteps=args.rollout_steps,
    )
    rocket_state0 = build_state(rocket_level, rocket_params)
    raw_rocket = FactoriaXEnv(achievement_fn=rocket_conditions)
    rocket_sample = _time_env(
        "rocket/raw",
        raw_rocket,
        rocket_state0,
        rocket_params,
        args.num_envs,
        args.rollout_steps,
        args.seed,
    )

    samples = [raw_sample, rew_sample, full_sample, rocket_sample]

    print(f"\nnum_envs={args.num_envs}, rollout_steps={args.rollout_steps}")
    print("=" * 88)
    print(
        f"{'label':<24}{'compile_s':>11}{'steady_s':>11}{'first_sps':>14}{'steady_sps':>14}{'speedup':>13}"
    )
    print("-" * 88)
    for s in samples:
        speedup = s.compile_secs / s.steady_secs if s.steady_secs > 0 else 0.0
        print(
            f"{s.label:<24}"
            f"{s.compile_secs:>11.3f}"
            f"{s.steady_secs:>11.3f}"
            f"{s.first_pass_sps:>14,.0f}"
            f"{s.steady_sps:>14,.0f}"
            f"{speedup:>12.1f}x"
        )
    print("=" * 88)


if __name__ == "__main__":
    main()
