"""Run both scripted rocket agents and compare their episode timings.

Both agents share the same rocket benchmark level (pre-placed
furnace + assembler, five ore patches). The difference is the plan:

- ``naive``   — :mod:`baselines.rocket.scripted.agent` runs every
  recipe serially through the pre-placed machines.
- ``factory`` — :mod:`baselines.rocket.scripted.agent_factory`
  spends a starter phase to place two extra furnaces and two extra
  assemblers, then uses :class:`PipelinedProduce` to rotate bulk
  work across the 3-machine batteries.

Both agents are deterministic, so the comparison is reproducible
from a single seed.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from baselines.rocket.scripted.agent import make_scripted_rocket_agent
from baselines.rocket.scripted.agent_factory import make_factory_rocket_agent
from factoriax.benchmarks.rocket import (
    NUM_ROCKET_ACHIEVEMENTS,
    ROCKET_ACHIEVEMENT_INFO,
    ROCKET_ACHIEVEMENT_WEIGHTS,
    ROCKET_BLOCKED_ACTIONS,
    build_rocket_level,
    rocket_conditions,
)
from factoriax.constants import MAX_ACHIEVEMENTS, Action
from factoriax.envs import FactoriaXEnv
from factoriax.envs.achievement_wrapper import AchievementState, AchievementWrapper
from factoriax.envs.action_mask_wrapper import ActionMaskWrapper
from factoriax.levels import build_state
from factoriax.observations import global_array
from factoriax.state import EnvParams

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Per-agent episode summary."""

    label: str
    steps: int
    wall_sec: float
    score: float
    unlocked: int
    unlock_step: np.ndarray  # shape (NUM_ROCKET_ACHIEVEMENTS,), int32


def _run(
    label: str,
    agent_factory,
    max_steps: int,
    seed: int,
) -> RunResult:
    """Drive one agent through the rocket benchmark env."""
    env_params = EnvParams(
        map_width=32,
        map_height=32,
        num_players=1,
        max_timesteps=max_steps,
    )
    level = build_rocket_level()
    env_state = build_state(level, env_params)
    state = AchievementState(
        env_state=env_state,
        achievements_unlocked=jnp.zeros(MAX_ACHIEVEMENTS, dtype=jnp.bool_),
    )
    env = ActionMaskWrapper(
        AchievementWrapper(FactoriaXEnv(), rocket_conditions),
        ROCKET_BLOCKED_ACTIONS,
    )
    jit_step = jax.jit(env.step_env)
    # JIT the obs too — without this, every tick pays ~50ms for the
    # 10-channel scatter-based global_array.
    jit_obs = jax.jit(lambda s: global_array(s, env_params, 0))
    agent = agent_factory(env_params)

    unlock_step = np.full((NUM_ROCKET_ACHIEVEMENTS,), -1, dtype=np.int32)
    key = jax.random.PRNGKey(seed)
    t0 = time.perf_counter()

    for t in range(max_steps):
        obs = np.asarray(jit_obs(state.env_state))
        action = agent.act(obs)
        key, subkey = jax.random.split(key)
        _, state, _, done, _ = jit_step(
            subkey,
            state,
            jnp.int32(action),
            env_params,
        )
        mask = np.asarray(state.achievements_unlocked)[:NUM_ROCKET_ACHIEVEMENTS]
        newly = (unlock_step < 0) & mask
        unlock_step[newly] = t
        if agent.is_done or bool(done):
            break

    wall = time.perf_counter() - t0
    final_mask = np.asarray(state.achievements_unlocked)[:NUM_ROCKET_ACHIEVEMENTS]
    weights = np.asarray(ROCKET_ACHIEVEMENT_WEIGHTS)[:NUM_ROCKET_ACHIEVEMENTS]
    score = float(np.sum(weights * final_mask))

    return RunResult(
        label=label,
        steps=t + 1,
        wall_sec=wall,
        score=score,
        unlocked=int(final_mask.sum()),
        unlock_step=unlock_step,
    )


def _print_summary(results: list[RunResult]) -> None:
    """Print a side-by-side comparison of the runs."""
    print()
    print("=" * 72)
    header = f"{'':<24}" + "".join(f"{r.label:>16}" for r in results)
    print(header)
    print("-" * 72)
    rows = [
        ("Total ticks", [str(r.steps) for r in results]),
        ("Achievements", [f"{r.unlocked}/{NUM_ROCKET_ACHIEVEMENTS}" for r in results]),
        ("Score", [f"{r.score:.0f}" for r in results]),
        ("Wall clock", [f"{r.wall_sec:.1f}s" for r in results]),
    ]
    for name, vals in rows:
        print(f"{name:<24}" + "".join(f"{v:>16}" for v in vals))
    print("=" * 72)

    # Achievement-by-achievement unlock timing.
    print()
    print("Unlock ticks (earlier is better). ``-`` = not unlocked.")
    print(f"{'Achievement':<24}" + "".join(f"{r.label:>16}" for r in results))
    print("-" * 72)
    for i, info in enumerate(ROCKET_ACHIEVEMENT_INFO):
        cells = []
        for r in results:
            t = int(r.unlock_step[i])
            cells.append(f"{t:>16}" if t >= 0 else f"{'-':>16}")
        print(f"{info.id:<24}" + "".join(cells))


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compare the naive and factory rocket agents.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=16000,
        help="Episode budget (default matches the benchmark).",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    _ = Action  # symbol check

    print("Running NAIVE agent…")
    naive = _run("naive", make_scripted_rocket_agent, args.max_steps, args.seed)
    print(
        f"  → {naive.steps} ticks, "
        f"{naive.unlocked}/{NUM_ROCKET_ACHIEVEMENTS} unlocked.",
    )

    print("Running FACTORY agent…")
    factory = _run(
        "factory",
        make_factory_rocket_agent,
        args.max_steps,
        args.seed,
    )
    print(
        f"  → {factory.steps} ticks, "
        f"{factory.unlocked}/{NUM_ROCKET_ACHIEVEMENTS} unlocked.",
    )

    _print_summary([naive, factory])


if __name__ == "__main__":
    main()
