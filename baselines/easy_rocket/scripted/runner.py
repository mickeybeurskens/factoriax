"""Drive the scripted agent through one easy_rocket episode.

Builds the easy_rocket env, constructs the
:class:`~baselines.easy_rocket.scripted.agent.ScriptedAgent` from the
initial state, runs it tick-by-tick to ``done`` or
``max_timesteps``, then writes the rollout to mp4 plus a per-phase
status report.

Usage::

    uv run python -m baselines.easy_rocket.scripted.runner
    uv run python -m baselines.easy_rocket.scripted.runner --seed 7
    uv run python -m baselines.easy_rocket.scripted.runner \\
        --out-dir /tmp/scripted --no-save-video
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable
from pathlib import Path

import jax
import numpy as np

import factoriax
from factoriax.analysis.video import compose_frame_with_inventory, write_video
from factoriax.engine.constants import NUM_ACTIONS, Action
from factoriax.engine.levels import build_state
from factoriax.engine.state import EnvParams, EnvState
from factoriax.scenarios.easy_rocket import (
    NUM_EASY_ROCKET_ACHIEVEMENTS,
    EasyRocketScenario,
    easy_rocket_conditions,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger("easy_rocket_scripted")

#: Signature every scripted policy implements. State-reader: takes the
#: full ``EnvState`` and returns the chosen action as a JAX scalar.
ScriptedPolicy = Callable[[EnvState, EnvParams], jax.Array]


def _make_env_and_state(seed: int) -> tuple[object, EnvState, EnvParams]:
    """Build the easy_rocket env, initial state, and params.

    Level and params both come from :class:`EasyRocketScenario` — the
    single source of truth for the scenario's runtime config (recipe
    table, episode budget, entity budget). The runner applies no
    overrides; to change a setting, change the scenario. ``make`` builds
    only the env/wrapper stack; its generic default params are discarded
    in favour of the scenario's.
    """
    scenario_level = EasyRocketScenario(seed=seed).levels()[0]
    level, env_params = scenario_level.level, scenario_level.env_params
    env, _ = factoriax.make(
        level,
        obs="global",
        achievement_fn=easy_rocket_conditions,
    )
    state0 = build_state(level, env_params)
    return env, state0, env_params


def run_policy(
    policy: ScriptedPolicy,
    seed: int = 0,
    debug_every: int = 0,
    agent_for_debug: object | None = None,
) -> tuple[list[EnvState], list[int]]:
    """Drive *policy* through one easy_rocket episode.

    Steps the JIT'd env, calling *policy* with the current state at
    each tick. Stops at ``done`` or the scenario's episode budget
    (``env_params.max_timesteps``). Returns the full state and action
    trajectories for downstream rendering.

    When ``debug_every > 0`` and ``agent_for_debug`` is a
    :class:`ScriptedAgent`, the runner logs the agent's per-phase
    debug status every ``debug_every`` ticks plus once on every
    phase transition.
    """
    env, state, env_params = _make_env_and_state(seed)
    jit_step = jax.jit(env.step_env)
    rng = jax.random.PRNGKey(seed)

    states: list[EnvState] = [state]
    actions: list[int] = []
    last_phase = -1

    for tick in range(env_params.max_timesteps):
        action = policy(state, env_params)
        if agent_for_debug is not None and debug_every > 0:
            cur_phase = getattr(agent_for_debug, "current_phase", -1)
            phase_changed = cur_phase != last_phase
            if phase_changed or (tick > 0 and tick % debug_every == 0):
                phases = getattr(agent_for_debug, "phases", [])
                if 0 <= cur_phase < len(phases):
                    ph = phases[cur_phase]
                    status = ph.debug_status(state)
                    logger.info(
                        "  t=%4d a=%-22s %s",
                        tick,
                        Action(int(action)).name,
                        status,
                    )
                last_phase = cur_phase
        rng, subkey = jax.random.split(rng)
        _obs, state, _r, done, _info = jit_step(subkey, state, action, env_params)
        actions.append(int(action))
        states.append(state)
        if bool(done):
            break

    return states, actions


def _save_video(states: list[EnvState], out_path: Path, fps: int) -> None:
    """Compose per-tick map+inventory frames and write to mp4."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames = [compose_frame_with_inventory(s) for s in states]
    write_video(out_path, frames, fps)
    logger.info("Saved video: %s (%d frames)", out_path, len(frames))


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="runs/easy_rocket_scripted",
        help="Directory for the rollout video.",
    )
    parser.add_argument("--no-save-video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument(
        "--debug-every",
        type=int,
        default=0,
        help="Log agent status every N ticks (0 = off). Also logs on phase change.",
    )
    args = parser.parse_args()

    from baselines.easy_rocket.scripted.agent import (  # noqa: PLC0415
        ScriptedAgent,
    )

    _, init_state, env_params = _make_env_and_state(args.seed)
    agent = ScriptedAgent(init_state, env_params.recipe_table)

    t0 = time.perf_counter()
    states, actions = run_policy(
        agent,
        seed=args.seed,
        debug_every=args.debug_every,
        agent_for_debug=agent,
    )
    elapsed = time.perf_counter() - t0

    final_state = states[-1]
    ach_mask = np.asarray(final_state.achievements_unlocked)[
        :NUM_EASY_ROCKET_ACHIEVEMENTS
    ]
    ach_count = int(ach_mask.sum())
    action_counts = np.bincount(np.asarray(actions), minlength=NUM_ACTIONS)
    top1 = int(np.argmax(action_counts))
    logger.info(
        "Done. ticks=%d  ach=%d/%d  top_action=%s (%.0f%%)  rollout=%.2fs",
        len(actions),
        ach_count,
        NUM_EASY_ROCKET_ACHIEVEMENTS,
        Action(top1).name,
        100.0 * action_counts[top1] / max(1, len(actions)),
        elapsed,
    )
    logger.info("Phase report:")
    for name, ok, ticks in agent.report.phase_outcomes:
        logger.info(
            "  %-40s %s  ticks=%d",
            name,
            "OK" if ok else "FAIL",
            ticks,
        )
    if agent.report.halted:
        logger.info("HALTED: %s", agent.report.halt_reason)

    if not args.no_save_video:
        _save_video(states, Path(args.out_dir) / "rollout.mp4", args.video_fps)


if __name__ == "__main__":
    main()
