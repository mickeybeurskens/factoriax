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

from factoriax.analysis.video import compose_frame_with_inventory, write_video
from factoriax.engine.constants import NUM_ACTIONS, Action
from factoriax.make import env_from_name
from factoriax.engine.envs.easy_rocket import (
    EASY_ROCKET_ACHIEVEMENT_NAMES,
    NUM_EASY_ROCKET_ACHIEVEMENTS,
)
from factoriax.engine.state import EnvParams, EnvState

_SCENARIO_ID = "EasyRocket-v1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger("easy_rocket_scripted")

#: Signature every scripted policy implements. State-reader: takes the
#: full ``EnvState`` and returns the chosen action as a JAX scalar.
ScriptedPolicy = Callable[[EnvState, EnvParams], jax.Array]


def _make_env_and_state(seed: int) -> tuple[object, EnvState, EnvParams]:
    """Build the easy_rocket env, an initial state, and params.

    Loads the scenario via ``factoriax.make("EasyRocket-v1")`` — the registry is
    the single source of truth for the env (keyed procgen reset, achievement
    hook, reward) and its params. The initial state is drawn from the scenario's
    keyed generator with ``PRNGKey(seed)``, so each seed gives a distinct (but
    reproducible) layout the scripted agent plans around.
    """
    env, env_params = env_from_name(_SCENARIO_ID)
    _, state0 = env.reset_env(jax.random.PRNGKey(seed), env_params)
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


def _first_unlock_steps(states: list[EnvState], num_ach: int) -> np.ndarray:
    """Per-bit first-unlock step over the rollout (-1 if never unlocked).

    ``states[t]`` is the state after ``t`` env steps, so the index of the
    first ``True`` in a bit's column is the env-step at which it fired —
    the scripted analogue of the PPO trainer's per-bit
    ``first_unlock_step``.
    """
    masks = np.stack(
        [np.asarray(s.achievements_unlocked)[:num_ach] for s in states]
    )  # (T+1, num_ach) bool
    fu = np.full(num_ach, -1, dtype=int)
    for bit in range(num_ach):
        hits = np.flatnonzero(masks[:, bit])
        if hits.size:
            fu[bit] = int(hits[0])
    return fu


def _save_video(states: list[EnvState], out_path: Path, fps: int) -> None:
    """Compose per-tick map+inventory frames and write to mp4."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames = [compose_frame_with_inventory(s) for s in states]
    write_video(out_path, frames, fps)
    logger.info("Saved video: %s (%d frames)", out_path, len(frames))


def _log_wandb(
    *,
    project: str,
    run_name: str | None,
    seed: int,
    max_timesteps: int,
    ach_mask: np.ndarray,
    first_unlock_step: np.ndarray,
    ticks: int,
    elapsed: float,
    phase_outcomes: list[tuple[str, bool, int]],
    video_path: Path,
    video_fps: int,
) -> None:
    """Log the scripted rollout to wandb: per-achievement results + video.

    Each achievement bit is logged as a 0/1 scalar under ``achievements/``
    and collected into a table for a readable per-bit view. The per-bit
    ``bit/<name>/recent_unlock_rate`` (0/1 over this single episode) and
    ``bit/<name>/first_unlock_step`` mirror the PPO trainer's keys so the
    paper-side empirical-order analysis reads scripted and PPO runs the
    same way. The mp4 rollout is logged inline. Raises ``ImportError``
    (caught by the caller) if wandb is not installed.
    """
    import wandb  # noqa: PLC0415  # type: ignore[import-untyped]

    run = wandb.init(
        project=project,
        name=run_name or f"scripted_easy_rocket_seed{seed}",
        config={"seed": seed, "max_timesteps": max_timesteps, "policy": "scripted"},
        tags=["easy_rocket", "scripted", "achievement", "validation"],
    )
    unlocked = ach_mask.astype(bool)
    table = wandb.Table(columns=["index", "name", "unlocked", "first_unlock_step"])
    log: dict[str, object] = {}
    for idx, name in enumerate(EASY_ROCKET_ACHIEVEMENT_NAMES):
        hit = bool(unlocked[idx])
        fu = int(first_unlock_step[idx])
        log[f"achievements/{name}"] = int(hit)
        log[f"bit/{name}/recent_unlock_rate"] = float(hit)
        if fu >= 0:
            log[f"bit/{name}/first_unlock_step"] = float(fu)
        table.add_data(idx, name, hit, fu)
    phase_table = wandb.Table(columns=["phase", "ok", "ticks"])
    for name, ok, pticks in phase_outcomes:
        phase_table.add_data(name, bool(ok), pticks)
    log.update(
        {
            "achievements/unlocked_count": int(unlocked.sum()),
            "achievements/total": len(EASY_ROCKET_ACHIEVEMENT_NAMES),
            "achievements/fraction": float(unlocked.mean()),
            "achievements/table": table,
            "rollout/ticks": ticks,
            "rollout/seconds": elapsed,
            "phases/outcomes": phase_table,
            "final/rollout": wandb.Video(str(video_path), fps=video_fps, format="mp4"),
        }
    )
    run.log(log)
    run.finish()
    logger.info("Logged scripted rollout to wandb project %r.", project)


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
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log per-achievement results and the rollout video to wandb.",
    )
    parser.add_argument("--wandb-project", type=str, default="factoriax_easy_rocket")
    parser.add_argument("--wandb-run-name", type=str, default=None)
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
    first_unlock = _first_unlock_steps(states, NUM_EASY_ROCKET_ACHIEVEMENTS)
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
    logger.info("Achievements:")
    for idx, name in enumerate(EASY_ROCKET_ACHIEVEMENT_NAMES):
        fu = int(first_unlock[idx])
        logger.info("  %-20s %s", name, f"OK @ step {fu}" if fu >= 0 else "--")
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

    video_path = Path(args.out_dir) / "rollout.mp4"
    need_video = not args.no_save_video or args.wandb
    if need_video:
        _save_video(states, video_path, args.video_fps)

    if args.wandb:
        try:
            _log_wandb(
                project=args.wandb_project,
                run_name=args.wandb_run_name,
                seed=args.seed,
                max_timesteps=env_params.max_timesteps,
                ach_mask=ach_mask,
                first_unlock_step=first_unlock,
                ticks=len(actions),
                elapsed=elapsed,
                phase_outcomes=agent.report.phase_outcomes,
                video_path=video_path,
                video_fps=args.video_fps,
            )
        except ImportError:
            logger.error("wandb not installed. Run: uv add wandb")


if __name__ == "__main__":
    main()
