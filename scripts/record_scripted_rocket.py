"""Record a complete scripted-rocket-agent run as a debugger trajectory.

Drives :func:`baselines.rocket.scripted.agent.make_scripted_rocket_agent`
through the rocket benchmark level and saves the full per-tick
``EnvState`` history to a ``.npz`` trajectory file that the agent
debugger can replay.

Usage::

    uv run python scripts/record_scripted_rocket.py
    uv run python -m factoriax.agentdebugger /tmp/scripted_rocket.npz

The agent runs against the unwrapped ``FactoriaXEnv`` so the saved
states are plain ``EnvState`` (no ``AchievementState`` wrapping,
which ``trajectory_to_states`` wouldn't understand). Blocked actions
from :data:`ROCKET_BLOCKED_ACTIONS` are substituted with ``NOOP`` in
software — matching the behaviour of the training-time mask wrapper.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from baselines.rocket.scripted.agent import make_scripted_rocket_agent
from baselines.rocket.scripted.agent_advanced_factory import (
    make_advanced_factory_rocket_agent,
)
from factoriax.analysis.trajectory import states_to_trajectory
from factoriax.benchmarks.rocket import (
    NUM_ROCKET_ACHIEVEMENTS,
    ROCKET_ACHIEVEMENT_INFO,
    ROCKET_ACHIEVEMENT_WEIGHTS,
    ROCKET_BLOCKED_ACTIONS,
    build_rocket_level,
    rocket_conditions,
)
from factoriax.constants import Action
from factoriax.envs import FactoriaXEnv
from factoriax.levels import build_state
from factoriax.observations import global_array
from factoriax.state import EnvParams

logger = logging.getLogger(__name__)


_AGENTS: dict[str, object] = {
    "naive": make_scripted_rocket_agent,
    "advanced_factory": make_advanced_factory_rocket_agent,
}


def record(
    out_path: Path,
    max_steps: int,
    seed: int,
    agent_kind: str = "naive",
) -> dict[str, object]:
    """Run the scripted agent and save its trajectory.

    Args:
        out_path: Destination ``.npz`` path.
        max_steps: Episode budget; the run stops early when the agent
            reports ``is_done`` or the env signals ``done``.
        seed: JAX PRNG seed.

    Returns:
        Summary dict (``steps``, ``unlocked``, ``duration_sec``).
    """
    env_params = EnvParams(
        map_width=32,
        map_height=32,
        num_players=1,
        max_timesteps=max_steps,
    )
    level = build_rocket_level()
    state = build_state(level, env_params)

    env = FactoriaXEnv()
    jit_step = jax.jit(env.step_env)
    # Without JIT, global_array pays ~50 ms/tick on the 10-channel obs.
    jit_obs = jax.jit(lambda s: global_array(s, env_params, 0))
    if agent_kind not in _AGENTS:
        raise ValueError(
            f"unknown agent {agent_kind!r}; choose from {list(_AGENTS)}",
        )
    agent = _AGENTS[agent_kind](env_params)

    # The unwrapped env emits 0 reward every step — the rocket reward
    # lives on :class:`AchievementWrapper`, which we can't use here
    # without losing raw EnvState. Recompute it locally: latch the
    # achievement mask across ticks and take the weighted sum of newly
    # unlocked slots each step.
    weights_np = np.asarray(ROCKET_ACHIEVEMENT_WEIGHTS)[:NUM_ROCKET_ACHIEVEMENTS]
    unlocked_latched = np.zeros(NUM_ROCKET_ACHIEVEMENTS, dtype=bool)

    states: list = [state]
    actions: list[int] = []
    rewards: list[float] = []
    unlock_step = np.full((NUM_ROCKET_ACHIEVEMENTS,), -1, dtype=np.int32)

    key = jax.random.PRNGKey(seed)
    t0 = time.perf_counter()
    for t in range(max_steps):
        obs = np.asarray(jit_obs(state))
        raw = int(agent.act(obs))
        # Replicate the ActionMaskWrapper: masked actions become NOOP.
        action = int(Action.NOOP) if raw in ROCKET_BLOCKED_ACTIONS else raw

        key, subkey = jax.random.split(key)
        _, state, _, done, _ = jit_step(
            subkey,
            state,
            jnp.int32(action),
            env_params,
        )

        # Evaluate the achievement conditions on the unwrapped state,
        # latch them, and emit the weighted step reward.
        cond = np.asarray(rocket_conditions(state)[:NUM_ROCKET_ACHIEVEMENTS]).astype(
            bool
        )
        newly = cond & ~unlocked_latched
        step_reward = float(np.sum(weights_np * newly))
        unlocked_latched |= newly
        fresh = (unlock_step < 0) & newly
        unlock_step[fresh] = t

        states.append(state)
        actions.append(action)
        rewards.append(step_reward)

        if agent.is_done or bool(done):
            break

    elapsed = time.perf_counter() - t0

    traj = states_to_trajectory(
        states,
        actions=np.asarray(actions, dtype=np.int32),
        rewards=np.asarray(rewards, dtype=np.float32),
    )
    traj.save(str(out_path))

    unlocked = [
        ROCKET_ACHIEVEMENT_INFO[i].id
        for i in range(NUM_ROCKET_ACHIEVEMENTS)
        if unlock_step[i] >= 0
    ]
    return {
        "steps": len(actions),
        "unlocked": unlocked,
        "unlock_step": unlock_step.tolist(),
        "duration_sec": elapsed,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Record a scripted rocket-agent trajectory.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/scripted_rocket.npz"),
        help="Destination .npz file (default: /tmp/scripted_rocket.npz).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=8000,
        help="Episode budget (default 8000, matches the integration test).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--agent",
        choices=sorted(_AGENTS.keys()),
        default="naive",
        help="Which scripted agent to record (naive or factory).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("Recording %s agent -> %s", args.agent, args.out)

    summary = record(args.out, args.max_steps, args.seed, args.agent)

    logger.info(
        "Recorded %d steps in %.1fs (%d/%d achievements unlocked).",
        summary["steps"],
        summary["duration_sec"],
        len(summary["unlocked"]),
        NUM_ROCKET_ACHIEVEMENTS,
    )
    if summary["unlocked"]:
        logger.info("Unlocked: %s", ", ".join(summary["unlocked"]))
    logger.info(
        "Replay with: uv run python -m factoriax.agentdebugger %s",
        args.out,
    )


if __name__ == "__main__":
    main()
