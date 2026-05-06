"""End-to-end tests for the scripted rocket agent.

Runs the agent against the real ``RocketBenchmark`` env (with the
hand-craft action mask applied) and reports which achievements fire.
If a subset is missing the assertion message lists the exact unfired
ids and their unlock timesteps — the scripted agent doubles as the
reward-correctness oracle described in the design doc.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from baselines.rocket.scripted.agent import make_scripted_rocket_agent
from factoriax.benchmarks.rocket import (
    NUM_ROCKET_ACHIEVEMENTS,
    ROCKET_ACHIEVEMENT_INFO,
    ROCKET_BLOCKED_ACTIONS,
    build_rocket_level,
    rocket_conditions,
)
from factoriax.envs import FactoriaXEnv
from factoriax.envs.action_mask_wrapper import ActionMaskWrapper
from factoriax.levels import build_state
from factoriax.observations import global_array
from factoriax.state import EnvParams

pytestmark = pytest.mark.slow


def _run_agent(max_steps: int, seed: int = 0):
    """Drive the scripted rocket agent through the masked benchmark env.

    Returns:
        Tuple ``(final_mask, unlock_timestep)`` where ``final_mask`` is
        a length-NUM_ROCKET_ACHIEVEMENTS bool array and
        ``unlock_timestep[i]`` is the tick on which achievement ``i``
        first unlocked (or ``-1`` if never).
    """
    env_params = EnvParams(
        map_width=32,
        map_height=32,
        num_players=1,
        max_timesteps=max_steps,
    )
    level = build_rocket_level()
    state = build_state(level, env_params)
    env = ActionMaskWrapper(
        FactoriaXEnv(achievement_fn=rocket_conditions),
        ROCKET_BLOCKED_ACTIONS,
    )
    jit_step = jax.jit(env.step_env)
    jit_obs = jax.jit(lambda s: global_array(s, env_params, 0))
    agent = make_scripted_rocket_agent(env_params)

    unlock_timestep = np.full((NUM_ROCKET_ACHIEVEMENTS,), -1, dtype=np.int32)
    key = jax.random.PRNGKey(seed)

    for t in range(max_steps):
        obs = np.asarray(jit_obs(state))
        action = agent.act(obs)
        key, subkey = jax.random.split(key)
        _, state, _, done, _ = jit_step(
            subkey,
            state,
            jnp.int32(action),
            env_params,
        )
        mask = np.asarray(state.achievements_unlocked)[:NUM_ROCKET_ACHIEVEMENTS]
        newly = (unlock_timestep < 0) & mask
        unlock_timestep[newly] = t
        if agent.is_done or bool(done):
            break

    final_mask = np.asarray(state.achievements_unlocked)[:NUM_ROCKET_ACHIEVEMENTS]
    return final_mask, unlock_timestep


def _missing_names(mask: np.ndarray) -> list[str]:
    return [
        ROCKET_ACHIEVEMENT_INFO[i].id
        for i in range(NUM_ROCKET_ACHIEVEMENTS)
        if not mask[i]
    ]


def test_rocket_agent_unlocks_basic_tier() -> None:
    """Agent should at minimum unlock the five collect_* ore achievements."""
    mask, _ = _run_agent(max_steps=1500)
    ids = {info.id: i for i, info in enumerate(ROCKET_ACHIEVEMENT_INFO)}
    for name in (
        "collect_iron",
        "collect_copper",
        "collect_tin",
        "collect_coal",
        "collect_silicon",
    ):
        assert mask[ids[name]], (
            f"{name} should fire once the agent mines one of each ore"
        )


def test_rocket_agent_unlocks_all_achievements() -> None:
    """The scripted agent should unlock all 38 rocket achievements.

    Uses the 8000-step benchmark budget — the naive serial plan
    completes in ~5100 ticks.
    """
    mask, timing = _run_agent(max_steps=8000)
    missing = _missing_names(mask)
    timing_str = ", ".join(
        f"{ROCKET_ACHIEVEMENT_INFO[i].id}@{timing[i]}"
        for i in range(NUM_ROCKET_ACHIEVEMENTS)
        if timing[i] >= 0
    )
    assert not missing, (
        f"Missing {len(missing)}/{NUM_ROCKET_ACHIEVEMENTS} achievements: "
        f"{missing}\nUnlocked: {timing_str}"
    )
