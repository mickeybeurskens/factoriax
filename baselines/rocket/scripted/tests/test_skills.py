"""Tests for skills and goals against the real FactoriaX env.

Each test drives the scripted agent's inner skill/goal loop by
stepping the wrapped env directly. Tests are kept short (<~200 env
steps) so the amortised JIT compile is the dominant cost, not the
rollout.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from baselines.rocket.scripted import goals, skills
from baselines.rocket.scripted.world_model import decode_observation
from factoriax.benchmarks.rocket import build_rocket_level, rocket_conditions
from factoriax.constants import MAX_ACHIEVEMENTS, Action, ItemType
from factoriax.envs import FactoriaXEnv
from factoriax.envs.achievement_wrapper import AchievementState, AchievementWrapper
from factoriax.levels import build_state
from factoriax.observations import global_array
from factoriax.state import EnvParams

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Fixtures: a shared wrapped env + jitted step so each test reuses JIT work.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def env_params() -> EnvParams:
    return EnvParams(map_width=32, map_height=32, num_players=1, max_timesteps=2000)


@pytest.fixture(scope="module")
def initial_state(env_params: EnvParams):
    level = build_rocket_level()
    env_state = build_state(level, env_params)
    return AchievementState(
        env_state=env_state,
        achievements_unlocked=jnp.zeros(MAX_ACHIEVEMENTS, dtype=jnp.bool_),
    )


@pytest.fixture(scope="module")
def jit_step():
    env = AchievementWrapper(FactoriaXEnv(), rocket_conditions)
    return env, jax.jit(env.step_env)


def _view(state, env_params: EnvParams):
    """Decode a fresh WorldView from the current state."""
    obs = np.asarray(global_array(state.env_state, env_params, 0))
    return decode_observation(
        obs,
        map_height=env_params.map_height,
        map_width=env_params.map_width,
        max_timesteps=env_params.max_timesteps,
    )


def _rollout(
    state,
    skill_or_goal,
    env_params: EnvParams,
    jit_step_fn,
    env,
    max_steps: int = 200,
) -> tuple[object, list[int], str]:
    """Run *skill_or_goal* to completion; return (final_state, actions, verdict)."""
    key = jax.random.PRNGKey(0)
    actions: list[int] = []
    for _ in range(max_steps):
        view = _view(state, env_params)
        result, action = skill_or_goal.step(view)
        if result is skills.Result.DONE:
            return state, actions, "done"
        if result is skills.Result.FAIL:
            return state, actions, "fail"
        assert action is not None
        actions.append(int(action))
        key, subkey = jax.random.split(key)
        _, state, _, _, _ = jit_step_fn(
            subkey,
            state,
            jnp.int32(int(action)),
            env_params,
        )
    return state, actions, "timeout"


# ---------------------------------------------------------------------------
# NavigateAdjacent
# ---------------------------------------------------------------------------


def test_navigate_adjacent_reaches_ore(initial_state, env_params, jit_step):
    """Navigator should arrive adjacent to the nearest iron ore patch."""
    env, step_fn = jit_step
    start_view = _view(initial_state, env_params)
    target = start_view.ore_tiles(ItemType.IRON_ORE)[0]

    nav = skills.NavigateAdjacent(target)
    state, actions, verdict = _rollout(
        initial_state,
        nav,
        env_params,
        step_fn,
        env,
        max_steps=100,
    )
    assert verdict == "done", f"got {verdict} after {len(actions)} steps"
    final_view = _view(state, env_params)
    assert target in final_view.adjacent_tiles(final_view.player.pos) or (
        final_view.player.pos == target
    )


# ---------------------------------------------------------------------------
# FaceAndInteract (via mining)
# ---------------------------------------------------------------------------


def test_stand_on_and_act_mines_one_ore(initial_state, env_params, jit_step):
    """StandOnAndAct(MINE) on an ore tile should raise ore inventory by 1.

    Mining reads the block under the player (``game_logic.mine_block``
    uses ``state.player_positions``), so the agent must stand ON the
    ore tile, not adjacent to it.
    """
    env, step_fn = jit_step
    start_view = _view(initial_state, env_params)
    ore_item = ItemType.IRON_ORE
    target = start_view.ore_tiles(ore_item)[0]

    skill = skills.StandOnAndAct(target, int(Action.MINE))
    state, actions, verdict = _rollout(
        initial_state,
        skill,
        env_params,
        step_fn,
        env,
        max_steps=150,
    )
    assert verdict == "done"
    final_view = _view(state, env_params)
    assert final_view.player.held(ore_item) >= 1


# ---------------------------------------------------------------------------
# MineOre goal
# ---------------------------------------------------------------------------


def test_mine_ore_goal_accumulates(initial_state, env_params, jit_step):
    """MineOre should eventually satisfy a count target."""
    env, step_fn = jit_step
    goal = goals.MineOre(ItemType.IRON_ORE, count=2)
    state, actions, verdict = _rollout(
        initial_state,
        goal,
        env_params,
        step_fn,
        env,
        max_steps=400,
    )
    assert verdict == "done"
    final_view = _view(state, env_params)
    assert final_view.player.held(ItemType.IRON_ORE) >= 2


# ---------------------------------------------------------------------------
# CraftItem goal
# ---------------------------------------------------------------------------


def test_craft_item_fails_without_ingredients(initial_state, env_params, jit_step):
    """CraftItem reports FAIL when ingredients aren't in inventory."""
    env, step_fn = jit_step
    goal = goals.CraftItem(ItemType.IRON_PLATE, count=1)
    _, _, verdict = _rollout(
        initial_state,
        goal,
        env_params,
        step_fn,
        env,
        max_steps=10,
    )
    assert verdict == "fail"
