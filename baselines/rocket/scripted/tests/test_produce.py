"""Machine-based production goal tests.

Each test seeds a state with enough raw materials for a small recipe
cycle, runs the production goal against the masked rocket env, and
asserts the output item lands in the player's inventory.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from baselines.rocket.scripted import goals, skills
from baselines.rocket.scripted.world_model import decode_observation
from factoriax.benchmarks.rocket import (
    ROCKET_BLOCKED_ACTIONS,
    build_rocket_level,
    rocket_conditions,
)
from factoriax.constants import ItemType
from factoriax.envs import FactoriaXEnv
from factoriax.envs.action_mask_wrapper import ActionMaskWrapper
from factoriax.levels import build_state
from factoriax.observations import global_array
from factoriax.state import EnvParams

pytestmark = pytest.mark.slow


def _setup_env(starting_inventory: dict[int, int]):
    """Build the masked rocket env with the given starting inventory.

    Returns ``(jit_step_fn, initial_state, env_params)``.
    """
    env_params = EnvParams(
        map_width=32,
        map_height=32,
        num_players=1,
        max_timesteps=500,
    )
    level = build_rocket_level()
    state = build_state(level, env_params)
    # Patch the starting player inventory with the seed items.
    inv = np.asarray(state.player_inventory).copy()
    for item_id, count in starting_inventory.items():
        inv[0, int(item_id)] = count
    state = state.replace(player_inventory=jnp.asarray(inv))
    env = ActionMaskWrapper(
        FactoriaXEnv(achievement_fn=rocket_conditions),
        ROCKET_BLOCKED_ACTIONS,
    )
    return jax.jit(env.step_env), state, env_params


_JIT_OBS_CACHE: dict[tuple[int, int, int], object] = {}


def _jit_obs(env_params: EnvParams):
    """Memoize JIT-compiled obs per env_params shape (see test_skills)."""
    key = (env_params.map_width, env_params.map_height, env_params.max_timesteps)
    fn = _JIT_OBS_CACHE.get(key)
    if fn is None:
        fn = jax.jit(lambda s: global_array(s, env_params, 0))
        _JIT_OBS_CACHE[key] = fn
    return fn


def _view(state, env_params: EnvParams):
    obs = np.asarray(_jit_obs(env_params)(state))
    return decode_observation(
        obs,
        map_height=env_params.map_height,
        map_width=env_params.map_width,
        max_timesteps=env_params.max_timesteps,
    )


def _rollout(state, goal, jit_step, env_params, max_steps: int = 400):
    key = jax.random.PRNGKey(0)
    for _ in range(max_steps):
        view = _view(state, env_params)
        result, action = goal.step(view)
        if result is skills.Result.DONE:
            return state, "done"
        if result is skills.Result.FAIL:
            return state, "fail"
        assert action is not None
        key, sub = jax.random.split(key)
        _, state, _, _, _ = jit_step(sub, state, jnp.int32(int(action)), env_params)
    return state, "timeout"


def test_produce_in_furnace_makes_iron_plate() -> None:
    """Feed the pre-placed furnace iron ore + coal → withdraw an iron plate."""
    jit_step, state, env_params = _setup_env(
        {int(ItemType.IRON_ORE): 4, int(ItemType.COAL): 2},
    )
    goal = goals.ProduceInFurnace(ItemType.IRON_PLATE, count=1)
    final_state, verdict = _rollout(state, goal, jit_step, env_params, max_steps=60)
    assert verdict == "done", f"got {verdict}"
    view = _view(final_state, env_params)
    assert view.player.held(ItemType.IRON_PLATE) >= 1


def test_produce_in_assembler_makes_wire() -> None:
    """Feed the pre-placed assembler copper + tin plates → withdraw wire.

    The WIRE recipe is ``COPPER_PLATE + TIN_PLATE`` (recipes.py); the
    earlier ``IRON_PLATE`` fixture was orphaned by the science-lab
    recipe rework that re-paired plate inputs.
    """
    jit_step, state, env_params = _setup_env(
        {
            int(ItemType.COPPER_PLATE): 2,
            int(ItemType.TIN_PLATE): 2,
        },
    )
    goal = goals.ProduceInAssembler(ItemType.WIRE, count=1)
    final_state, verdict = _rollout(state, goal, jit_step, env_params, max_steps=60)
    assert verdict == "done", f"got {verdict}"
    view = _view(final_state, env_params)
    assert view.player.held(ItemType.WIRE) >= 1
