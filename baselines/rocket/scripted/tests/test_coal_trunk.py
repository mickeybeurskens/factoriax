"""Tests for :class:`baselines.rocket.scripted.goals.BuildCoalTrunk`.

Module 2 of the advanced factory: feeder arm pulling coal off a
source pallet, plus a chain of belts carrying the coal across the
map. Validated in two stages:

1. **Placement test** — run the goal, assert every belt + the
   feeder arm land at expected tiles with expected facings.
2. **Flow test** — pre-build the trunk via LevelBuilder, preload
   the source pallet with coal, run NOOPs, assert the destination
   pallet at the trunk terminus accumulates coal. This validates
   that the engine fix from commit 61ff84a (arms drain ent_asm_out)
   plus run_conveyor_belts' existing buf-to-buf push form a
   working chain.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from baselines.rocket.scripted import goals, skills
from baselines.rocket.scripted.world_model import decode_observation
from factoriax.engine.constants import (
    Action,
    Direction,
    ItemType,
    Machine,
)
from factoriax.engine.levels import LevelBuilder, build_state
from factoriax.engine.observations import global_array
from factoriax.engine.state import EnvParams
from factoriax.envs import FactoriaXEnv
from factoriax.envs.action_mask_wrapper import ActionMaskWrapper
from factoriax.scenarios.rocket import (
    ROCKET_BLOCKED_ACTIONS,
    rocket_conditions,
)

pytestmark = pytest.mark.slow

_MAP_SIZE = 12
_SPAWN = (1, 1)

# Trunk geometry. Source pallet at (3, 4) (NORTH of arm); feeder arm
# at (3, 5) facing DOWN, so its "behind" is the source pallet and its
# front pushes south onto belt 0 at (3, 6). Belts 0-3 carry coal east;
# the corner belt at (7, 6) faces UP, pushing north into the
# destination pallet at (7, 5).
#
# Geometric note: the feeder arm's stand tile is the source pallet's
# tile. The placement-test fixture therefore omits the source pallet
# at level-build time (the goal places the arm into dirt) and only
# the flow-test fixture pre-places everything via LevelBuilder, which
# bypasses the walkability check.
_SOURCE_PALLET = (3, 4)
_FEEDER_ARM = (3, 5)
_FEEDER_ARM_DIR = int(Direction.DOWN)
_BELT_SPECS: list[tuple[tuple[int, int], int]] = [
    ((3, 6), int(Direction.RIGHT)),
    ((4, 6), int(Direction.RIGHT)),
    ((5, 6), int(Direction.RIGHT)),
    ((6, 6), int(Direction.RIGHT)),
    ((7, 6), int(Direction.UP)),
]
_DEST_PALLET = (7, 5)

_JIT_STEP_CACHE: dict[tuple[int, int, int], object] = {}
_JIT_OBS_CACHE: dict[tuple[int, int, int], object] = {}


def _shared_jit_step(env_params: EnvParams):
    key = (env_params.map_width, env_params.map_height, env_params.max_timesteps)
    fn = _JIT_STEP_CACHE.get(key)
    if fn is None:
        env = ActionMaskWrapper(
            FactoriaXEnv(achievement_fn=rocket_conditions),
            ROCKET_BLOCKED_ACTIONS,
        )
        fn = jax.jit(env.step_env)
        _JIT_STEP_CACHE[key] = fn
    return fn


def _jit_obs(env_params: EnvParams):
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


def _build_trunk_level(*, prebuild_trunk: bool = False, coal_count: int = 100):
    """Construct the trunk test level.

    Args:
        prebuild_trunk: When true, place the feeder arm + every belt
            at level-build time so the flow test can run NOOPs without
            depending on BuildCoalTrunk's placement code.
        coal_count: Items in the source pallet at level start.

    Returns:
        ``(jit_step_fn, EnvState, env_params)``.
    """
    builder = LevelBuilder(_MAP_SIZE, _MAP_SIZE)
    builder.set_player_position(*_SPAWN)

    # Destination pallet (empty) is always present.
    builder.place_machine(
        *_DEST_PALLET,
        int(Machine.PALLET),
        int(Direction.DOWN),
    )

    if prebuild_trunk:
        # Flow test path: build the entire trunk + source pallet at
        # level-build time. LevelBuilder doesn't enforce walkability,
        # so this dodges the placement geometry conflict entirely.
        builder.place_machine(
            *_SOURCE_PALLET,
            int(Machine.PALLET),
            int(Direction.DOWN),
        )
        builder.set_machine_inventory(
            _SOURCE_PALLET[0],
            _SOURCE_PALLET[1],
            int(ItemType.COAL),
            coal_count,
        )
        builder.place_machine(
            *_FEEDER_ARM,
            int(Machine.ARM),
            _FEEDER_ARM_DIR,
        )
        for tile, direction in _BELT_SPECS:
            builder.place_machine(
                *tile,
                int(Machine.CONVEYOR_BELT),
                int(direction),
            )

    level = builder.build("coal_trunk_test")
    env_params = EnvParams(
        map_width=_MAP_SIZE,
        map_height=_MAP_SIZE,
        num_players=1,
        max_timesteps=400,
    )
    state = build_state(level, env_params)

    inv = np.asarray(state.player_inventory).copy()
    inv[0, int(ItemType.ARM)] = 1
    inv[0, int(ItemType.CONVEYOR_BELT)] = len(_BELT_SPECS)
    state = state.replace(player_inventory=jnp.asarray(inv))

    return _shared_jit_step(env_params), state, env_params


def test_build_coal_trunk_places_arm_and_belts() -> None:
    """The goal places the feeder arm and every belt at expected tiles."""
    jit_step, state, env_params = _build_trunk_level()
    goal = goals.BuildCoalTrunk(_FEEDER_ARM, _FEEDER_ARM_DIR, _BELT_SPECS)
    final_state, verdict = _rollout(state, goal, jit_step, env_params, max_steps=400)

    assert verdict == "done", f"got {verdict}"

    mt = np.asarray(final_state.machine_types)
    tile_entity = np.asarray(final_state.tile_entity)
    ent_direction = np.asarray(final_state.ent_direction)

    arm_eid = int(tile_entity[_FEEDER_ARM[1], _FEEDER_ARM[0]])
    assert mt[_FEEDER_ARM[1], _FEEDER_ARM[0]] == int(Machine.ARM)
    assert arm_eid >= 0
    assert int(ent_direction[arm_eid]) == _FEEDER_ARM_DIR

    for (x, y), direction in _BELT_SPECS:
        belt_eid = int(tile_entity[y, x])
        assert mt[y, x] == int(Machine.CONVEYOR_BELT), f"expected belt at ({x}, {y})"
        assert belt_eid >= 0
        assert int(ent_direction[belt_eid]) == int(direction), (
            f"belt at ({x}, {y}) facing {int(ent_direction[belt_eid])}, "
            f"expected {direction}"
        )


def test_build_coal_trunk_consumes_bootstrap_inventory() -> None:
    """After building, the player's inventory has zero ARM and BELTs left."""
    jit_step, state, env_params = _build_trunk_level()
    goal = goals.BuildCoalTrunk(_FEEDER_ARM, _FEEDER_ARM_DIR, _BELT_SPECS)
    final_state, _ = _rollout(state, goal, jit_step, env_params, max_steps=400)
    inv = np.asarray(final_state.player_inventory[0])
    assert int(inv[int(ItemType.ARM)]) == 0
    assert int(inv[int(ItemType.CONVEYOR_BELT)]) == 0


def test_coal_trunk_delivers_coal_to_destination() -> None:
    """Pre-built trunk + loaded source pallet → coal accumulates at sink.

    Bypasses the goal under test: the trunk is placed at level-build
    time so we can validate the end-to-end flow on its own. Each tick
    the feeder arm moves one coal onto belt 0; belts push along the
    chain at one tile per tick; the corner belt at (7, 6) pushes
    north into the destination pallet at (7, 5). A 60-tick window is
    plenty for several handoffs.
    """
    jit_step, state, env_params = _build_trunk_level(prebuild_trunk=True)

    key = jax.random.PRNGKey(0)
    for _ in range(60):
        key, sub = jax.random.split(key)
        _, state, _, _, _ = jit_step(
            sub,
            state,
            jnp.int32(int(Action.NOOP)),
            env_params,
        )

    dest_eid = int(state.tile_entity[_DEST_PALLET[1], _DEST_PALLET[0]])
    assert dest_eid >= 0
    dest_buf = int(state.ent_buf_count[dest_eid])
    dest_type = int(state.ent_buf_type[dest_eid])
    assert dest_type == int(ItemType.COAL), (
        f"expected COAL in destination pallet, got ItemType={dest_type}"
    )
    assert dest_buf > 0, f"destination empty after 60 ticks; buf={dest_buf}"
