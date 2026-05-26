"""Tests for :class:`baselines.rocket.scripted.goals.BuildSmelterCell`.

The smelter cell is Module 1 of the advanced factory: one miner
sitting on the south edge of an ore patch, an ore pallet south of
it, a furnace south of the pallet, and an arm pushing the smelt
output east into a plate pallet. Coal supply lives in Phase 3 (the
coal trunk); these tests exercise placement only and a short
production check with hand-injected coal.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from baselines.rocket.scripted import goals, skills
from baselines.rocket.scripted.world_model import decode_observation
from factoriax.constants import (
    Action,
    BlockType,
    Direction,
    ItemType,
    Machine,
)
from factoriax.envs import FactoriaXEnv
from factoriax.envs.action_mask_wrapper import ActionMaskWrapper
from factoriax.levels import LevelBuilder, build_state
from factoriax.observations import global_array
from factoriax.scenarios.rocket import (
    ROCKET_BLOCKED_ACTIONS,
    rocket_conditions,
)
from factoriax.state import EnvParams

pytestmark = pytest.mark.slow

_MAP_SIZE = 16
_PATCH_X = 5
_PATCH_Y = 5
_PATCH_SIZE = 3
_SPAWN = (8, 12)

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


def _build_iron_level(
    *,
    prebuild_cell: bool = False,
    coal_feeder: bool = False,
) -> tuple[object, object, EnvParams]:
    """Build a 16x16 level with one 3x3 IRON patch.

    Args:
        prebuild_cell: When true, the cell's machines are placed at
            level-build time (not by the goal under test) so we can
            isolate post-construction behaviour. The miner pallet is
            *not* preloaded; the engine's miner reduction does that.
        coal_feeder: When true, a pallet preloaded with 200 coal is
            placed south of the furnace. The furnace auto-pulls one
            coal per tick into its input slot — same role the Phase
            3 coal trunk will play, but without depending on the
            trunk module.

    Returns:
        ``(jit_step_fn, EnvState, env_params)``.
    """
    builder = LevelBuilder(_MAP_SIZE, _MAP_SIZE)
    builder.fill_rect(
        _PATCH_X, _PATCH_Y, _PATCH_SIZE, _PATCH_SIZE, BlockType.IRON, resources=280
    )
    builder.set_player_position(*_SPAWN)

    if prebuild_cell:
        # Coordinates must mirror BuildSmelterCell's layout exactly.
        mx = _PATCH_X + _PATCH_SIZE // 2
        my_se = _PATCH_Y + _PATCH_SIZE - 1
        builder.place_machine(mx, my_se, int(Machine.MINER), int(Direction.DOWN))
        # Ore feeder is a belt facing DOWN — combiners pull only
        # from facing belts under the directional Phase 0.
        builder.place_machine(
            mx, my_se + 1, int(Machine.CONVEYOR_BELT), int(Direction.DOWN)
        )
        builder.place_machine(mx, my_se + 2, int(Machine.FURNACE), int(Direction.DOWN))
        builder.place_machine(
            mx + 1,
            my_se + 2,
            int(Machine.ARM),
            int(Direction.RIGHT),
        )
        builder.place_machine(
            mx + 2,
            my_se + 2,
            int(Machine.PALLET),
            int(Direction.DOWN),
        )
        if coal_feeder:
            # Coal feeder belt south of the furnace, facing UP. The
            # furnace's directional Phase 0 pulls one coal per tick
            # from this belt because its direction points at the
            # furnace. The belt itself starts pre-loaded with coal.
            builder.place_machine(
                mx,
                my_se + 3,
                int(Machine.CONVEYOR_BELT),
                int(Direction.UP),
            )
            builder.set_machine_inventory(
                mx,
                my_se + 3,
                int(ItemType.COAL),
                200,
            )

    level = builder.build("smelter_cell_test")
    env_params = EnvParams(
        map_width=_MAP_SIZE,
        map_height=_MAP_SIZE,
        num_players=1,
        max_timesteps=400,
    )
    state = build_state(level, env_params)

    # Player bootstrap inventory: 1 MINER, 1 CONVEYOR_BELT (ore
    # feeder), 1 PALLET (plate-bus), 1 FURNACE, 1 ARM.
    inv = np.asarray(state.player_inventory).copy()
    inv[0, int(ItemType.MINER)] = 1
    inv[0, int(ItemType.CONVEYOR_BELT)] = 1
    inv[0, int(ItemType.PALLET)] = 1
    inv[0, int(ItemType.FURNACE)] = 1
    inv[0, int(ItemType.ARM)] = 1
    state = state.replace(player_inventory=jnp.asarray(inv))

    return _shared_jit_step(env_params), state, env_params


def test_build_smelter_cell_places_all_five_entities() -> None:
    """The goal places miner + 2 pallets + furnace + arm at expected tiles."""
    jit_step, state, env_params = _build_iron_level()
    goal = goals.BuildSmelterCell(_PATCH_X, _PATCH_Y, patch_size=_PATCH_SIZE)
    final_state, verdict = _rollout(state, goal, jit_step, env_params, max_steps=400)

    assert verdict == "done", f"got {verdict}"

    mx = _PATCH_X + _PATCH_SIZE // 2
    my_se = _PATCH_Y + _PATCH_SIZE - 1
    mt = np.asarray(final_state.machine_types)
    # Note: machine_types is indexed [y, x].
    assert mt[my_se, mx] == int(Machine.MINER)
    assert mt[my_se + 1, mx] == int(Machine.CONVEYOR_BELT)
    assert mt[my_se + 2, mx] == int(Machine.FURNACE)
    assert mt[my_se + 2, mx + 1] == int(Machine.ARM)
    assert mt[my_se + 2, mx + 2] == int(Machine.PALLET)


def test_build_smelter_cell_consumes_bootstrap_inventory() -> None:
    """After building, the player's inventory has zero machines left."""
    jit_step, state, env_params = _build_iron_level()
    goal = goals.BuildSmelterCell(_PATCH_X, _PATCH_Y, patch_size=_PATCH_SIZE)
    final_state, verdict = _rollout(state, goal, jit_step, env_params, max_steps=400)
    assert verdict == "done"
    inv = np.asarray(final_state.player_inventory[0])
    assert int(inv[int(ItemType.MINER)]) == 0
    assert int(inv[int(ItemType.PALLET)]) == 0
    assert int(inv[int(ItemType.CONVEYOR_BELT)]) == 0
    assert int(inv[int(ItemType.FURNACE)]) == 0
    assert int(inv[int(ItemType.ARM)]) == 0


def test_smelter_cell_produces_iron_plate_when_fed_coal() -> None:
    """A pre-built cell + adjacent coal pallet accumulates iron plates.

    Bypasses the goal under test by hand-placing the cell at level
    build time, then feeds coal via an adjacent pallet so we can
    validate the layout's *output* behaviour without depending on
    the still-to-come coal trunk (Phase 3). The adjacent-pallet
    feed mirrors the trunk's eventual role — one coal pulled per
    tick into the furnace's input slot.
    """
    jit_step, state, env_params = _build_iron_level(
        prebuild_cell=True,
        coal_feeder=True,
    )

    mx = _PATCH_X + _PATCH_SIZE // 2
    my_se = _PATCH_Y + _PATCH_SIZE - 1
    plate_pallet_tile = (mx + 2, my_se + 2)

    # Run NOOPs for ~150 ticks; that's plenty for several smelt cycles
    # (recipe ticks=2, plus arm transfers and miner pushes).
    key = jax.random.PRNGKey(0)
    for _ in range(150):
        key, sub = jax.random.split(key)
        _, state, _, _, _ = jit_step(
            sub,
            state,
            jnp.int32(int(Action.NOOP)),
            env_params,
        )

    pallet_eid = int(state.tile_entity[plate_pallet_tile[1], plate_pallet_tile[0]])
    assert pallet_eid >= 0
    plate_buf = int(state.ent_buf_count[pallet_eid])
    plate_type = int(state.ent_buf_type[pallet_eid])
    assert plate_type == int(ItemType.IRON_PLATE), (
        f"expected IRON_PLATE in plate pallet, got ItemType={plate_type}"
    )
    assert plate_buf > 0, f"plate pallet empty after 150 ticks; buf={plate_buf}"
