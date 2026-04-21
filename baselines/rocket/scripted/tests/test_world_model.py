"""Tests for baselines.rocket.scripted.world_model.

The decoder is validated by round-tripping: construct a known state via
the benchmark's level builder, generate its global observation, decode
it with :func:`decode_observation`, and check that every derived view
matches the original state.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from baselines.rocket.scripted import world_model as wm
from factoriax.benchmarks.rocket import build_rocket_level, rocket_conditions
from factoriax.constants import (
    MAX_ACHIEVEMENTS,
    Action,
    Direction,
    ItemType,
    MachineType,
)
from factoriax.envs import FactoriaXEnv
from factoriax.envs.achievement_wrapper import AchievementState, AchievementWrapper
from factoriax.levels import build_state
from factoriax.observations import global_array
from factoriax.state import EnvParams

# ---------------------------------------------------------------------------
# Helpers for synthesising states without stepping the env.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rocket_env_params() -> EnvParams:
    """Rocket benchmark EnvParams (32x32, 2000 steps)."""
    return EnvParams(map_width=32, map_height=32, num_players=1, max_timesteps=2000)


@pytest.fixture(scope="module")
def rocket_initial_state(rocket_env_params: EnvParams):
    """Deterministic initial env state from the rocket level."""
    level = build_rocket_level()
    return build_state(level, rocket_env_params)


def _obs_from_state(state, params: EnvParams) -> np.ndarray:
    """Compute global_array for player 0 and copy to a numpy array."""
    return np.asarray(global_array(state, params, 0))


# ---------------------------------------------------------------------------
# Shape + round-trip tests
# ---------------------------------------------------------------------------


def test_decode_shape_and_dtypes(
    rocket_initial_state,
    rocket_env_params: EnvParams,
) -> None:
    """Decoded grids have the right shape and integer dtype."""
    obs = _obs_from_state(rocket_initial_state, rocket_env_params)
    view = wm.decode_observation(
        obs,
        map_height=rocket_env_params.map_height,
        map_width=rocket_env_params.map_width,
        max_timesteps=rocket_env_params.max_timesteps,
    )
    shape = (rocket_env_params.map_height, rocket_env_params.map_width)
    assert view.block_type.shape == shape
    assert view.machine_type.shape == shape
    assert view.block_resources.shape == shape
    assert view.buffer_type.shape == shape
    assert view.walkable.shape == shape
    assert view.block_type.dtype == np.int32
    assert view.machine_type.dtype == np.int32
    assert view.walkable.dtype == bool


def test_decode_block_type_round_trip(
    rocket_initial_state,
    rocket_env_params: EnvParams,
) -> None:
    """Every tile's block_type survives obs round-trip."""
    obs = _obs_from_state(rocket_initial_state, rocket_env_params)
    view = wm.decode_observation(
        obs,
        map_height=rocket_env_params.map_height,
        map_width=rocket_env_params.map_width,
        max_timesteps=rocket_env_params.max_timesteps,
    )
    expected = np.asarray(rocket_initial_state.map)
    np.testing.assert_array_equal(view.block_type, expected)


def test_decode_machine_type_round_trip(
    rocket_initial_state,
    rocket_env_params: EnvParams,
) -> None:
    """Machine grid round-trips (matches engine state)."""
    obs = _obs_from_state(rocket_initial_state, rocket_env_params)
    view = wm.decode_observation(
        obs,
        map_height=rocket_env_params.map_height,
        map_width=rocket_env_params.map_width,
        max_timesteps=rocket_env_params.max_timesteps,
    )
    expected = np.asarray(rocket_initial_state.machine_types)
    np.testing.assert_array_equal(view.machine_type, expected)
    # Rocket level pre-places exactly one furnace and one assembler.
    assert int((view.machine_type == int(MachineType.FURNACE)).sum()) == 1
    assert int((view.machine_type == int(MachineType.ASSEMBLER)).sum()) == 1


def test_decode_block_resources_round_trip(
    rocket_initial_state,
    rocket_env_params: EnvParams,
) -> None:
    """Resource counts per tile survive round-trip."""
    obs = _obs_from_state(rocket_initial_state, rocket_env_params)
    view = wm.decode_observation(
        obs,
        map_height=rocket_env_params.map_height,
        map_width=rocket_env_params.map_width,
        max_timesteps=rocket_env_params.max_timesteps,
    )
    expected = np.asarray(rocket_initial_state.block_resources)
    np.testing.assert_array_equal(view.block_resources, expected)


def test_decode_player_pos_and_direction(
    rocket_initial_state,
    rocket_env_params: EnvParams,
) -> None:
    """Player position and direction round-trip."""
    obs = _obs_from_state(rocket_initial_state, rocket_env_params)
    view = wm.decode_observation(
        obs,
        map_height=rocket_env_params.map_height,
        map_width=rocket_env_params.map_width,
        max_timesteps=rocket_env_params.max_timesteps,
    )
    expected_pos = np.asarray(rocket_initial_state.player_positions[0])
    assert view.player.pos == (int(expected_pos[0]), int(expected_pos[1]))
    expected_dir = int(rocket_initial_state.player_directions[0])
    assert view.player.direction == expected_dir


def test_decode_inventory_round_trip(
    rocket_initial_state,
    rocket_env_params: EnvParams,
) -> None:
    """Player inventory recovered exactly when starting empty."""
    obs = _obs_from_state(rocket_initial_state, rocket_env_params)
    view = wm.decode_observation(
        obs,
        map_height=rocket_env_params.map_height,
        map_width=rocket_env_params.map_width,
        max_timesteps=rocket_env_params.max_timesteps,
    )
    expected = np.asarray(rocket_initial_state.player_inventory[0])
    np.testing.assert_array_equal(view.player.inventory, expected)
    # Rocket benchmark starts with empty inventory.
    assert view.player.inventory.sum() == 0


# ---------------------------------------------------------------------------
# Derived-view tests
# ---------------------------------------------------------------------------


def test_ore_tiles_finds_all_five_patches(
    rocket_initial_state,
    rocket_env_params: EnvParams,
) -> None:
    """The five ore types each have at least one patch tile."""
    obs = _obs_from_state(rocket_initial_state, rocket_env_params)
    view = wm.decode_observation(
        obs,
        map_height=rocket_env_params.map_height,
        map_width=rocket_env_params.map_width,
        max_timesteps=rocket_env_params.max_timesteps,
    )
    for item in (
        ItemType.IRON_ORE,
        ItemType.COPPER_ORE,
        ItemType.TIN_ORE,
        ItemType.COAL,
        ItemType.SILICON,
    ):
        tiles = view.ore_tiles(item)
        assert tiles, f"no {ItemType(int(item)).name} patch found"
        # Each reported tile should really hold the expected block.
        for x, y in tiles:
            assert view.block_resources[y, x] > 0


def test_walkable_excludes_water_and_machines(
    rocket_initial_state,
    rocket_env_params: EnvParams,
) -> None:
    """Walkable mask matches non-water, non-machine tiles."""
    obs = _obs_from_state(rocket_initial_state, rocket_env_params)
    view = wm.decode_observation(
        obs,
        map_height=rocket_env_params.map_height,
        map_width=rocket_env_params.map_width,
        max_timesteps=rocket_env_params.max_timesteps,
    )
    # The rocket level has no water, and exactly 2 machines (furnace +
    # assembler) pre-placed. Only those 2 tiles should be non-walkable.
    non_walkable = (~view.walkable).sum()
    assert int(non_walkable) == 2


def test_player_spawn_is_walkable(
    rocket_initial_state,
    rocket_env_params: EnvParams,
) -> None:
    """Spawn tile must itself be a valid standing tile."""
    obs = _obs_from_state(rocket_initial_state, rocket_env_params)
    view = wm.decode_observation(
        obs,
        map_height=rocket_env_params.map_height,
        map_width=rocket_env_params.map_width,
        max_timesteps=rocket_env_params.max_timesteps,
    )
    px, py = view.player.pos
    assert view.walkable[py, px]


# ---------------------------------------------------------------------------
# BFS pathfinding tests
# ---------------------------------------------------------------------------


def test_plan_path_to_self_is_empty(
    rocket_initial_state,
    rocket_env_params: EnvParams,
) -> None:
    """No actions needed when goal is the player's own tile."""
    obs = _obs_from_state(rocket_initial_state, rocket_env_params)
    view = wm.decode_observation(
        obs,
        map_height=rocket_env_params.map_height,
        map_width=rocket_env_params.map_width,
        max_timesteps=rocket_env_params.max_timesteps,
    )
    path = view.plan_path(view.player.pos, mode="exactly_on")
    assert path == []


def test_plan_path_reaches_nearest_ore(
    rocket_initial_state,
    rocket_env_params: EnvParams,
) -> None:
    """A path exists from spawn to every ore type's nearest tile."""
    obs = _obs_from_state(rocket_initial_state, rocket_env_params)
    view = wm.decode_observation(
        obs,
        map_height=rocket_env_params.map_height,
        map_width=rocket_env_params.map_width,
        max_timesteps=rocket_env_params.max_timesteps,
    )
    for item in (
        ItemType.IRON_ORE,
        ItemType.COPPER_ORE,
        ItemType.TIN_ORE,
        ItemType.COAL,
        ItemType.SILICON,
    ):
        tiles = view.ore_tiles(item)
        assert tiles
        # Pick any tile — a path must exist on this mostly-open map.
        target = tiles[0]
        path = view.plan_path(target, mode="adjacent_or_on")
        assert path is not None
        # All actions are legal movement actions.
        for a in path:
            assert a in {
                int(Action.UP),
                int(Action.DOWN),
                int(Action.LEFT),
                int(Action.RIGHT),
            }


def test_plan_path_optimal_length_on_synthetic_grid() -> None:
    """BFS returns an optimal-length path on a hand-built grid."""
    # 5x5 walkable grid with player at (0, 0), target at (4, 4).
    h, w = 5, 5
    walkable = np.ones((h, w), dtype=bool)
    start = (0, 0)
    target_set = {(4, 4)}
    path = wm._bfs(walkable, start, target_set)
    assert path is not None
    assert len(path) == 8  # Manhattan distance, grid is open


def test_plan_path_blocked_returns_none() -> None:
    """BFS returns None when the target is unreachable."""
    h, w = 3, 3
    walkable = np.ones((h, w), dtype=bool)
    # Wall the right column off from the left.
    walkable[:, 1] = False
    start = (0, 0)
    target_set = {(2, 2)}
    path = wm._bfs(walkable, start, target_set)
    assert path is None


# ---------------------------------------------------------------------------
# Action-helper tests
# ---------------------------------------------------------------------------


def test_direction_toward_returns_expected_direction() -> None:
    """direction_toward maps adjacency deltas to the right Direction."""
    assert wm.direction_toward((3, 3), (2, 3)) == int(Direction.LEFT)
    assert wm.direction_toward((3, 3), (4, 3)) == int(Direction.RIGHT)
    assert wm.direction_toward((3, 3), (3, 2)) == int(Direction.UP)
    assert wm.direction_toward((3, 3), (3, 4)) == int(Direction.DOWN)
    assert wm.direction_toward((3, 3), (5, 3)) is None  # not 4-adjacent


def test_face_action_maps_to_correct_action() -> None:
    assert wm.face_action(int(Direction.UP)) == int(Action.FACE_UP)
    assert wm.face_action(int(Direction.LEFT)) == int(Action.FACE_LEFT)


def test_withdraw_action_matches_enum() -> None:
    assert wm.withdraw_action() == int(Action.WITHDRAW)


def test_deposit_action_matches_enum() -> None:
    assert wm.deposit_action(ItemType.COAL) == int(Action.DEPOSIT_COAL)
    assert wm.deposit_action(ItemType.MOTOR) == int(Action.DEPOSIT_MOTOR)


def test_place_action_matches_enum() -> None:
    assert wm.place_action(MachineType.MINER) == int(Action.PLACE_MINER)
    assert wm.place_action(MachineType.ROCKET) == int(Action.PLACE_ROCKET)


# ---------------------------------------------------------------------------
# Integration: stepping the env through the decoder
# ---------------------------------------------------------------------------


def test_decode_tracks_player_movement(
    rocket_initial_state,
    rocket_env_params: EnvParams,
) -> None:
    """After emitting MOVE_DOWN, decoded pos reflects the new coordinates."""
    env = AchievementWrapper(FactoriaXEnv(), rocket_conditions)
    ach_state = AchievementState(
        env_state=rocket_initial_state,
        achievements_unlocked=jnp.zeros(MAX_ACHIEVEMENTS, dtype=jnp.bool_),
    )
    key = jax.random.PRNGKey(0)
    _, new_state, _, _, _ = env.step_env(
        key,
        ach_state,
        jnp.int32(int(Action.DOWN)),
        rocket_env_params,
    )
    obs_new = np.asarray(
        global_array(new_state.env_state, rocket_env_params, 0),
    )
    view_new = wm.decode_observation(
        obs_new,
        map_height=rocket_env_params.map_height,
        map_width=rocket_env_params.map_width,
        max_timesteps=rocket_env_params.max_timesteps,
    )
    old_pos = (
        int(rocket_initial_state.player_positions[0, 0]),
        int(rocket_initial_state.player_positions[0, 1]),
    )
    expected_new_pos = (old_pos[0], old_pos[1] + 1)
    assert view_new.player.pos == expected_new_pos
