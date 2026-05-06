"""Tests for the FactoriaX environment."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from factoriax import (
    Action,
    BlockType,
    Direction,
    EnvParams,
    EnvState,
    make_factoriax_env,
)
from factoriax.constants import (
    NUM_ACTIONS,
    SOLID_BLOCKS,
    MachineType,
)
from factoriax.game_logic import (
    get_block_at,
    is_game_over,
    is_position_in_bounds,
    is_position_walkable,
    move_player,
)
from factoriax.observations import NUM_PLAYER_SCALARS, NUM_SPATIAL_CHANNELS
from factoriax.renderer import (
    create_default_textures,
)
from factoriax.world_gen import generate_world


class TestConstants:
    """Tests for constants module."""

    def test_resource_blocks_are_walkable(self) -> None:
        """Resource blocks should not be in SOLID_BLOCKS."""
        solid_set = set(int(b) for b in SOLID_BLOCKS)
        assert int(BlockType.IRON) not in solid_set
        assert int(BlockType.COPPER) not in solid_set
        assert int(BlockType.COAL) not in solid_set


class TestWorldGen:
    """Tests for world generation."""

    def test_generate_world_creates_valid_state(self) -> None:
        """Generated world should have valid state structure."""
        rng = random.PRNGKey(0)
        params = EnvParams()
        state = generate_world(rng, params)

        assert state.map.shape == (params.map_height, params.map_width)
        assert state.player_positions.shape == (params.num_players, 2)
        assert state.timestep == 0

    def test_player_spawns_on_dirt(self) -> None:
        """Players should always spawn on dirt tiles."""
        rng = random.PRNGKey(0)
        params = EnvParams()
        state = generate_world(rng, params)

        for i in range(params.num_players):
            px, py = state.player_positions[i]
            assert state.map[py, px] == BlockType.DIRT

    def test_world_gen_is_deterministic(self) -> None:
        """Same seed should produce same world."""
        params = EnvParams()
        state1 = generate_world(random.PRNGKey(42), params)
        state2 = generate_world(random.PRNGKey(42), params)

        assert jnp.array_equal(state1.map, state2.map)
        assert jnp.array_equal(state1.player_positions, state2.player_positions)

    def test_different_seeds_produce_different_worlds(self) -> None:
        """Different seeds should produce different worlds."""
        params = EnvParams()
        state1 = generate_world(random.PRNGKey(0), params)
        state2 = generate_world(random.PRNGKey(1), params)

        assert not jnp.array_equal(state1.map, state2.map)


class TestEnvStateSchema:
    """Tests for the EnvState schema — fields the engine guarantees."""

    def test_generate_state_initializes_achievements_unlocked(self) -> None:
        """Procedural state has all-False achievements_unlocked of correct shape."""
        from factoriax.constants import MAX_ACHIEVEMENTS
        from factoriax.levels import generate_state

        state = generate_state(random.PRNGKey(0), EnvParams())

        assert state.achievements_unlocked.shape == (MAX_ACHIEVEMENTS,)
        assert state.achievements_unlocked.dtype == jnp.bool_
        assert not bool(state.achievements_unlocked.any())

    def test_build_state_initializes_achievements_unlocked(self) -> None:
        """Level-built state has all-False achievements_unlocked of correct shape."""
        from factoriax.constants import MAX_ACHIEVEMENTS
        from factoriax.levels import build_state, get_level

        level = get_level("15x15_resources")
        params = EnvParams(map_width=15, map_height=15, num_players=1)
        state = build_state(level, params)

        assert state.achievements_unlocked.shape == (MAX_ACHIEVEMENTS,)
        assert state.achievements_unlocked.dtype == jnp.bool_
        assert not bool(state.achievements_unlocked.any())


class TestGameLogic:
    """Tests for game logic."""

    @pytest.fixture
    def simple_state(self, state_factory) -> EnvState:
        """Create a simple state for testing."""
        world_map = jnp.array(
            [
                [BlockType.DIRT, BlockType.DIRT, BlockType.WATER],
                [BlockType.DIRT, BlockType.DIRT, BlockType.DIRT],
                [BlockType.WATER, BlockType.DIRT, BlockType.DIRT],
            ],
            dtype=jnp.int32,
        )
        return state_factory(world_map=world_map, player_position=(1, 1))

    def test_is_position_in_bounds(self) -> None:
        """Position bounds checking should work correctly."""
        assert is_position_in_bounds(jnp.array([0, 0]), 3, 3)
        assert is_position_in_bounds(jnp.array([2, 2]), 3, 3)
        assert not is_position_in_bounds(jnp.array([-1, 0]), 3, 3)
        assert not is_position_in_bounds(jnp.array([3, 0]), 3, 3)
        assert not is_position_in_bounds(jnp.array([0, -1]), 3, 3)
        assert not is_position_in_bounds(jnp.array([0, 3]), 3, 3)

    def test_get_block_at_returns_correct_block(
        self,
        simple_state: EnvState,
    ) -> None:
        """Should return correct block type at position."""
        assert get_block_at(simple_state, jnp.array([0, 0])) == BlockType.DIRT
        assert get_block_at(simple_state, jnp.array([2, 0])) == BlockType.WATER

    def test_get_block_at_out_of_bounds(
        self,
        simple_state: EnvState,
    ) -> None:
        """Out of bounds positions should return OUT_OF_BOUNDS."""
        assert get_block_at(simple_state, jnp.array([-1, 0])) == BlockType.OUT_OF_BOUNDS
        assert get_block_at(simple_state, jnp.array([0, 5])) == BlockType.OUT_OF_BOUNDS

    def test_is_position_walkable_dirt(
        self,
        simple_state: EnvState,
    ) -> None:
        """Dirt should be walkable."""
        assert is_position_walkable(simple_state, jnp.array([0, 0]))
        assert is_position_walkable(simple_state, jnp.array([1, 1]))

    def test_is_position_walkable_water(
        self,
        simple_state: EnvState,
    ) -> None:
        """Water should not be walkable."""
        assert not is_position_walkable(simple_state, jnp.array([2, 0]))
        assert not is_position_walkable(simple_state, jnp.array([0, 2]))

    def test_is_position_walkable_out_of_bounds(
        self,
        simple_state: EnvState,
    ) -> None:
        """Out of bounds should not be walkable."""
        assert not is_position_walkable(simple_state, jnp.array([-1, 0]))
        assert not is_position_walkable(simple_state, jnp.array([5, 5]))

    def test_is_position_walkable_conveyor_belt(
        self,
        state_factory,
    ) -> None:
        """Conveyor belts should be walkable despite being machines."""
        world_map = jnp.array(
            [[BlockType.DIRT, BlockType.DIRT, BlockType.DIRT]],
            dtype=jnp.int32,
        )
        machine_types = jnp.array(
            [[MachineType.NONE, MachineType.CONVEYOR_BELT, MachineType.MINER]],
            dtype=jnp.int32,
        )
        state = state_factory(
            world_map=world_map,
            player_position=(0, 0),
            machine_types=machine_types,
        )
        # Belt tile is walkable
        assert is_position_walkable(state, jnp.array([1, 0]))
        # Miner tile is not walkable
        assert not is_position_walkable(state, jnp.array([2, 0]))

    def test_up_moves_north(self, state_factory) -> None:
        """UP should move the player north (y-1) on the map."""
        state = state_factory(
            world_map=jnp.full((3, 3), BlockType.DIRT, dtype=jnp.int32),
            player_position=(1, 1),
            player_direction=int(Direction.RIGHT),
        )
        new_state = move_player(state, Action.UP, 0)
        assert jnp.array_equal(new_state.player_positions[0], jnp.array([1, 0]))
        # Movement updates facing to the direction of travel.
        assert int(new_state.player_directions[0]) == Direction.UP

    def test_face_does_not_move(self, state_factory) -> None:
        """FACE_LEFT should change facing without moving."""
        state = state_factory(
            world_map=jnp.full((3, 3), BlockType.DIRT, dtype=jnp.int32),
            player_position=(1, 1),
            player_direction=int(Direction.UP),
        )
        new_state = move_player(state, Action.FACE_LEFT, 0)
        assert jnp.array_equal(new_state.player_positions[0], jnp.array([1, 1]))
        assert int(new_state.player_directions[0]) == Direction.LEFT

    @pytest.mark.parametrize(
        "action, expected_dir",
        [
            (Action.FACE_UP, Direction.UP),
            (Action.FACE_DOWN, Direction.DOWN),
            (Action.FACE_LEFT, Direction.LEFT),
            (Action.FACE_RIGHT, Direction.RIGHT),
        ],
    )
    def test_face_sets_direction_without_moving(
        self, state_factory, action: Action, expected_dir: Direction
    ) -> None:
        """FACE_* actions should snap facing to the target direction."""
        state = state_factory(
            world_map=jnp.full((3, 3), BlockType.DIRT, dtype=jnp.int32),
            player_position=(1, 1),
            player_direction=int(Direction.UP),
        )
        new_state = move_player(state, action, 0)
        assert jnp.array_equal(new_state.player_positions[0], jnp.array([1, 1]))
        assert int(new_state.player_directions[0]) == expected_dir

    def test_move_player_blocked_by_water(self, state_factory) -> None:
        """Player should not move into water."""
        world_map = jnp.array(
            [
                [BlockType.DIRT, BlockType.WATER],
                [BlockType.DIRT, BlockType.DIRT],
            ],
            dtype=jnp.int32,
        )
        state = state_factory(
            world_map=world_map,
            player_position=(0, 0),
        )
        # Water is at (1, 0). RIGHT moves x+1.
        new_state = move_player(state, Action.RIGHT, 0)
        assert jnp.array_equal(new_state.player_positions[0], jnp.array([0, 0]))

    def test_move_player_blocked_by_bounds(self, state_factory) -> None:
        """Player should not move out of bounds."""
        state = state_factory(
            world_map=jnp.full((3, 3), BlockType.DIRT, dtype=jnp.int32),
            player_position=(0, 0),
        )
        new_state = move_player(state, Action.LEFT, 0)
        assert jnp.array_equal(new_state.player_positions[0], jnp.array([0, 0]))

    def test_noop_does_not_change_position(self, state_factory) -> None:
        """NOOP should not change player position or direction."""
        state = state_factory(
            world_map=jnp.full((3, 3), BlockType.DIRT, dtype=jnp.int32),
            player_position=(1, 1),
            player_direction=int(Direction.DOWN),
        )
        new_state = move_player(state, Action.NOOP, 0)
        assert jnp.array_equal(new_state.player_positions[0], state.player_positions[0])
        assert int(new_state.player_directions[0]) == Direction.DOWN

    def test_is_game_over_before_max_timesteps(
        self,
        simple_state: EnvState,
    ) -> None:
        """Game should not be over before max timesteps."""
        params = EnvParams(max_timesteps=1000)
        assert not is_game_over(simple_state, params)

    def test_is_game_over_at_max_timesteps(
        self,
        simple_state: EnvState,
    ) -> None:
        """Game should be over at max timesteps."""
        params = EnvParams(max_timesteps=100)
        state = simple_state.replace(timestep=100)
        assert is_game_over(state, params)


class TestRenderer:
    """Tests for rendering."""

    def test_resource_textures_have_expected_colors(self) -> None:
        """Resource textures should be dominated by their base color.

        Ore textures now paint darker patches and occasional bright
        pixels on top of the base fill, so any given pixel might not
        equal the base color. The mode (most common color) still is.
        """
        textures = create_default_textures()

        expected = {
            int(BlockType.IRON): (180, 185, 200),
            int(BlockType.COPPER): (200, 120, 45),
            int(BlockType.COAL): (50, 50, 55),
        }
        for block_id, base_rgb in expected.items():
            tex = textures[block_id]
            # Collapse each pixel to an int tag, find the most common.
            pixels = tex[..., :3].reshape(-1, 3)
            as_int = (
                pixels[:, 0].astype(np.uint32) * 65536
                + pixels[:, 1].astype(np.uint32) * 256
                + pixels[:, 2].astype(np.uint32)
            )
            mode_tag = np.bincount(as_int).argmax()
            mode_rgb = (
                int(mode_tag >> 16) & 0xFF,
                int(mode_tag >> 8) & 0xFF,
                int(mode_tag) & 0xFF,
            )
            assert mode_rgb == base_rgb, (
                f"block {block_id}: dominant color {mode_rgb}, expected {base_rgb}"
            )


class TestEnvironment:
    """Tests for the gymnax environment interface."""

    def test_make_factoriax_env(self) -> None:
        """Environment factory should return env and params."""
        env, params = make_factoriax_env()
        assert env is not None
        assert params is not None

    def test_reset_returns_obs_and_state(self) -> None:
        """Reset should return observation and state."""
        env, params = make_factoriax_env()
        rng = random.PRNGKey(0)
        obs, state = env.reset_env(rng, params)

        assert obs is not None
        assert state is not None
        assert state.timestep == 0

    def test_step_returns_correct_tuple(self) -> None:
        """Step should return (obs, state, reward, done, info)."""
        env, params = make_factoriax_env()
        rng = random.PRNGKey(0)
        rng, reset_key, step_key = random.split(rng, 3)
        obs, state = env.reset_env(reset_key, params)

        obs, new_state, reward, done, info = env.step_env(
            step_key, state, Action.RIGHT, params
        )

        assert obs is not None
        assert new_state is not None
        assert isinstance(float(reward), float)
        assert isinstance(bool(done), bool)
        assert isinstance(info, dict)

    def test_step_increments_timestep(self) -> None:
        """Each step should increment the timestep."""
        env, params = make_factoriax_env()
        rng = random.PRNGKey(0)
        rng, reset_key, step_key = random.split(rng, 3)
        _, state = env.reset_env(reset_key, params)

        _, new_state, _, _, _ = env.step_env(
            step_key,
            state,
            Action.NOOP,
            params,
        )
        assert new_state.timestep == state.timestep + 1

    def test_action_space(self) -> None:
        """Action space should match the number of defined actions."""
        env, params = make_factoriax_env()
        action_space = env.action_space(params)
        assert action_space.n == NUM_ACTIONS

    def test_observation_space(self) -> None:
        """Observation space should match expected dimensions."""
        env, params = make_factoriax_env()
        obs_space = env.observation_space(params)
        expected_size = (
            NUM_SPATIAL_CHANNELS * params.map_width * params.map_height
            + NUM_PLAYER_SCALARS
        )
        assert obs_space.shape == (expected_size,)

    def test_jit_compilation(self) -> None:
        """Environment should be JIT-compilable."""
        env, params = make_factoriax_env()

        @jax.jit
        def run_episode(rng: jax.Array) -> jax.Array:
            rng, reset_key = random.split(rng)
            obs, state = env.reset_env(reset_key, params)
            rng, step_key = random.split(rng)
            obs, state, reward, done, _ = env.step_env(
                step_key, state, Action.RIGHT, params
            )
            return reward

        reward = run_episode(random.PRNGKey(0))
        assert reward is not None
