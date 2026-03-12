"""Tests for the FactoriaX environment."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from factoriax import Action, BlockType, EnvParams, EnvState, make_factoriax_env
from factoriax.constants import SOLID_BLOCKS
from factoriax.game_logic import (
    get_block_at,
    is_game_over,
    is_position_in_bounds,
    is_position_walkable,
    move_player,
)
from factoriax.renderer import create_default_textures, render_pixels
from factoriax.world_gen import generate_world


class TestConstants:
    """Tests for constants module."""

    def test_block_types_have_unique_values(self) -> None:
        """Block types should have distinct integer values."""
        values = [
            BlockType.INVALID,
            BlockType.OUT_OF_BOUNDS,
            BlockType.DIRT,
            BlockType.WATER,
            BlockType.IRON,
            BlockType.COPPER,
            BlockType.COAL,
        ]
        assert len(values) == len(set(values))

    def test_resource_block_types_exist(self) -> None:
        """Resource block types should exist with expected values."""
        assert BlockType.IRON == 4
        assert BlockType.COPPER == 5
        assert BlockType.COAL == 6

    def test_resource_blocks_are_walkable(self) -> None:
        """Resource blocks should not be in SOLID_BLOCKS."""
        solid_set = set(int(b) for b in SOLID_BLOCKS)
        assert int(BlockType.IRON) not in solid_set
        assert int(BlockType.COPPER) not in solid_set
        assert int(BlockType.COAL) not in solid_set

    def test_action_values(self) -> None:
        """Actions should be numbered 0-4."""
        assert Action.NOOP == 0
        assert Action.LEFT == 1
        assert Action.RIGHT == 2
        assert Action.UP == 3
        assert Action.DOWN == 4


class TestWorldGen:
    """Tests for world generation."""

    def test_generate_world_creates_valid_state(self) -> None:
        """Generated world should have valid state structure."""
        rng = random.PRNGKey(0)
        params = EnvParams()
        state = generate_world(rng, params)

        assert state.map.shape == (params.map_height, params.map_width)
        assert state.player_position.shape == (2,)
        assert state.timestep == 0

    def test_player_spawns_on_dirt(self) -> None:
        """Player should always spawn on a dirt tile."""
        rng = random.PRNGKey(0)
        params = EnvParams()
        state = generate_world(rng, params)

        px, py = state.player_position
        assert state.map[py, px] == BlockType.DIRT

    def test_world_gen_is_deterministic(self) -> None:
        """Same seed should produce same world."""
        params = EnvParams()
        state1 = generate_world(random.PRNGKey(42), params)
        state2 = generate_world(random.PRNGKey(42), params)

        assert jnp.array_equal(state1.map, state2.map)
        assert jnp.array_equal(state1.player_position, state2.player_position)

    def test_different_seeds_produce_different_worlds(self) -> None:
        """Different seeds should produce different worlds."""
        params = EnvParams()
        state1 = generate_world(random.PRNGKey(0), params)
        state2 = generate_world(random.PRNGKey(1), params)

        assert not jnp.array_equal(state1.map, state2.map)


class TestGameLogic:
    """Tests for game logic."""

    @pytest.fixture
    def simple_state(self) -> EnvState:
        """Create a simple state for testing."""
        world_map = jnp.array(
            [
                [BlockType.DIRT, BlockType.DIRT, BlockType.WATER],
                [BlockType.DIRT, BlockType.DIRT, BlockType.DIRT],
                [BlockType.WATER, BlockType.DIRT, BlockType.DIRT],
            ],
            dtype=jnp.int32,
        )
        return EnvState(
            map=world_map,
            player_position=jnp.array([1, 1], dtype=jnp.int32),
            player_direction=Action.DOWN,
            timestep=0,
        )

    def test_is_position_in_bounds(self) -> None:
        """Position bounds checking should work correctly."""
        assert is_position_in_bounds(jnp.array([0, 0]), 3, 3)
        assert is_position_in_bounds(jnp.array([2, 2]), 3, 3)
        assert not is_position_in_bounds(jnp.array([-1, 0]), 3, 3)
        assert not is_position_in_bounds(jnp.array([3, 0]), 3, 3)
        assert not is_position_in_bounds(jnp.array([0, -1]), 3, 3)
        assert not is_position_in_bounds(jnp.array([0, 3]), 3, 3)

    def test_get_block_at_returns_correct_block(self, simple_state: EnvState) -> None:
        """Should return correct block type at position."""
        assert get_block_at(simple_state, jnp.array([0, 0])) == BlockType.DIRT
        assert get_block_at(simple_state, jnp.array([2, 0])) == BlockType.WATER

    def test_get_block_at_out_of_bounds(self, simple_state: EnvState) -> None:
        """Out of bounds positions should return OUT_OF_BOUNDS."""
        assert get_block_at(simple_state, jnp.array([-1, 0])) == BlockType.OUT_OF_BOUNDS
        assert get_block_at(simple_state, jnp.array([0, 5])) == BlockType.OUT_OF_BOUNDS

    def test_is_position_walkable_dirt(self, simple_state: EnvState) -> None:
        """Dirt should be walkable."""
        assert is_position_walkable(simple_state, jnp.array([0, 0]))
        assert is_position_walkable(simple_state, jnp.array([1, 1]))

    def test_is_position_walkable_water(self, simple_state: EnvState) -> None:
        """Water should not be walkable."""
        assert not is_position_walkable(simple_state, jnp.array([2, 0]))
        assert not is_position_walkable(simple_state, jnp.array([0, 2]))

    def test_is_position_walkable_out_of_bounds(self, simple_state: EnvState) -> None:
        """Out of bounds should not be walkable."""
        assert not is_position_walkable(simple_state, jnp.array([-1, 0]))
        assert not is_position_walkable(simple_state, jnp.array([5, 5]))

    def test_move_player_left(self, simple_state: EnvState) -> None:
        """Player should move left on dirt."""
        new_state = move_player(simple_state, Action.LEFT)
        assert jnp.array_equal(new_state.player_position, jnp.array([0, 1]))
        assert new_state.player_direction == Action.LEFT

    def test_move_player_right(self, simple_state: EnvState) -> None:
        """Player should move right on dirt."""
        new_state = move_player(simple_state, Action.RIGHT)
        assert jnp.array_equal(new_state.player_position, jnp.array([2, 1]))
        assert new_state.player_direction == Action.RIGHT

    def test_move_player_up(self, simple_state: EnvState) -> None:
        """Player should move up on dirt."""
        new_state = move_player(simple_state, Action.UP)
        assert jnp.array_equal(new_state.player_position, jnp.array([1, 0]))
        assert new_state.player_direction == Action.UP

    def test_move_player_down(self, simple_state: EnvState) -> None:
        """Player should move down on dirt."""
        new_state = move_player(simple_state, Action.DOWN)
        assert jnp.array_equal(new_state.player_position, jnp.array([1, 2]))
        assert new_state.player_direction == Action.DOWN

    def test_move_player_blocked_by_water(self) -> None:
        """Player should not move into water."""
        world_map = jnp.array(
            [
                [BlockType.DIRT, BlockType.WATER],
                [BlockType.DIRT, BlockType.DIRT],
            ],
            dtype=jnp.int32,
        )
        state = EnvState(
            map=world_map,
            player_position=jnp.array([0, 0], dtype=jnp.int32),
            player_direction=Action.DOWN,
            timestep=0,
        )
        new_state = move_player(state, Action.RIGHT)
        assert jnp.array_equal(new_state.player_position, jnp.array([0, 0]))
        assert new_state.player_direction == Action.RIGHT

    def test_move_player_blocked_by_bounds(self, simple_state: EnvState) -> None:
        """Player should not move out of bounds."""
        state = simple_state.replace(player_position=jnp.array([0, 0], dtype=jnp.int32))
        new_state = move_player(state, Action.LEFT)
        assert jnp.array_equal(new_state.player_position, jnp.array([0, 0]))

    def test_noop_does_not_change_position(self, simple_state: EnvState) -> None:
        """NOOP should not change player position or direction."""
        new_state = move_player(simple_state, Action.NOOP)
        assert jnp.array_equal(new_state.player_position, simple_state.player_position)
        assert new_state.player_direction == simple_state.player_direction

    def test_is_game_over_before_max_timesteps(self, simple_state: EnvState) -> None:
        """Game should not be over before max timesteps."""
        params = EnvParams(max_timesteps=1000)
        assert not is_game_over(simple_state, params)

    def test_is_game_over_at_max_timesteps(self, simple_state: EnvState) -> None:
        """Game should be over at max timesteps."""
        params = EnvParams(max_timesteps=100)
        state = simple_state.replace(timestep=100)
        assert is_game_over(state, params)


class TestRenderer:
    """Tests for rendering."""

    def test_create_default_textures_creates_all_textures(self) -> None:
        """Default textures should include all block types."""
        textures = create_default_textures()
        assert BlockType.DIRT in textures
        assert BlockType.WATER in textures
        assert BlockType.IRON in textures
        assert BlockType.COPPER in textures
        assert BlockType.COAL in textures

    def test_textures_have_correct_shape(self) -> None:
        """Textures should be 16x16 RGBA."""
        textures = create_default_textures()
        for texture in textures.values():
            assert texture.shape == (16, 16, 4)
            assert texture.dtype == np.uint8

    def test_resource_textures_have_expected_colors(self) -> None:
        """Resource textures should have distinct recognizable colors."""
        textures = create_default_textures()

        iron = textures[int(BlockType.IRON)]
        assert iron[0, 0, 0] == 192  # Silver/gray RGB
        assert iron[0, 0, 1] == 192
        assert iron[0, 0, 2] == 192

        copper = textures[int(BlockType.COPPER)]
        assert copper[0, 0, 0] == 184  # Orange-brown RGB
        assert copper[0, 0, 1] == 115
        assert copper[0, 0, 2] == 51

        coal = textures[int(BlockType.COAL)]
        assert coal[0, 0, 0] == 54  # Dark gray RGB
        assert coal[0, 0, 1] == 54
        assert coal[0, 0, 2] == 54

    def test_render_pixels_returns_correct_shape(self) -> None:
        """Rendered image should have correct dimensions."""
        rng = random.PRNGKey(0)
        params = EnvParams(map_width=8, map_height=8)
        state = generate_world(rng, params)

        pixels = render_pixels(state, block_pixel_size=16)
        assert pixels.shape == (8 * 16, 8 * 16, 3)

    def test_render_pixels_returns_rgb(self) -> None:
        """Rendered image should be RGB (not RGBA)."""
        rng = random.PRNGKey(0)
        params = EnvParams(map_width=4, map_height=4)
        state = generate_world(rng, params)

        pixels = render_pixels(state)
        assert pixels.shape[2] == 3


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

        _, new_state, _, _, _ = env.step_env(step_key, state, Action.NOOP, params)
        assert new_state.timestep == state.timestep + 1

    def test_action_space(self) -> None:
        """Action space should be Discrete(5)."""
        env, params = make_factoriax_env()
        action_space = env.action_space(params)
        assert action_space.n == 5

    def test_observation_space(self) -> None:
        """Observation space should match expected dimensions."""
        env, params = make_factoriax_env()
        obs_space = env.observation_space(params)
        expected_size = params.map_width * params.map_height + 4
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
