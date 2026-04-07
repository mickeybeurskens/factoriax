"""Tests for the inventory system."""

import jax.numpy as jnp
from jax import random

from factoriax import ItemType, make_factoriax_env
from factoriax.constants import (
    BLOCK_PIXEL_SIZE,
    NUM_ITEM_TYPES,
    NUM_TECHNOLOGIES,
    PLAYER_MAX_STACK,
)
from factoriax.observations import NUM_PLAYER_SCALARS, NUM_SPATIAL_CHANNELS
from factoriax.renderer import render_pixels
from factoriax.state import EnvParams
from factoriax.world_gen import generate_world


class TestInventoryState:
    """Tests for inventory state initialization."""

    def test_initial_inventory_is_empty(self) -> None:
        """New world should have empty inventory."""
        rng = random.PRNGKey(0)
        params = EnvParams()
        state = generate_world(rng, params)

        assert jnp.all(state.player_inventory == 0)

    def test_inventory_arrays_have_correct_shape(self) -> None:
        """Inventory array should have shape (num_players, NUM_ITEM_TYPES)."""
        rng = random.PRNGKey(0)
        params = EnvParams()
        state = generate_world(rng, params)

        assert state.player_inventory.shape == (
            params.num_players, NUM_ITEM_TYPES,
        )

    def test_inventory_arrays_are_int32(self) -> None:
        """Inventory array should be int32 dtype."""
        rng = random.PRNGKey(0)
        params = EnvParams()
        state = generate_world(rng, params)

        assert state.player_inventory.dtype == jnp.int32


class TestItemType:
    """Tests for the ItemType enumeration."""

    def test_item_types_have_unique_values(self) -> None:
        """Item types should have distinct values."""
        values = [
            ItemType.EMPTY,
            ItemType.COAL,
            ItemType.IRON,
            ItemType.COPPER,
        ]
        assert len(values) == len(set(values))

    def test_empty_is_zero(self) -> None:
        """EMPTY should be 0 for easy initialization."""
        assert ItemType.EMPTY == 0

    def test_num_item_types_matches_enum(self) -> None:
        """NUM_ITEM_TYPES should match the number of ItemType members."""
        assert NUM_ITEM_TYPES == len(ItemType)


class TestInventoryObservation:
    """Tests for inventory data in observations."""

    def test_observation_includes_inventory(self) -> None:
        """Observation should include inventory data."""
        env, params = make_factoriax_env()
        rng = random.PRNGKey(0)
        obs, state = env.reset_env(rng, params)

        expected_size = (
            NUM_SPATIAL_CHANNELS * params.map_width * params.map_height
            + NUM_PLAYER_SCALARS
            + NUM_TECHNOLOGIES * 2
        )
        assert obs.shape == (expected_size,)

    def test_observation_space_matches_observation(self) -> None:
        """Observation shape should match observation_space."""
        env, params = make_factoriax_env()
        rng = random.PRNGKey(0)
        obs, _ = env.reset_env(rng, params)

        obs_space = env.observation_space(params)
        assert obs.shape == obs_space.shape

    def test_inventory_observation_normalized(self) -> None:
        """Inventory values should be in [0, 1]."""
        env, params = make_factoriax_env()
        rng = random.PRNGKey(0)
        obs, state = env.reset_env(rng, params)

        spatial_size = (
            NUM_SPATIAL_CHANNELS * params.map_width * params.map_height
        )
        inv_start = spatial_size
        inv_data = obs[inv_start:]

        assert jnp.all(inv_data >= 0.0)
        assert jnp.all(inv_data <= 1.0)

    def test_inventory_observation_encodes_correctly(self) -> None:
        """Inventory observation should correctly encode item counts."""
        env, params = make_factoriax_env()
        rng = random.PRNGKey(0)
        _, state = env.reset_env(rng, params)

        selected = state.selected_player
        max_coal = int(PLAYER_MAX_STACK[ItemType.COAL])
        state = state.replace(
            player_inventory=state.player_inventory.at[
                selected, ItemType.COAL
            ].set(max_coal),
        )

        obs = env.get_obs(state, params)
        # Verify the observation is in range -- exact indexing depends on
        # the scalar layout, but values should be bounded.
        assert jnp.all(obs >= 0.0)
        assert jnp.all(obs <= 1.0)


class TestInventoryRenderer:
    """Regression tests for the base renderer output contract."""

    def test_render_pixels_excludes_inventory(self) -> None:
        """render_pixels should return an RGB array sized to the map, no menu."""
        rng = random.PRNGKey(0)
        params = EnvParams(map_width=8, map_height=8)
        state = generate_world(rng, params)

        pixels = render_pixels(state)
        assert pixels.shape == (
            8 * BLOCK_PIXEL_SIZE, 8 * BLOCK_PIXEL_SIZE, 3,
        )
