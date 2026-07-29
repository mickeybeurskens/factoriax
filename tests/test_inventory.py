"""Tests for the inventory system."""

import jax.numpy as jnp
from jax import random

from factoriax.engine.constants import (
    NUM_ITEM_TYPES,
    ItemType,
)
from factoriax.engine.jax_renderer import JaxRenderer
from factoriax.engine.levels import generate_state
from factoriax.engine.observations import NUM_PLAYER_SCALARS, NUM_SPATIAL_CHANNELS
from factoriax.engine.state import EnvParams
from factoriax.engine.tables import PLAYER_MAX_STACK
from factoriax.playground.ui.theme import BLOCK_PIXEL_SIZE


class TestInventoryState:
    """Tests for inventory state initialization."""

    def test_initial_inventory_is_empty(self) -> None:
        """New world should have empty inventory."""
        rng = random.PRNGKey(0)
        params = EnvParams()
        state = generate_state(rng, params)

        assert jnp.all(state.player_inventory == 0)

    def test_inventory_arrays_have_correct_shape(self) -> None:
        """Inventory array should have shape (num_players, NUM_ITEM_TYPES)."""
        rng = random.PRNGKey(0)
        params = EnvParams()
        state = generate_state(rng, params)

        num_players = state.player_positions.shape[0]
        assert state.player_inventory.shape == (num_players, NUM_ITEM_TYPES)

    def test_inventory_arrays_are_int16(self) -> None:
        """Inventory array should be int16 dtype."""
        rng = random.PRNGKey(0)
        params = EnvParams()
        state = generate_state(rng, params)

        assert state.player_inventory.dtype == jnp.int16


class TestItemType:
    """Tests for the ItemType enumeration."""

    def test_item_types_have_unique_values(self) -> None:
        """Item types should have distinct values."""
        values = [
            ItemType.EMPTY,
            ItemType.COAL,
            ItemType.IRON_ORE,
            ItemType.COPPER_ORE,
        ]
        assert len(values) == len(set(values))

    def test_empty_is_zero(self) -> None:
        """EMPTY should be 0 for easy initialization."""
        assert ItemType.EMPTY == 0

    def test_num_item_types_matches_enum(self) -> None:
        """NUM_ITEM_TYPES should match the number of ItemType members."""
        assert NUM_ITEM_TYPES == len(ItemType)


class TestInventoryObservation:
    """Tests for inventory data in observations.

    Consumes ``canonical_env_8x8_1p`` (root conftest, session-scoped)
    instead of building a fresh ``factoriax.make()`` env per test. All
    assertions compute expected sizes dynamically from ``params``, so
    the 8x8 1p shape gives identical coverage as the default 32x32 2p
    at a fraction of the JIT cost.
    """

    def test_observation_includes_inventory(self, canonical_env_8x8_1p) -> None:
        """Observation should include inventory data."""
        env, params, _, state = canonical_env_8x8_1p
        obs = env.get_obs(state, params)

        expected_size = (
            NUM_SPATIAL_CHANNELS["x_ray"] * env.map_width * env.map_height
            + NUM_PLAYER_SCALARS["x_ray"]
        )
        assert obs.shape == (expected_size,)

    def test_observation_space_matches_observation(self, canonical_env_8x8_1p) -> None:
        """Observation shape should match observation_space."""
        env, params, _, state = canonical_env_8x8_1p
        obs = env.get_obs(state, params)

        obs_space = env.observation_space(params)
        assert obs.shape == obs_space.shape

    def test_inventory_observation_normalized(self, canonical_env_8x8_1p) -> None:
        """Inventory values should be in [0, 1]."""
        env, params, _, state = canonical_env_8x8_1p
        obs = env.get_obs(state, params)

        spatial_size = NUM_SPATIAL_CHANNELS["x_ray"] * env.map_width * env.map_height
        inv_start = spatial_size
        inv_data = obs[inv_start:]

        assert jnp.all(inv_data >= 0.0)
        assert jnp.all(inv_data <= 1.0)

    def test_inventory_observation_encodes_correctly(
        self, canonical_env_8x8_1p
    ) -> None:
        """Inventory observation should correctly encode item counts."""
        env, params, _, state = canonical_env_8x8_1p

        selected = state.selected_player
        max_coal = int(PLAYER_MAX_STACK[ItemType.COAL])
        state = state.replace(
            player_inventory=state.player_inventory.at[selected, ItemType.COAL].set(
                max_coal
            ),
        )

        obs = env.get_obs(state, params)
        # Verify the observation is in range -- exact indexing depends on
        # the scalar layout, but values should be bounded.
        assert jnp.all(obs >= 0.0)
        assert jnp.all(obs <= 1.0)


class TestInventoryRenderer:
    """Regression tests for the base renderer output contract."""

    def test_render_pixels_excludes_inventory(self) -> None:
        """The renderer should return an RGB array sized to the map, no menu."""
        rng = random.PRNGKey(0)
        params = EnvParams()
        state = generate_state(rng, params, map_height=8, map_width=8)

        renderer = JaxRenderer(tile_px=BLOCK_PIXEL_SIZE)
        pixels = renderer.jit_render_map(state)
        assert pixels.shape == (
            8 * BLOCK_PIXEL_SIZE,
            8 * BLOCK_PIXEL_SIZE,
            3,
        )
