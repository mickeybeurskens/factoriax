"""Tests for the per-player scalar block of the observation.

The scalars carry what the spatial grid cannot: the player position, the
facing, the inventory, and the afford vector. The afford vector is indexed by
item, not by recipe, so a new recipe must not shift what an agent reads.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from factoriax.engine.constants import (
    NUM_ITEM_TYPES,
    BlockType,
)
from factoriax.engine.observations import (
    NUM_PLAYER_SCALARS,
    NUM_SPATIAL_CHANNELS,
    _x_ray_scalars,
)
from tests.helpers.observations import (
    DEFAULT_PARAMS as _DEFAULT_PARAMS,
)
from tests.helpers.observations import (
    MAP_H as _MAP_H,
)
from tests.helpers.observations import (
    MAP_W as _MAP_W,
)
from tests.helpers.observations import (
    WINDOW as _WINDOW,
)

_GLOBAL_OBS_SIZE = (
    NUM_SPATIAL_CHANNELS["x_ray"] * _MAP_W * _MAP_H + NUM_PLAYER_SCALARS["x_ray"]
)
_LOCAL_OBS_SIZE = (
    NUM_SPATIAL_CHANNELS["x_ray"] * _WINDOW * _WINDOW + NUM_PLAYER_SCALARS["x_ray"]
)


class TestPlayerScalars:
    """Tests for the internal _x_ray_scalars helper."""

    def test_shape(self, state_factory) -> None:
        """Scalar vector has the expected length."""
        state = state_factory(
            world_map=jnp.ones((8, 8), dtype=jnp.int32) * int(BlockType.DIRT),
        )
        out = _x_ray_scalars(state, _DEFAULT_PARAMS, 0)
        expected = NUM_PLAYER_SCALARS["x_ray"]
        assert out.shape == (expected,)

    def test_values_in_range(self, state_factory) -> None:
        """All scalar values must lie in [0, 1]."""
        state = state_factory(
            world_map=jnp.ones((8, 8), dtype=jnp.int32) * int(BlockType.DIRT),
            player_position=(3, 5),
            timestep=50,
        )
        out = np.array(_x_ray_scalars(state, _DEFAULT_PARAMS, 0))
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_position_normalized(self, state_factory) -> None:
        """Normalized position components match pos / map_dim."""
        state = state_factory(
            world_map=jnp.ones((8, 8), dtype=jnp.int32) * int(BlockType.DIRT),
            player_position=(4, 6),
        )
        out = np.array(_x_ray_scalars(state, _DEFAULT_PARAMS, 0))
        assert out[0] == pytest.approx(4 / _MAP_W)
        assert out[1] == pytest.approx(6 / _MAP_H)

    def test_timestep_normalized(self, state_factory) -> None:
        """Timestep fraction matches timestep / max_timesteps."""
        state = state_factory(
            world_map=jnp.ones((8, 8), dtype=jnp.int32) * int(BlockType.DIRT),
            timestep=40,
        )
        out = np.array(_x_ray_scalars(state, _DEFAULT_PARAMS, 0))
        assert out[3] == pytest.approx(40 / _DEFAULT_PARAMS.max_timesteps)

    def test_player_idx_selects_correct_inventory(self, state_factory) -> None:
        """Different player_idx reads from the correct inventory row."""
        from factoriax.engine.constants import ItemType

        inv = jnp.zeros((2, NUM_ITEM_TYPES), dtype=jnp.int32)
        inv = inv.at[1, ItemType.IRON_ORE].set(5)
        state = state_factory(
            world_map=jnp.ones((8, 8), dtype=jnp.int32) * int(BlockType.DIRT),
            player_positions=jnp.array([[0, 0], [3, 3]], dtype=jnp.int32),
            player_inventory=inv,
        )
        scalars_p0 = np.array(_x_ray_scalars(state, _DEFAULT_PARAMS, 0))
        scalars_p1 = np.array(_x_ray_scalars(state, _DEFAULT_PARAMS, 1))
        # Player inventories differ, so the scalar vectors differ too.
        assert not np.allclose(scalars_p0, scalars_p1)


class TestAffordIsItemIndexed:
    """The afford block is indexed by item type, not by recipe id.

    Scenarios ship their own :class:`RecipeBook`, so recipe id ``3`` names a
    different item in each one. Indexing by item type keeps the block
    comparable across the curriculum, and keeps its width independent of how
    many recipes a scenario defines.
    """

    def test_width_is_item_count_not_recipe_count(self) -> None:
        """Scalar width tracks NUM_ITEM_TYPES, so a scenario's book cannot move it."""
        from factoriax.engine.constants import NUM_ITEM_TYPES

        # 4 pose scalars, then one afford and one inventory entry per item.
        assert NUM_PLAYER_SCALARS["superficial"] == 4 + 2 * NUM_ITEM_TYPES

    def test_books_of_different_size_give_equal_obs_width(self) -> None:
        """Two scenarios with different recipe counts observe the same width.

        easy_rocket ships 10 recipes and science_tiers 12. Before the afford
        block was item-indexed both were padded to the base book's 27, so the
        widths matched only by accident and the surplus entries repeated the
        last recipe.
        """
        from jax import random

        from factoriax.engine.envs.easy_rocket import easy_rocket
        from factoriax.engine.envs.science_tiers import science_tiers

        st_env, st_params = science_tiers()
        er_env, er_params = easy_rocket(obs=st_env.obs, obs_radius=st_env.obs_radius)
        assert (
            er_params.recipe_table.outputs.shape[0]
            != st_params.recipe_table.outputs.shape[0]
        )

        er_obs, _ = er_env.reset_env(random.PRNGKey(0), er_params)
        st_obs, _ = st_env.reset_env(random.PRNGKey(0), st_params)
        assert er_obs.shape == st_obs.shape
        assert er_obs.shape == er_env.observation_space(er_params).shape
        assert st_obs.shape == st_env.observation_space(st_params).shape

    def test_uncraftable_items_read_zero(self, state_factory) -> None:
        """Raw ore has no recipe, so its afford entry is always 0."""
        from factoriax.engine.constants import NUM_ITEM_TYPES, ItemType

        world_map = jnp.full((_MAP_H, _MAP_W), int(BlockType.DIRT), dtype=jnp.int32)
        state = state_factory(world_map=world_map)
        scalars = _x_ray_scalars(state, _DEFAULT_PARAMS, 0)
        afford = scalars[4 : 4 + NUM_ITEM_TYPES]

        assert afford[int(ItemType.IRON_ORE)] == 0.0
        assert afford[int(ItemType.EMPTY)] == 0.0
