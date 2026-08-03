"""Tests for the global x-ray observation.

``global_x_ray`` flattens every spatial channel over the whole map, then
appends the player scalars. These tests cover the shape, the value ranges,
the normalisation, and that the encoder survives ``jit`` and ``vmap``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from factoriax.engine.constants import (
    BlockType,
)
from factoriax.engine.observations import (
    NUM_PLAYER_SCALARS,
    NUM_SPATIAL_CHANNELS,
    global_x_ray,
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


class TestGlobalArray:
    """Tests for global_x_ray."""

    def test_shape(self, state_factory) -> None:
        """Output shape includes map, player scalars, and inventory."""
        state = state_factory(
            world_map=jnp.ones((8, 8), dtype=jnp.int32) * int(BlockType.DIRT),
        )
        out = global_x_ray(state, _DEFAULT_PARAMS, 0)
        assert out.shape == (_GLOBAL_OBS_SIZE,)

    def test_values_in_range(self, state_factory) -> None:
        """All values must lie in [0, 1]."""
        state = state_factory(
            world_map=jnp.array([[int(BlockType.DIRT)] * 8] * 8, dtype=jnp.int32),
            player_position=(2, 2),
        )
        out = np.array(global_x_ray(state, _DEFAULT_PARAMS, 0))
        assert out.min() >= 0.0, f"min={out.min()}"
        assert out.max() <= 1.0, f"max={out.max()}"

    def test_block_channel_correct(self, state_factory) -> None:
        """The first H*W elements (block channel) match the normalized map."""
        world_map = jnp.array([[int(BlockType.COAL)] * 8] * 8, dtype=jnp.int32)
        state = state_factory(world_map=world_map)
        out = np.array(global_x_ray(state, _DEFAULT_PARAMS, 0))
        tiles = _MAP_W * _MAP_H
        expected_val = float(BlockType.COAL) / float(max(BlockType))
        np.testing.assert_allclose(out[:tiles], expected_val)

    def test_player_idx_independent(self, state_factory) -> None:
        """Two players at different positions produce different observations."""
        state = state_factory(
            world_map=jnp.ones((8, 8), dtype=jnp.int32) * int(BlockType.DIRT),
            player_positions=jnp.array([[0, 0], [7, 7]], dtype=jnp.int32),
        )
        obs0 = np.array(global_x_ray(state, _DEFAULT_PARAMS, 0))
        obs1 = np.array(global_x_ray(state, _DEFAULT_PARAMS, 1))
        # Spatial channels are identical. Player scalars differ.
        spatial_size = NUM_SPATIAL_CHANNELS["x_ray"] * _MAP_W * _MAP_H
        np.testing.assert_array_equal(obs0[:spatial_size], obs1[:spatial_size])
        assert not np.allclose(obs0[spatial_size:], obs1[spatial_size:])

    def test_jit_compatible(self, state_factory) -> None:
        """global_x_ray survives jax.jit without error."""
        state = state_factory(
            world_map=jnp.ones((8, 8), dtype=jnp.int32) * int(BlockType.DIRT),
        )
        jit_fn = jax.jit(global_x_ray)
        out = jit_fn(state, _DEFAULT_PARAMS, 0)
        assert out.shape == (_GLOBAL_OBS_SIZE,)

    def test_vmap_over_player_idx(self, state_factory) -> None:
        """vmap over player_idx produces a (num_players, obs_size) array."""
        state = state_factory(
            world_map=jnp.ones((8, 8), dtype=jnp.int32) * int(BlockType.DIRT),
            player_positions=jnp.array([[0, 0], [4, 4]], dtype=jnp.int32),
        )
        vmap_fn = jax.vmap(global_x_ray, in_axes=(None, None, 0))
        all_obs = vmap_fn(state, _DEFAULT_PARAMS, jnp.arange(2))
        assert all_obs.shape == (2, _GLOBAL_OBS_SIZE)

    def test_matches_env_get_obs(self, state_factory) -> None:
        """global_x_ray for selected_player matches FactoriaxEnv.get_obs."""
        from factoriax.engine.envs.base import FactoriaxEnv

        state = state_factory(
            world_map=jnp.ones((8, 8), dtype=jnp.int32) * int(BlockType.DIRT),
            player_position=(3, 3),
            selected_player=0,
        )
        env = FactoriaxEnv()
        env_obs = np.array(env.get_obs(state, _DEFAULT_PARAMS))
        obs_fn = np.array(global_x_ray(state, _DEFAULT_PARAMS, state.selected_player))
        np.testing.assert_allclose(env_obs, obs_fn)
