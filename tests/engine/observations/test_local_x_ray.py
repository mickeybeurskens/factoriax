"""Tests for the local x-ray observation.

``local_x_ray`` crops a window of radius ``RADIUS`` around the player and
flattens it. These tests cover the shape, the padding outside the map, and
that the crop follows the player.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from factoriax.engine.constants import (
    BLOCK_MAX_RESOURCES,
    BlockType,
    Machine,
)
from factoriax.engine.observations import (
    _X_RAY_SPATIAL_CHANNEL_NAMES,
    NUM_PLAYER_SCALARS,
    NUM_SPATIAL_CHANNELS,
    local_x_ray,
)
from factoriax.engine.state import EnvParams

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_DEFAULT_PARAMS = EnvParams(
    max_timesteps=100,
)
_MAP_W: int = 8
_MAP_H: int = 8

_GLOBAL_OBS_SIZE = (
    NUM_SPATIAL_CHANNELS["x_ray"] * _MAP_W * _MAP_H + NUM_PLAYER_SCALARS["x_ray"]
)


# ---------------------------------------------------------------------------
# _x_ray_scalars
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# global_x_ray
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# local_x_ray
# ---------------------------------------------------------------------------

_RADIUS = 3
_WINDOW = 2 * _RADIUS + 1
_LOCAL_OBS_SIZE = (
    NUM_SPATIAL_CHANNELS["x_ray"] * _WINDOW**2 + NUM_PLAYER_SCALARS["x_ray"]
)


class TestLocalArray:
    """Tests for local_x_ray."""

    def test_shape_default_radius(self, state_factory) -> None:
        """Output shape is correct for radius=10 (default)."""
        radius = 10
        window = 2 * radius + 1
        expected = (
            NUM_SPATIAL_CHANNELS["x_ray"] * window**2 + NUM_PLAYER_SCALARS["x_ray"]
        )
        state = state_factory(
            world_map=jnp.ones((32, 32), dtype=jnp.int32) * int(BlockType.DIRT),
            player_position=(15, 15),
        )
        params = EnvParams()
        out = local_x_ray(state, params, 0)
        assert out.shape == (expected,)

    def test_shape_custom_radius(self, state_factory) -> None:
        """Output shape is correct for a custom radius."""
        state = state_factory(
            world_map=jnp.ones((8, 8), dtype=jnp.int32) * int(BlockType.DIRT),
            player_position=(3, 3),
        )
        out = local_x_ray(state, _DEFAULT_PARAMS, 0, radius=_RADIUS)
        assert out.shape == (_LOCAL_OBS_SIZE,)

    def test_values_in_range(self, state_factory) -> None:
        """All values must lie in [0, 1]."""
        state = state_factory(
            world_map=jnp.ones((8, 8), dtype=jnp.int32) * int(BlockType.COAL),
            player_position=(3, 3),
            block_resources=jnp.full((8, 8), BLOCK_MAX_RESOURCES, dtype=jnp.int16),
        )
        out = np.array(local_x_ray(state, _DEFAULT_PARAMS, 0, radius=_RADIUS))
        assert out.min() >= 0.0, f"min={out.min()}"
        assert out.max() <= 1.0, f"max={out.max()}"

    def test_oob_padding_at_corner(self, state_factory) -> None:
        """Tiles outside the map must be filled with normalized OUT_OF_BOUNDS."""
        # Player at corner (0, 0): the window extends outside the map.
        state = state_factory(
            world_map=jnp.ones((8, 8), dtype=jnp.int32) * int(BlockType.DIRT),
            player_position=(0, 0),
        )
        out = np.array(local_x_ray(state, _DEFAULT_PARAMS, 0, radius=_RADIUS))
        map_flat = out[: _WINDOW**2]
        oob_val = float(BlockType.OUT_OF_BOUNDS) / float(max(BlockType))
        dirt_val = float(BlockType.DIRT) / float(max(BlockType))
        # Top-left element of the window is fully outside -> OUT_OF_BOUNDS.
        assert map_flat[0] == pytest.approx(oob_val), (
            f"Expected OOB value {oob_val:.4f} at window corner, got {map_flat[0]:.4f}"
        )
        # Centre of the window (player tile) is DIRT.
        centre = _RADIUS * _WINDOW + _RADIUS
        assert map_flat[centre] == pytest.approx(dirt_val)

    def test_window_moves_with_player(self, state_factory) -> None:
        """Placing the player at different positions yields different map windows."""
        # Single COAL tile at (5, 5); all other tiles are DIRT.
        world_map = jnp.ones((8, 8), dtype=jnp.int32) * int(BlockType.DIRT)
        world_map = world_map.at[5, 5].set(int(BlockType.COAL))

        state_near = state_factory(world_map=world_map, player_position=(5, 5))
        state_far = state_factory(world_map=world_map, player_position=(0, 0))

        window_size = _WINDOW**2
        obs_near = np.array(local_x_ray(state_near, _DEFAULT_PARAMS, 0, radius=_RADIUS))
        obs_far = np.array(local_x_ray(state_far, _DEFAULT_PARAMS, 0, radius=_RADIUS))

        # The map segment of the near observation holds a COAL value.
        coal_val = float(BlockType.COAL) / float(max(BlockType))
        assert np.any(np.isclose(obs_near[:window_size], coal_val))
        # The far observation's map segment holds only DIRT and OOB.
        assert not np.any(np.isclose(obs_far[:window_size], coal_val))

    def test_machine_channel_present(self, state_factory) -> None:
        """A MINER at the player's position appears in the machine channel."""
        world_map = jnp.ones((8, 8), dtype=jnp.int32) * int(BlockType.DIRT)
        machine_types = jnp.zeros((8, 8), dtype=jnp.int32)
        machine_types = machine_types.at[3, 3].set(int(Machine.MINER))

        state = state_factory(
            world_map=world_map,
            player_position=(3, 3),
            machine_types=machine_types,
        )
        out = np.array(local_x_ray(state, _DEFAULT_PARAMS, 0, radius=_RADIUS))

        window_size = _WINDOW**2
        machine_flat = out[window_size : 2 * window_size]
        miner_val = float(Machine.MINER) / float(max(Machine))
        centre = _RADIUS * _WINDOW + _RADIUS
        assert machine_flat[centre] == pytest.approx(miner_val)

    def test_resource_channel_present(self, state_factory) -> None:
        """Block resources are reflected in the resource channel."""
        world_map = jnp.ones((8, 8), dtype=jnp.int32) * int(BlockType.COAL)
        resources = jnp.zeros((8, 8), dtype=jnp.int16)
        resources = resources.at[3, 3].set(BLOCK_MAX_RESOURCES)

        state = state_factory(
            world_map=world_map,
            player_position=(3, 3),
            block_resources=resources,
        )
        out = np.array(local_x_ray(state, _DEFAULT_PARAMS, 0, radius=_RADIUS))

        window_size = _WINDOW**2
        resource_flat = out[2 * window_size : 3 * window_size]
        centre = _RADIUS * _WINDOW + _RADIUS
        assert resource_flat[centre] == pytest.approx(1.0)  # fully normalized

    def test_jit_compatible(self, state_factory) -> None:
        """local_x_ray survives jax.jit when radius is bound via partial."""
        state = state_factory(
            world_map=jnp.ones((8, 8), dtype=jnp.int32) * int(BlockType.DIRT),
            player_position=(3, 3),
        )
        jit_fn = jax.jit(functools.partial(local_x_ray, radius=_RADIUS))
        out = jit_fn(state, _DEFAULT_PARAMS, 0)
        assert out.shape == (_LOCAL_OBS_SIZE,)

    def test_vmap_over_player_idx(self, state_factory) -> None:
        """vmap over player_idx yields (num_players, obs_size) output."""
        state = state_factory(
            world_map=jnp.ones((8, 8), dtype=jnp.int32) * int(BlockType.DIRT),
            player_positions=jnp.array([[1, 1], [6, 6]], dtype=jnp.int32),
        )
        vmap_fn = jax.vmap(
            functools.partial(local_x_ray, radius=_RADIUS),
            in_axes=(None, None, 0),
        )
        all_obs = vmap_fn(state, _DEFAULT_PARAMS, jnp.arange(2))
        assert all_obs.shape == (2, _LOCAL_OBS_SIZE)
        # The two players are at different positions so their windows differ.
        assert not np.allclose(np.array(all_obs[0]), np.array(all_obs[1]))

    def test_player_at_map_edge_no_error(self, state_factory) -> None:
        """Player at the maximum corner (W-1, H-1) must not raise."""
        state = state_factory(
            world_map=jnp.ones((8, 8), dtype=jnp.int32) * int(BlockType.DIRT),
            player_position=(7, 7),
        )
        out = local_x_ray(state, _DEFAULT_PARAMS, 0, radius=_RADIUS)
        assert out.shape == (_LOCAL_OBS_SIZE,)


# ---------------------------------------------------------------------------
# rgb
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Generic 3-slot machine projection
#
# Every machine's contents land on seven per-tile spatial channels:
# slot{0,1,2}_type, slot{0,1,2}_count, machine_direction. Combiners
# (assembler/furnace) write asm_in[0] → slot 0, asm_in[1] → slot 1,
# asm_out → slot 2. Buffer machines (miner/pallet/belt) write buf → slot 2
# and leave slots 0/1 at zero. The channels let an agent see "what is
# in each machine" from the global obs without having to navigate to
# and face each tile.
# ---------------------------------------------------------------------------


def _slice_channel(obs: jax.Array, params: EnvParams, name: str) -> np.ndarray:
    """Return the ``(H, W)`` float view of a named spatial channel."""
    idx = _X_RAY_SPATIAL_CHANNEL_NAMES.index(name)
    tile_count = _MAP_W * _MAP_H
    start = idx * tile_count
    end = start + tile_count
    return np.asarray(obs[start:end]).reshape(_MAP_H, _MAP_W)


# ---------------------------------------------------------------------------
# Affordability block indexing
# ---------------------------------------------------------------------------
