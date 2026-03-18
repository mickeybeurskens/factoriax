"""Tests for factoriax.observations: global_array, local_array, and rgb.

Each function is tested for correct output shape, value range, JAX
compatibility, and behavioural correctness (e.g. the local window actually
moves with the player, border tiles are padded correctly).
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from factoriax.constants import (
    BLOCK_MAX_RESOURCES,
    NUM_INVENTORY_SLOTS,
    BlockType,
    MachineType,
)
from factoriax.observations import _player_scalars, global_array, local_array, rgb
from factoriax.state import EnvParams

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_DEFAULT_PARAMS = EnvParams(
    max_timesteps=100,
    map_width=8,
    map_height=8,
    num_players=2,
)

_GLOBAL_OBS_SIZE = (
    _DEFAULT_PARAMS.map_width * _DEFAULT_PARAMS.map_height + 4 + 2 * NUM_INVENTORY_SLOTS
)


# ---------------------------------------------------------------------------
# _player_scalars
# ---------------------------------------------------------------------------


class TestPlayerScalars:
    """Tests for the internal _player_scalars helper."""

    def test_shape(self, state_factory) -> None:
        """Scalar vector has the expected length."""
        state = state_factory(
            world_map=jnp.ones((8, 8), dtype=jnp.int32) * int(BlockType.DIRT),
        )
        out = _player_scalars(state, _DEFAULT_PARAMS, 0)
        assert out.shape == (4 + 2 * NUM_INVENTORY_SLOTS,)

    def test_values_in_range(self, state_factory) -> None:
        """All scalar values must lie in [0, 1]."""
        state = state_factory(
            world_map=jnp.ones((8, 8), dtype=jnp.int32) * int(BlockType.DIRT),
            player_position=(3, 5),
            timestep=50,
        )
        out = np.array(_player_scalars(state, _DEFAULT_PARAMS, 0))
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_position_normalized(self, state_factory) -> None:
        """Normalized position components match pos / map_dim."""
        state = state_factory(
            world_map=jnp.ones((8, 8), dtype=jnp.int32) * int(BlockType.DIRT),
            player_position=(4, 6),
        )
        out = np.array(_player_scalars(state, _DEFAULT_PARAMS, 0))
        assert out[0] == pytest.approx(4 / _DEFAULT_PARAMS.map_width)
        assert out[1] == pytest.approx(6 / _DEFAULT_PARAMS.map_height)

    def test_timestep_normalized(self, state_factory) -> None:
        """Timestep fraction matches timestep / max_timesteps."""
        state = state_factory(
            world_map=jnp.ones((8, 8), dtype=jnp.int32) * int(BlockType.DIRT),
            timestep=40,
        )
        out = np.array(_player_scalars(state, _DEFAULT_PARAMS, 0))
        assert out[3] == pytest.approx(40 / _DEFAULT_PARAMS.max_timesteps)

    def test_player_idx_selects_correct_inventory(self, state_factory) -> None:
        """Different player_idx reads from the correct inventory row."""
        inv_items = jnp.zeros((2, NUM_INVENTORY_SLOTS), dtype=jnp.int32)
        inv_items = inv_items.at[1, 0].set(2)  # player 1 has item 2 in slot 0
        state = state_factory(
            world_map=jnp.ones((8, 8), dtype=jnp.int32) * int(BlockType.DIRT),
            player_positions=jnp.array([[0, 0], [3, 3]], dtype=jnp.int32),
            num_players=2,
            inventory_items=inv_items,
        )
        scalars_p0 = np.array(_player_scalars(state, _DEFAULT_PARAMS, 0))
        scalars_p1 = np.array(_player_scalars(state, _DEFAULT_PARAMS, 1))
        # Inventory items start at index 4
        assert scalars_p0[4] == pytest.approx(0.0)
        assert scalars_p1[4] > 0.0


# ---------------------------------------------------------------------------
# global_array
# ---------------------------------------------------------------------------


class TestGlobalArray:
    """Tests for global_array."""

    def test_shape(self, state_factory) -> None:
        """Output shape matches map_h * map_w + 4 + 2 * NUM_INVENTORY_SLOTS."""
        state = state_factory(
            world_map=jnp.ones((8, 8), dtype=jnp.int32) * int(BlockType.DIRT),
        )
        out = global_array(state, _DEFAULT_PARAMS, 0)
        assert out.shape == (_GLOBAL_OBS_SIZE,)

    def test_values_in_range(self, state_factory) -> None:
        """All values must lie in [0, 1]."""
        state = state_factory(
            world_map=jnp.array([[int(BlockType.DIRT)] * 8] * 8, dtype=jnp.int32),
            player_position=(2, 2),
        )
        out = np.array(global_array(state, _DEFAULT_PARAMS, 0))
        assert out.min() >= 0.0, f"min={out.min()}"
        assert out.max() <= 1.0, f"max={out.max()}"

    def test_map_segment_correct(self, state_factory) -> None:
        """The first map_h * map_w elements must match the normalized map."""
        world_map = jnp.array([[int(BlockType.COAL)] * 8] * 8, dtype=jnp.int32)
        state = state_factory(world_map=world_map)
        out = np.array(global_array(state, _DEFAULT_PARAMS, 0))
        map_size = _DEFAULT_PARAMS.map_width * _DEFAULT_PARAMS.map_height
        expected_val = float(BlockType.COAL) / float(BlockType.COAL)  # = 1.0
        np.testing.assert_allclose(out[:map_size], expected_val)

    def test_player_idx_independent(self, state_factory) -> None:
        """Two players at different positions produce different observations."""
        state = state_factory(
            world_map=jnp.ones((8, 8), dtype=jnp.int32) * int(BlockType.DIRT),
            player_positions=jnp.array([[0, 0], [7, 7]], dtype=jnp.int32),
            num_players=2,
        )
        obs0 = np.array(global_array(state, _DEFAULT_PARAMS, 0))
        obs1 = np.array(global_array(state, _DEFAULT_PARAMS, 1))
        # Map segment is identical; player scalars differ.
        map_size = _DEFAULT_PARAMS.map_width * _DEFAULT_PARAMS.map_height
        np.testing.assert_array_equal(obs0[:map_size], obs1[:map_size])
        assert not np.allclose(obs0[map_size:], obs1[map_size:])

    def test_jit_compatible(self, state_factory) -> None:
        """global_array survives jax.jit without error."""
        state = state_factory(
            world_map=jnp.ones((8, 8), dtype=jnp.int32) * int(BlockType.DIRT),
        )
        jit_fn = jax.jit(global_array)
        out = jit_fn(state, _DEFAULT_PARAMS, 0)
        assert out.shape == (_GLOBAL_OBS_SIZE,)

    def test_vmap_over_player_idx(self, state_factory) -> None:
        """vmap over player_idx produces a (num_players, obs_size) array."""
        state = state_factory(
            world_map=jnp.ones((8, 8), dtype=jnp.int32) * int(BlockType.DIRT),
            player_positions=jnp.array([[0, 0], [4, 4]], dtype=jnp.int32),
            num_players=2,
        )
        vmap_fn = jax.vmap(global_array, in_axes=(None, None, 0))
        all_obs = vmap_fn(state, _DEFAULT_PARAMS, jnp.arange(2))
        assert all_obs.shape == (2, _GLOBAL_OBS_SIZE)

    def test_matches_env_get_obs(self, state_factory) -> None:
        """global_array for selected_player matches FactoriaXEnv.get_obs."""
        from factoriax import FactoriaXEnv

        state = state_factory(
            world_map=jnp.ones((8, 8), dtype=jnp.int32) * int(BlockType.DIRT),
            player_position=(3, 3),
            selected_player=0,
        )
        env = FactoriaXEnv()
        env_obs = np.array(env.get_obs(state, _DEFAULT_PARAMS))
        obs_fn = np.array(global_array(state, _DEFAULT_PARAMS, state.selected_player))
        np.testing.assert_allclose(env_obs, obs_fn)


# ---------------------------------------------------------------------------
# local_array
# ---------------------------------------------------------------------------

_RADIUS = 3
_WINDOW = 2 * _RADIUS + 1
_LOCAL_OBS_SIZE = 3 * _WINDOW**2 + 4 + 2 * NUM_INVENTORY_SLOTS


class TestLocalArray:
    """Tests for local_array."""

    def test_shape_default_radius(self, state_factory) -> None:
        """Output shape is correct for radius=10 (default)."""
        radius = 10
        window = 2 * radius + 1
        expected = 3 * window**2 + 4 + 2 * NUM_INVENTORY_SLOTS
        state = state_factory(
            world_map=jnp.ones((32, 32), dtype=jnp.int32) * int(BlockType.DIRT),
            player_position=(15, 15),
        )
        params = EnvParams(map_width=32, map_height=32, num_players=1)
        out = local_array(state, params, 0)
        assert out.shape == (expected,)

    def test_shape_custom_radius(self, state_factory) -> None:
        """Output shape is correct for a custom radius."""
        state = state_factory(
            world_map=jnp.ones((8, 8), dtype=jnp.int32) * int(BlockType.DIRT),
            player_position=(3, 3),
        )
        out = local_array(state, _DEFAULT_PARAMS, 0, radius=_RADIUS)
        assert out.shape == (_LOCAL_OBS_SIZE,)

    def test_values_in_range(self, state_factory) -> None:
        """All values must lie in [0, 1]."""
        state = state_factory(
            world_map=jnp.ones((8, 8), dtype=jnp.int32) * int(BlockType.COAL),
            player_position=(3, 3),
            block_resources=jnp.full((8, 8), BLOCK_MAX_RESOURCES, dtype=jnp.int16),
        )
        out = np.array(local_array(state, _DEFAULT_PARAMS, 0, radius=_RADIUS))
        assert out.min() >= 0.0, f"min={out.min()}"
        assert out.max() <= 1.0, f"max={out.max()}"

    def test_oob_padding_at_corner(self, state_factory) -> None:
        """Tiles outside the map must be filled with normalized OUT_OF_BOUNDS."""
        # Player at corner (0, 0): the window extends outside the map.
        state = state_factory(
            world_map=jnp.ones((8, 8), dtype=jnp.int32) * int(BlockType.DIRT),
            player_position=(0, 0),
        )
        out = np.array(local_array(state, _DEFAULT_PARAMS, 0, radius=_RADIUS))
        map_flat = out[: _WINDOW**2]
        oob_val = float(BlockType.OUT_OF_BOUNDS) / float(BlockType.COAL)
        dirt_val = float(BlockType.DIRT) / float(BlockType.COAL)
        # Top-left element of the window is fully outside → OUT_OF_BOUNDS.
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
        obs_near = np.array(local_array(state_near, _DEFAULT_PARAMS, 0, radius=_RADIUS))
        obs_far = np.array(local_array(state_far, _DEFAULT_PARAMS, 0, radius=_RADIUS))

        # The map segment of the near observation should contain a COAL value.
        coal_val = float(BlockType.COAL) / float(BlockType.COAL)
        assert coal_val in obs_near[:window_size]
        # The far observation's map segment should contain only DIRT and OOB.
        assert coal_val not in obs_far[:window_size]

    def test_machine_channel_present(self, state_factory) -> None:
        """A MINER at the player's position appears in the machine channel."""
        world_map = jnp.ones((8, 8), dtype=jnp.int32) * int(BlockType.DIRT)
        machine_types = jnp.zeros((8, 8), dtype=jnp.int32)
        machine_types = machine_types.at[3, 3].set(int(MachineType.MINER))

        state = state_factory(
            world_map=world_map,
            player_position=(3, 3),
            machine_types=machine_types,
        )
        out = np.array(local_array(state, _DEFAULT_PARAMS, 0, radius=_RADIUS))

        window_size = _WINDOW**2
        machine_flat = out[window_size : 2 * window_size]
        miner_val = float(MachineType.MINER) / float(max(MachineType))
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
        out = np.array(local_array(state, _DEFAULT_PARAMS, 0, radius=_RADIUS))

        window_size = _WINDOW**2
        resource_flat = out[2 * window_size : 3 * window_size]
        centre = _RADIUS * _WINDOW + _RADIUS
        assert resource_flat[centre] == pytest.approx(1.0)  # fully normalized

    def test_jit_compatible(self, state_factory) -> None:
        """local_array survives jax.jit when radius is bound via partial."""
        state = state_factory(
            world_map=jnp.ones((8, 8), dtype=jnp.int32) * int(BlockType.DIRT),
            player_position=(3, 3),
        )
        jit_fn = jax.jit(functools.partial(local_array, radius=_RADIUS))
        out = jit_fn(state, _DEFAULT_PARAMS, 0)
        assert out.shape == (_LOCAL_OBS_SIZE,)

    def test_vmap_over_player_idx(self, state_factory) -> None:
        """vmap over player_idx yields (num_players, obs_size) output."""
        state = state_factory(
            world_map=jnp.ones((8, 8), dtype=jnp.int32) * int(BlockType.DIRT),
            player_positions=jnp.array([[1, 1], [6, 6]], dtype=jnp.int32),
            num_players=2,
        )
        vmap_fn = jax.vmap(
            functools.partial(local_array, radius=_RADIUS),
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
        out = local_array(state, _DEFAULT_PARAMS, 0, radius=_RADIUS)
        assert out.shape == (_LOCAL_OBS_SIZE,)


# ---------------------------------------------------------------------------
# rgb
# ---------------------------------------------------------------------------


class TestRgb:
    """Tests for the rgb observation function."""

    def test_shape_and_dtype(self, state_factory) -> None:
        """Output must be uint8 RGB with correct pixel dimensions."""
        bps = 4
        state = state_factory(
            world_map=jnp.ones((8, 8), dtype=jnp.int32) * int(BlockType.DIRT),
        )
        out = rgb(state, block_pixel_size=bps)
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.uint8
        assert out.ndim == 3
        assert out.shape == (8 * bps, 8 * bps, 3)

    def test_returns_numpy(self, state_factory) -> None:
        """rgb must return a numpy array, not a JAX array."""
        state = state_factory(
            world_map=jnp.ones((8, 8), dtype=jnp.int32) * int(BlockType.DIRT),
        )
        out = rgb(state, block_pixel_size=4)
        assert isinstance(out, np.ndarray)

    def test_different_maps_produce_different_images(self, state_factory) -> None:
        """A DIRT map and a WATER map must render visually differently."""
        bps = 4
        state_dirt = state_factory(
            world_map=jnp.ones((4, 4), dtype=jnp.int32) * int(BlockType.DIRT),
        )
        state_water = state_factory(
            world_map=jnp.ones((4, 4), dtype=jnp.int32) * int(BlockType.WATER),
        )
        img_dirt = rgb(state_dirt, block_pixel_size=bps)
        img_water = rgb(state_water, block_pixel_size=bps)
        assert not np.array_equal(img_dirt, img_water)

    @pytest.mark.parametrize("bps", [4, 8, 16])
    def test_block_pixel_size_scales_output(self, state_factory, bps: int) -> None:
        """Output dimensions must scale linearly with block_pixel_size.

        Args:
            bps: Block pixel size to test.
        """
        state = state_factory(
            world_map=jnp.ones((4, 4), dtype=jnp.int32) * int(BlockType.DIRT),
        )
        out = rgb(state, block_pixel_size=bps)
        assert out.shape == (4 * bps, 4 * bps, 3)
