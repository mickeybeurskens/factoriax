"""Tests for the superficial observation builders.

The superficial profile keeps only outwardly-visible spatial channels
(block_type, machine_type, machine_direction) and drops the 9
facing-tile readouts from the scalar block. It pairs with both
``global_superficial`` and ``local_superficial``.

These tests pin down: shape, value range, channel-by-channel agreement
with the x_ray builders for shared channels, and JIT compatibility.
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
    Direction,
    Machine,
)
from factoriax.engine.observations import (
    _SUPERFICIAL_SPATIAL_CHANNEL_NAMES,
    _X_RAY_SPATIAL_CHANNEL_NAMES,
    NUM_PLAYER_SCALARS,
    NUM_SPATIAL_CHANNELS,
    _common_scalars,
    _facing_scalars,
    _superficial_scalars,
    global_superficial,
    global_x_ray,
    local_superficial,
    local_x_ray,
)
from factoriax.engine.state import EnvParams

_DEFAULT_PARAMS = EnvParams(
    max_timesteps=100,
)
_MAP_W: int = 8
_MAP_H: int = 8

_GLOBAL_SIZE = (
    NUM_SPATIAL_CHANNELS["superficial"] * _MAP_W * _MAP_H
    + NUM_PLAYER_SCALARS["superficial"]
)
_RADIUS = 3
_WINDOW = 2 * _RADIUS + 1
_LOCAL_SIZE = (
    NUM_SPATIAL_CHANNELS["superficial"] * _WINDOW**2 + NUM_PLAYER_SCALARS["superficial"]
)


# ---------------------------------------------------------------------------
# Profile-level invariants
# ---------------------------------------------------------------------------


class TestProfileShape:
    """The two channel/scalar manifests have the expected counts."""

    def test_spatial_channel_counts(self) -> None:
        """x_ray has 10 channels, superficial has 3."""
        assert NUM_SPATIAL_CHANNELS["x_ray"] == 10
        assert NUM_SPATIAL_CHANNELS["superficial"] == 3

    def test_player_scalar_counts(self) -> None:
        """x_ray has 81 scalars, superficial has 72 (drops 9 facing).

        4 pose + 34 affordability + 34 inventory. Both item-indexed blocks are
        NUM_ITEM_TYPES wide, so a scenario's recipe count does not change these.
        """
        assert NUM_PLAYER_SCALARS["x_ray"] == 81
        assert NUM_PLAYER_SCALARS["superficial"] == 72
        assert NUM_PLAYER_SCALARS["x_ray"] - NUM_PLAYER_SCALARS["superficial"] == 9

    def test_superficial_channels_are_subset_of_x_ray(self) -> None:
        """Every superficial channel name must also appear in x_ray."""
        for name in _SUPERFICIAL_SPATIAL_CHANNEL_NAMES:
            assert name in _X_RAY_SPATIAL_CHANNEL_NAMES


# ---------------------------------------------------------------------------
# global_superficial
# ---------------------------------------------------------------------------


class TestGlobalSuperficial:
    """Tests for global_superficial."""

    def test_shape(self, state_factory) -> None:
        """Output length is 3 * H * W + 63."""
        state = state_factory(
            world_map=jnp.ones((8, 8), dtype=jnp.int32) * int(BlockType.DIRT),
        )
        out = global_superficial(state, _DEFAULT_PARAMS, 0)
        assert out.shape == (_GLOBAL_SIZE,)

    def test_values_in_range(self, state_factory) -> None:
        """All values must lie in [0, 1]."""
        state = state_factory(
            world_map=jnp.ones((8, 8), dtype=jnp.int32) * int(BlockType.COAL),
            player_position=(3, 3),
            block_resources=jnp.full((8, 8), BLOCK_MAX_RESOURCES, dtype=jnp.int16),
        )
        out = np.array(global_superficial(state, _DEFAULT_PARAMS, 0))
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_block_machine_direction_match_x_ray(self, state_factory) -> None:
        """The 3 superficial channels equal their counterparts in x_ray."""
        world_map = jnp.ones((8, 8), dtype=jnp.int32) * int(BlockType.DIRT)
        world_map = world_map.at[5, 5].set(int(BlockType.COAL))

        machine_types = jnp.full((8, 8), int(Machine.NONE), dtype=jnp.int32)
        machine_types = machine_types.at[2, 3].set(int(Machine.MINER))

        machine_direction = jnp.zeros((8, 8), dtype=jnp.int8)
        machine_direction = machine_direction.at[2, 3].set(int(Direction.RIGHT))

        state = state_factory(
            world_map=world_map,
            machine_types=machine_types,
            machine_direction=machine_direction,
        )
        x_ray = np.array(global_x_ray(state, _DEFAULT_PARAMS, 0))
        super_obs = np.array(global_superficial(state, _DEFAULT_PARAMS, 0))

        tile_count = _MAP_W * _MAP_H
        for super_idx, name in enumerate(_SUPERFICIAL_SPATIAL_CHANNEL_NAMES):
            x_ray_idx = _X_RAY_SPATIAL_CHANNEL_NAMES.index(name)
            x_ray_channel = x_ray[x_ray_idx * tile_count : (x_ray_idx + 1) * tile_count]
            super_channel = super_obs[
                super_idx * tile_count : (super_idx + 1) * tile_count
            ]
            np.testing.assert_array_equal(
                super_channel,
                x_ray_channel,
                err_msg=f"channel '{name}' diverges between x_ray and superficial",
            )

    def test_scalar_tail_equals_common(self, state_factory) -> None:
        """Tail of global_superficial equals _common_scalars exactly."""
        from factoriax.engine.constants import ItemType

        inv = jnp.zeros((2, len(ItemType)), dtype=jnp.int32)
        inv = inv.at[0, ItemType.IRON_ORE].set(7)
        state = state_factory(
            world_map=jnp.ones((8, 8), dtype=jnp.int32) * int(BlockType.DIRT),
            player_position=(2, 4),
            timestep=21,
            player_inventory=inv,
        )
        out = np.array(global_superficial(state, _DEFAULT_PARAMS, 0))
        spatial_size = NUM_SPATIAL_CHANNELS["superficial"] * _MAP_W * _MAP_H
        tail = out[spatial_size:]
        expected = np.array(_common_scalars(state, _DEFAULT_PARAMS, 0))
        np.testing.assert_allclose(tail, expected)

    def test_jit_compatible(self, state_factory) -> None:
        """global_superficial survives jax.jit."""
        state = state_factory(
            world_map=jnp.ones((8, 8), dtype=jnp.int32) * int(BlockType.DIRT),
        )
        out = jax.jit(global_superficial)(state, _DEFAULT_PARAMS, 0)
        assert out.shape == (_GLOBAL_SIZE,)


# ---------------------------------------------------------------------------
# local_superficial
# ---------------------------------------------------------------------------


class TestLocalSuperficial:
    """Tests for local_superficial."""

    def test_shape(self, state_factory) -> None:
        """Output length is 3 * (2r+1)^2 + 63."""
        state = state_factory(
            world_map=jnp.ones((8, 8), dtype=jnp.int32) * int(BlockType.DIRT),
            player_position=(3, 3),
        )
        out = local_superficial(state, _DEFAULT_PARAMS, 0, radius=_RADIUS)
        assert out.shape == (_LOCAL_SIZE,)

    def test_values_in_range(self, state_factory) -> None:
        """All values must lie in [0, 1]."""
        state = state_factory(
            world_map=jnp.ones((8, 8), dtype=jnp.int32) * int(BlockType.COAL),
            player_position=(3, 3),
            block_resources=jnp.full((8, 8), BLOCK_MAX_RESOURCES, dtype=jnp.int16),
        )
        out = np.array(local_superficial(state, _DEFAULT_PARAMS, 0, radius=_RADIUS))
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_oob_padding_at_corner(self, state_factory) -> None:
        """Tiles outside the map fill with normalized OUT_OF_BOUNDS."""
        state = state_factory(
            world_map=jnp.ones((8, 8), dtype=jnp.int32) * int(BlockType.DIRT),
            player_position=(0, 0),
        )
        out = np.array(local_superficial(state, _DEFAULT_PARAMS, 0, radius=_RADIUS))
        block_flat = out[: _WINDOW**2]
        oob_val = float(BlockType.OUT_OF_BOUNDS) / float(max(BlockType))
        dirt_val = float(BlockType.DIRT) / float(max(BlockType))
        assert block_flat[0] == pytest.approx(oob_val)
        centre = _RADIUS * _WINDOW + _RADIUS
        assert block_flat[centre] == pytest.approx(dirt_val)

    def test_spatial_matches_local_x_ray_window(self, state_factory) -> None:
        """The 3 superficial channels equal the same windowed slice from local_x_ray."""
        world_map = jnp.ones((16, 16), dtype=jnp.int32) * int(BlockType.DIRT)
        world_map = world_map.at[7, 8].set(int(BlockType.IRON))
        machine_types = jnp.full((16, 16), int(Machine.NONE), dtype=jnp.int32)
        machine_types = machine_types.at[6, 6].set(int(Machine.ASSEMBLER))
        machine_direction = jnp.zeros((16, 16), dtype=jnp.int8)
        machine_direction = machine_direction.at[6, 6].set(int(Direction.UP))

        state = state_factory(
            world_map=world_map,
            player_position=(7, 7),
            machine_types=machine_types,
            machine_direction=machine_direction,
        )
        params = EnvParams(max_timesteps=100)
        x_ray = np.array(local_x_ray(state, params, 0, radius=_RADIUS))
        super_obs = np.array(local_superficial(state, params, 0, radius=_RADIUS))

        window_size = _WINDOW**2
        for super_idx, name in enumerate(_SUPERFICIAL_SPATIAL_CHANNEL_NAMES):
            x_ray_idx = _X_RAY_SPATIAL_CHANNEL_NAMES.index(name)
            x_ray_channel = x_ray[
                x_ray_idx * window_size : (x_ray_idx + 1) * window_size
            ]
            super_channel = super_obs[
                super_idx * window_size : (super_idx + 1) * window_size
            ]
            np.testing.assert_array_equal(
                super_channel,
                x_ray_channel,
                err_msg=(
                    f"channel '{name}' diverges between local_x_ray and "
                    f"local_superficial at the same window"
                ),
            )

    def test_scalar_tail_equals_common(self, state_factory) -> None:
        """Tail of local_superficial equals _common_scalars exactly."""
        state = state_factory(
            world_map=jnp.ones((8, 8), dtype=jnp.int32) * int(BlockType.DIRT),
            player_position=(3, 3),
            timestep=40,
        )
        out = np.array(local_superficial(state, _DEFAULT_PARAMS, 0, radius=_RADIUS))
        local_spatial = NUM_SPATIAL_CHANNELS["superficial"] * _WINDOW**2
        tail = out[local_spatial:]
        expected = np.array(_common_scalars(state, _DEFAULT_PARAMS, 0))
        np.testing.assert_allclose(tail, expected)

    def test_jit_compatible(self, state_factory) -> None:
        """local_superficial survives jax.jit when radius is bound via partial."""
        state = state_factory(
            world_map=jnp.ones((8, 8), dtype=jnp.int32) * int(BlockType.DIRT),
            player_position=(3, 3),
        )
        jit_fn = jax.jit(functools.partial(local_superficial, radius=_RADIUS))
        out = jit_fn(state, _DEFAULT_PARAMS, 0)
        assert out.shape == (_LOCAL_SIZE,)


# ---------------------------------------------------------------------------
# Scalar composition
# ---------------------------------------------------------------------------


class TestScalarComposition:
    """Compose-by-concat: x_ray = common ++ facing; superficial = common."""

    def test_x_ray_scalars_decompose_into_common_and_facing(
        self, state_factory
    ) -> None:
        """Concatenating _common_scalars and _facing_scalars must match x_ray."""
        from factoriax.engine.observations import _x_ray_scalars

        state = state_factory(
            world_map=jnp.ones((8, 8), dtype=jnp.int32) * int(BlockType.DIRT),
            player_position=(3, 3),
            timestep=12,
        )
        x_ray = np.array(_x_ray_scalars(state, _DEFAULT_PARAMS, 0))
        common = np.array(_common_scalars(state, _DEFAULT_PARAMS, 0))
        facing = np.array(_facing_scalars(state, 0))
        np.testing.assert_allclose(x_ray, np.concatenate([common, facing]))

    def test_superficial_scalars_equal_common(self, state_factory) -> None:
        """_superficial_scalars is exactly _common_scalars."""
        state = state_factory(
            world_map=jnp.ones((8, 8), dtype=jnp.int32) * int(BlockType.DIRT),
            player_position=(3, 3),
            timestep=12,
        )
        super_out = np.array(_superficial_scalars(state, _DEFAULT_PARAMS, 0))
        common = np.array(_common_scalars(state, _DEFAULT_PARAMS, 0))
        np.testing.assert_array_equal(super_out, common)
