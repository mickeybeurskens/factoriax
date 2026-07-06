"""Tests for factoriax.engine.observations: global_x_ray, local_x_ray, and rgb.

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

from factoriax.engine.constants import (
    BLOCK_MAX_RESOURCES,
    NUM_ITEM_TYPES,
    BlockType,
    Machine,
)
from factoriax.engine.observations import (
    _X_RAY_SPATIAL_CHANNEL_NAMES,
    NUM_PLAYER_SCALARS,
    NUM_SPATIAL_CHANNELS,
    _x_ray_scalars,
    global_x_ray,
    local_x_ray,
    rgb,
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
    NUM_SPATIAL_CHANNELS["x_ray"]
    * _MAP_W
    * _MAP_H
    + NUM_PLAYER_SCALARS["x_ray"]
)


# ---------------------------------------------------------------------------
# _x_ray_scalars
# ---------------------------------------------------------------------------


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
        # Player inventories differ, so the scalar vectors should differ.
        assert not np.allclose(scalars_p0, scalars_p1)


# ---------------------------------------------------------------------------
# global_x_ray
# ---------------------------------------------------------------------------


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
        # Spatial channels are identical; player scalars differ.
        spatial_size = (
            NUM_SPATIAL_CHANNELS["x_ray"]
            * _MAP_W
            * _MAP_H
        )
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

        # The map segment of the near observation should contain a COAL value.
        coal_val = float(BlockType.COAL) / float(max(BlockType))
        assert np.any(np.isclose(obs_near[:window_size], coal_val))
        # The far observation's map segment should contain only DIRT and OOB.
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


class TestSlotProjection:
    """Observation exposes every machine's three slots (A input, B input, output)."""

    def test_num_spatial_channels(self) -> None:
        """Grew from 4 to 10: base 3 + 6 slot + direction."""
        assert NUM_SPATIAL_CHANNELS["x_ray"] == 10
        for name in (
            "slot0_type",
            "slot0_count",
            "slot1_type",
            "slot1_count",
            "slot2_type",
            "slot2_count",
            "machine_direction",
        ):
            assert name in _X_RAY_SPATIAL_CHANNEL_NAMES

    def test_assembler_slots_show_asm_in_and_asm_out(
        self,
        state_factory,
    ) -> None:
        """Assembler at (2, 3) with iron+tin inputs and a frame output
        must surface those on slot 0, slot 1, slot 2 channels."""
        from factoriax.engine.constants import ItemType

        shape = (_MAP_H, _MAP_W)
        world_map = jnp.full(shape, int(BlockType.DIRT), dtype=jnp.int32)
        mt = jnp.full(shape, int(Machine.NONE), dtype=jnp.int32)
        mt = mt.at[3, 2].set(int(Machine.ASSEMBLER))
        ait = jnp.zeros((*shape, 2), dtype=jnp.int32)
        ait = ait.at[3, 2, 0].set(int(ItemType.IRON_PLATE))
        ait = ait.at[3, 2, 1].set(int(ItemType.TIN_PLATE))
        aic = jnp.zeros((*shape, 2), dtype=jnp.int32)
        aic = aic.at[3, 2, 0].set(1)
        aic = aic.at[3, 2, 1].set(1)
        aot = jnp.zeros(shape, dtype=jnp.int32).at[3, 2].set(int(ItemType.FRAME))
        aoc = jnp.zeros(shape, dtype=jnp.int32).at[3, 2].set(1)

        state = state_factory(
            world_map=world_map,
            machine_types=mt,
            asm_in_type=ait,
            asm_in_count=aic,
            asm_out_type=aot,
            asm_out_count=aoc,
        )
        obs = global_x_ray(state, _DEFAULT_PARAMS, 0)

        s0t = _slice_channel(obs, _DEFAULT_PARAMS, "slot0_type")
        s1t = _slice_channel(obs, _DEFAULT_PARAMS, "slot1_type")
        s2t = _slice_channel(obs, _DEFAULT_PARAMS, "slot2_type")
        s0c = _slice_channel(obs, _DEFAULT_PARAMS, "slot0_count")
        s1c = _slice_channel(obs, _DEFAULT_PARAMS, "slot1_count")
        s2c = _slice_channel(obs, _DEFAULT_PARAMS, "slot2_count")

        # At the assembler tile, the slot channels are normalised by
        # NUM_ITEM_TYPES for types / by _SLOT_COUNT_NORM for counts.
        # We only check "this channel is non-zero" here — the exact
        # normalisation constant can change without breaking the
        # agent's decoder.
        assert s0t[3, 2] > 0
        assert s1t[3, 2] > 0
        assert s2t[3, 2] > 0
        assert s0c[3, 2] > 0
        assert s1c[3, 2] > 0
        assert s2c[3, 2] > 0
        # Tiles without a machine stay zero.
        assert s0t[0, 0] == 0 and s1t[0, 0] == 0 and s2t[0, 0] == 0

    def test_pallet_puts_buffer_on_slot2(self, state_factory) -> None:
        """A pallet stores items in ``ent_buf``. That must land on
        slot 2 and leave slots 0 and 1 at zero."""
        from factoriax.engine.constants import ItemType

        shape = (_MAP_H, _MAP_W)
        world_map = jnp.full(shape, int(BlockType.DIRT), dtype=jnp.int32)
        mt = jnp.full(shape, int(Machine.NONE), dtype=jnp.int32)
        mt = mt.at[1, 1].set(int(Machine.PALLET))
        bt = jnp.zeros(shape, dtype=jnp.int32).at[1, 1].set(int(ItemType.COAL))
        bc = jnp.zeros(shape, dtype=jnp.int32).at[1, 1].set(50)

        state = state_factory(
            world_map=world_map,
            machine_types=mt,
            buffer_type=bt,
            buffer_count=bc,
        )
        obs = global_x_ray(state, _DEFAULT_PARAMS, 0)

        s0t = _slice_channel(obs, _DEFAULT_PARAMS, "slot0_type")
        s1t = _slice_channel(obs, _DEFAULT_PARAMS, "slot1_type")
        s2t = _slice_channel(obs, _DEFAULT_PARAMS, "slot2_type")
        s2c = _slice_channel(obs, _DEFAULT_PARAMS, "slot2_count")

        assert s0t[1, 1] == 0
        assert s1t[1, 1] == 0
        assert s2t[1, 1] > 0  # coal present in slot 2
        assert s2c[1, 1] > 0

    def test_empty_machine_has_zero_slots(self, state_factory) -> None:
        """A freshly placed assembler with no inputs/output shows 0 in
        every slot channel."""
        shape = (_MAP_H, _MAP_W)
        world_map = jnp.full(shape, int(BlockType.DIRT), dtype=jnp.int32)
        mt = jnp.full(shape, int(Machine.NONE), dtype=jnp.int32)
        mt = mt.at[2, 2].set(int(Machine.ASSEMBLER))

        state = state_factory(world_map=world_map, machine_types=mt)
        obs = global_x_ray(state, _DEFAULT_PARAMS, 0)

        for ch in ("slot0_type", "slot1_type", "slot2_type"):
            g = _slice_channel(obs, _DEFAULT_PARAMS, ch)
            assert g[2, 2] == 0
        for ch in ("slot0_count", "slot1_count", "slot2_count"):
            g = _slice_channel(obs, _DEFAULT_PARAMS, ch)
            assert g[2, 2] == 0


class TestLocalGlobalEquivalence:
    """Local window must match global obs tile-for-tile at the player.

    Invariant: ``local_x_ray(state, params, p, radius=R)`` is a pure
    spatial crop of ``global_x_ray(state, params, p)``. For every
    (y, x) inside the window, the per-channel value in the local obs
    must equal the corresponding tile in the global obs. The player
    scalars + research tail appended after the spatial block must be
    identical between the two. If this test fails, either ``local_x_ray``
    and ``global_x_ray`` have drifted in channel ordering / normalisation,
    or the window indexing is off — both of which silently corrupt
    training without obvious symptoms.
    """

    @pytest.mark.parametrize("radius", [1, 3, 5])
    def test_spatial_channels_match_global_window(
        self,
        state_factory,
        radius: int,
    ) -> None:
        """Every local-window tile equals the global tile at (py±r, px±r)."""
        from factoriax.engine.constants import Direction, ItemType

        h = w = 16
        # Sprinkle content that exercises every spatial channel:
        # terrain, machines, resources, all 3 slots, direction.
        world_map = jnp.full((h, w), int(BlockType.DIRT), dtype=jnp.int32)
        world_map = world_map.at[4, 6].set(int(BlockType.COAL))
        world_map = world_map.at[9, 2].set(int(BlockType.WATER))
        world_map = world_map.at[11, 12].set(int(BlockType.IRON))

        mt = jnp.full((h, w), int(Machine.NONE), dtype=jnp.int32)
        mt = mt.at[7, 7].set(int(Machine.ASSEMBLER))
        mt = mt.at[8, 9].set(int(Machine.MINER))
        mt = mt.at[5, 10].set(int(Machine.PALLET))

        md = jnp.zeros((h, w), dtype=jnp.int8)
        md = md.at[7, 7].set(int(Direction.RIGHT))
        md = md.at[8, 9].set(int(Direction.UP))
        md = md.at[5, 10].set(int(Direction.LEFT))

        ait = jnp.zeros((h, w, 2), dtype=jnp.int32)
        ait = ait.at[7, 7, 0].set(int(ItemType.IRON_PLATE))
        ait = ait.at[7, 7, 1].set(int(ItemType.TIN_PLATE))
        aic = jnp.zeros((h, w, 2), dtype=jnp.int32)
        aic = aic.at[7, 7, 0].set(3)
        aic = aic.at[7, 7, 1].set(2)
        aot = jnp.zeros((h, w), dtype=jnp.int32).at[7, 7].set(int(ItemType.FRAME))
        aoc = jnp.zeros((h, w), dtype=jnp.int32).at[7, 7].set(1)

        bt = jnp.zeros((h, w), dtype=jnp.int32)
        bt = bt.at[8, 9].set(int(ItemType.IRON_ORE))
        bt = bt.at[5, 10].set(int(ItemType.COAL))
        bc = jnp.zeros((h, w), dtype=jnp.int32)
        bc = bc.at[8, 9].set(7)
        bc = bc.at[5, 10].set(42)

        resources = jnp.zeros((h, w), dtype=jnp.int16)
        resources = resources.at[4, 6].set(BLOCK_MAX_RESOURCES)
        resources = resources.at[11, 12].set(BLOCK_MAX_RESOURCES // 2)

        # Place the player far enough from every edge that the whole
        # window is in-bounds; OOB padding behaviour is covered by
        # test_oob_padding_at_corner above.
        px, py = 7, 8
        assert px - radius >= 0 and px + radius < w
        assert py - radius >= 0 and py + radius < h

        state = state_factory(
            world_map=world_map,
            player_position=(px, py),
            machine_types=mt,
            machine_direction=md,
            buffer_type=bt,
            buffer_count=bc,
            asm_in_type=ait,
            asm_in_count=aic,
            asm_out_type=aot,
            asm_out_count=aoc,
            block_resources=resources,
        )

        params = EnvParams(max_timesteps=100)
        global_obs = np.array(global_x_ray(state, params, 0))
        local_obs = np.array(local_x_ray(state, params, 0, radius=radius))

        window = 2 * radius + 1
        tile_count = h * w
        for ch_idx, ch_name in enumerate(_X_RAY_SPATIAL_CHANNEL_NAMES):
            global_channel = global_obs[
                ch_idx * tile_count : (ch_idx + 1) * tile_count
            ].reshape(h, w)
            local_channel = local_obs[
                ch_idx * window * window : (ch_idx + 1) * window * window
            ].reshape(window, window)
            expected = global_channel[
                py - radius : py + radius + 1,
                px - radius : px + radius + 1,
            ]
            np.testing.assert_allclose(
                local_channel,
                expected,
                atol=0.0,
                err_msg=(
                    f"channel '{ch_name}' diverges between local "
                    f"(r={radius}) and global at player ({px}, {py})"
                ),
            )

    def test_scalar_tail_matches_global(self, state_factory) -> None:
        """Player scalars + research tail is byte-for-byte identical."""
        from factoriax.engine.constants import ItemType

        radius = _RADIUS
        inv = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int32)
        inv = inv.at[0, ItemType.IRON_ORE].set(11)
        state = state_factory(
            world_map=jnp.ones((8, 8), dtype=jnp.int32) * int(BlockType.DIRT),
            player_position=(4, 4),
            timestep=37,
            player_inventory=inv,
        )
        global_obs = np.array(global_x_ray(state, _DEFAULT_PARAMS, 0))
        local_obs = np.array(local_x_ray(state, _DEFAULT_PARAMS, 0, radius=radius))

        global_spatial = (
            NUM_SPATIAL_CHANNELS["x_ray"]
            * _MAP_W
            * _MAP_H
        )
        window = 2 * radius + 1
        local_spatial = NUM_SPATIAL_CHANNELS["x_ray"] * window * window
        np.testing.assert_array_equal(
            global_obs[global_spatial:],
            local_obs[local_spatial:],
        )


class TestMachineDirectionChannel:
    """Direction channel mirrors ``ent_direction`` for active machines."""

    def test_placed_miner_direction_visible(self, state_factory) -> None:
        """A miner facing RIGHT shows the RIGHT enum value at its tile."""
        from factoriax.engine.constants import Direction

        shape = (_MAP_H, _MAP_W)
        world_map = jnp.full(shape, int(BlockType.IRON), dtype=jnp.int32)
        mt = jnp.full(shape, int(Machine.NONE), dtype=jnp.int32)
        mt = mt.at[4, 4].set(int(Machine.MINER))
        md = jnp.zeros(shape, dtype=jnp.int8).at[4, 4].set(int(Direction.RIGHT))

        state = state_factory(
            world_map=world_map,
            machine_types=mt,
            machine_direction=md,
        )
        obs = global_x_ray(state, _DEFAULT_PARAMS, 0)
        d = _slice_channel(obs, _DEFAULT_PARAMS, "machine_direction")

        # Normalised by 4 (max direction value). RIGHT = 2.
        assert d[4, 4] == pytest.approx(2 / 4)
        assert d[0, 0] == 0  # no machine
