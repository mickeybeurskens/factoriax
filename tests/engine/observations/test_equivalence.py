"""The local window must agree with the global map where they overlap.

``local_x_ray`` crops a window around the player out of the same world that
``global_x_ray`` encodes whole. Where the window sits inside the map, the two
must report the same tile. Outside it, the window pads.
"""

from __future__ import annotations

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
    global_x_ray,
    local_x_ray,
)
from factoriax.engine.state import EnvParams
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
    RADIUS as _RADIUS,
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


class TestLocalGlobalEquivalence:
    """Local window must match global obs tile-for-tile at the player.

    Invariant: ``local_x_ray(state, params, p, radius=R)`` is a pure
    spatial crop of ``global_x_ray(state, params, p)``. For every
    (y, x) inside the window, the per-channel value in the local obs
    must equal the corresponding tile in the global obs. The player
    scalars + research tail appended after the spatial block must be
    identical between the two. If this test fails, either ``local_x_ray``
    and ``global_x_ray`` have drifted in channel ordering / normalisation,
    or the window indexing is off. Both faults silently corrupt
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
        # window is in-bounds. OOB padding behaviour is covered by
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

        global_spatial = NUM_SPATIAL_CHANNELS["x_ray"] * _MAP_W * _MAP_H
        window = 2 * radius + 1
        local_spatial = NUM_SPATIAL_CHANNELS["x_ray"] * window * window
        np.testing.assert_array_equal(
            global_obs[global_spatial:],
            local_obs[local_spatial:],
        )
