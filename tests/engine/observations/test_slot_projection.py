"""Tests for the grids the observation reconstructs from entity arrays.

The engine holds machine contents in flat ``ent_*`` arrays. The observation
projects them back onto per-tile grids: three uniform slots per machine, plus
a facing grid. A machine the projection misses is invisible to an agent.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from factoriax.engine.constants import (
    BlockType,
    Machine,
)
from factoriax.engine.observations import (
    _X_RAY_SPATIAL_CHANNEL_NAMES,
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
from tests.helpers.observations import (
    slice_channel as _slice_channel,
)

_GLOBAL_OBS_SIZE = (
    NUM_SPATIAL_CHANNELS["x_ray"] * _MAP_W * _MAP_H + NUM_PLAYER_SCALARS["x_ray"]
)
_LOCAL_OBS_SIZE = (
    NUM_SPATIAL_CHANNELS["x_ray"] * _WINDOW * _WINDOW + NUM_PLAYER_SCALARS["x_ray"]
)


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
        # We only check "this channel is non-zero" here. The exact
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
