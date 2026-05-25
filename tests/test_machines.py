"""Tests for the machine system."""

import jax.numpy as jnp
from jax import random

from factoriax import BlockType, EnvParams, EnvState, ItemType
from factoriax.constants import (
    MAX_MACHINE_STACK_SIZE,
    Direction,
    MachineType,
)
from factoriax.levels import generate_state
from factoriax.machines import run_conveyor_belts, run_miners, update_all_machines


def _eid(state: EnvState, y: int, x: int) -> int:
    """Look up the entity index at grid position (y, x).

    Args:
        state: Environment state with tile_entity grid.
        y: Row index.
        x: Column index.

    Returns:
        Entity index at the given tile.
    """
    return int(state.tile_entity[y, x])


class TestMachineInitialization:
    """Tests for machine state initialization."""

    def test_generate_state_initializes_no_machines(self) -> None:
        """Generated world should have no machines by default."""
        rng = random.PRNGKey(0)
        params = EnvParams()
        state = generate_state(rng, params)

        assert jnp.all(state.machine_types == MachineType.NONE)
        assert jnp.all(state.ent_power == 0)
        assert jnp.all(state.ent_buf_count == 0)

    def test_machine_arrays_match_map_shape(self) -> None:
        """Machine state arrays should have the expected shapes."""
        rng = random.PRNGKey(0)
        params = EnvParams(map_width=16, map_height=24)
        state = generate_state(rng, params)

        assert state.machine_types.shape == state.map.shape
        mm = params.resolved_max_machines()
        assert state.ent_power.shape == (mm,)
        assert state.ent_buf_type.shape == (mm,)
        assert state.ent_buf_count.shape == (mm,)


class TestMinerOperation:
    """Tests for miner machine behavior."""

    def test_miner_extracts_resources(self, state_factory) -> None:
        """Miner should extract resources from the block beneath it."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[50]], dtype=jnp.int16),
            machine_types=jnp.array([[MachineType.MINER]], dtype=jnp.int32),
        )
        params = EnvParams(map_width=1, map_height=1)

        new_state = run_miners(state, params)

        eid = _eid(new_state, 0, 0)
        assert new_state.block_resources[0, 0] == 47  # 50 - 3 (mining rate)
        assert new_state.ent_buf_count[eid] == 3
        assert new_state.ent_buf_type[eid] == ItemType.COAL

    def test_miner_stops_when_output_full(self, state_factory) -> None:
        """Miner should stop when output type is at max stack."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[50]], dtype=jnp.int16),
            machine_types=jnp.array([[MachineType.MINER]], dtype=jnp.int32),
            buffer_type=jnp.array([[int(ItemType.COAL)]], dtype=jnp.int8),
            buffer_count=jnp.array(
                [[MAX_MACHINE_STACK_SIZE]],
                dtype=jnp.int16,
            ),
        )
        params = EnvParams(map_width=1, map_height=1)

        new_state = run_miners(state, params)

        eid = _eid(new_state, 0, 0)
        assert new_state.block_resources[0, 0] == 50
        assert new_state.ent_buf_count[eid] == MAX_MACHINE_STACK_SIZE

    def test_miner_stops_when_no_resources(self, state_factory) -> None:
        """Miner should stop when block has no resources."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[0]], dtype=jnp.int16),
            machine_types=jnp.array([[MachineType.MINER]], dtype=jnp.int32),
        )
        params = EnvParams(map_width=1, map_height=1)

        new_state = run_miners(state, params)

        eid = _eid(new_state, 0, 0)
        assert new_state.ent_buf_count[eid] == 0

    def test_miner_depletes_block_to_dirt(self, state_factory) -> None:
        """Block should become dirt when fully depleted by miner."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[2]], dtype=jnp.int16),
            machine_types=jnp.array([[MachineType.MINER]], dtype=jnp.int32),
            max_machines=1,
        )
        params = EnvParams(map_width=1, map_height=1)

        new_state = run_miners(state, params)

        eid = _eid(new_state, 0, 0)
        assert new_state.map[0, 0] == BlockType.DIRT
        assert new_state.block_resources[0, 0] == 0
        assert new_state.ent_buf_count[eid] == 2

    def test_miner_mines_partial_when_limited_by_resources(
        self,
        state_factory,
    ) -> None:
        """Miner should extract only available resources when less than rate."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[1]], dtype=jnp.int16),
            machine_types=jnp.array([[MachineType.MINER]], dtype=jnp.int32),
        )
        params = EnvParams(map_width=1, map_height=1)

        new_state = run_miners(state, params)

        eid = _eid(new_state, 0, 0)
        assert new_state.block_resources[0, 0] == 0
        assert new_state.ent_buf_count[eid] == 1

    def test_miner_mines_partial_when_limited_by_space(
        self,
        state_factory,
    ) -> None:
        """Miner output slot caps at MINER_OUTPUT_CAP (no buffering).

        Full output slot: next tick mines 0 until a withdraw/arm/belt
        drains it.
        """
        from factoriax.machines import MINER_OUTPUT_CAP

        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[50]], dtype=jnp.int16),
            machine_types=jnp.array([[MachineType.MINER]], dtype=jnp.int32),
            buffer_type=jnp.array([[int(ItemType.COAL)]], dtype=jnp.int8),
            buffer_count=jnp.array([[MINER_OUTPUT_CAP - 1]], dtype=jnp.int16),
        )
        params = EnvParams(map_width=1, map_height=1)

        new_state = run_miners(state, params)

        eid = _eid(new_state, 0, 0)
        assert new_state.ent_buf_count[eid] == MINER_OUTPUT_CAP
        assert new_state.block_resources[0, 0] == 49  # one ore extracted


class TestMinerDifferentOres:
    """Tests for miners on different ore types."""

    import pytest

    @pytest.mark.parametrize(
        "block_type, item_type",
        [
            (BlockType.IRON, ItemType.IRON_ORE),
            (BlockType.COPPER, ItemType.COPPER_ORE),
        ],
        ids=["iron", "copper"],
    )
    def test_miner_on_ore(self, state_factory, block_type, item_type) -> None:
        """Miner should correctly mine the given ore type."""
        state = state_factory(
            world_map=jnp.array([[block_type]], dtype=jnp.int32),
            block_resources=jnp.array([[50]], dtype=jnp.int16),
            machine_types=jnp.array([[MachineType.MINER]], dtype=jnp.int32),
        )
        params = EnvParams(map_width=1, map_height=1)

        new_state = run_miners(state, params)

        eid = _eid(new_state, 0, 0)
        assert new_state.ent_buf_count[eid] == 3
        assert new_state.ent_buf_type[eid] == item_type


class TestMultipleMiners:
    """Tests for multiple miners operating in parallel."""

    def test_multiple_miners_update_in_parallel(self, state_factory) -> None:
        """Multiple miners should all update in a single step."""
        state = state_factory(
            world_map=jnp.array(
                [
                    [BlockType.COAL, BlockType.IRON],
                    [BlockType.COPPER, BlockType.DIRT],
                ],
                dtype=jnp.int32,
            ),
            block_resources=jnp.array([[50, 50], [50, 0]], dtype=jnp.int16),
            machine_types=jnp.array(
                [
                    [MachineType.MINER, MachineType.MINER],
                    [MachineType.MINER, MachineType.NONE],
                ],
                dtype=jnp.int32,
            ),
        )
        params = EnvParams(map_width=2, map_height=2)

        new_state = run_miners(state, params)

        assert new_state.block_resources[0, 0] == 47
        assert new_state.block_resources[0, 1] == 47
        assert new_state.block_resources[1, 0] == 47
        assert new_state.block_resources[1, 1] == 0

        eid_coal = _eid(new_state, 0, 0)
        eid_iron = _eid(new_state, 0, 1)
        eid_copper = _eid(new_state, 1, 0)
        assert new_state.ent_buf_count[eid_coal] == 3
        assert new_state.ent_buf_type[eid_coal] == ItemType.COAL
        assert new_state.ent_buf_count[eid_iron] == 3
        assert new_state.ent_buf_type[eid_iron] == ItemType.IRON_ORE
        assert new_state.ent_buf_count[eid_copper] == 3
        assert new_state.ent_buf_type[eid_copper] == ItemType.COPPER_ORE


class TestMinerPushDoesNotLeakIntoInactiveSlots:
    """Regression: a miner push must reach only the placed receiver.

    Inactive entity slots carry ``ent_y == ent_x == -1`` and the safety
    clip in :func:`run_miners` maps every one of them onto tile (0, 0).
    A miner standing next to (0, 0) and facing into it must not have its
    ore credited to those phantom slots, both because the slots later
    feed freshly-placed machines and because crediting more than one
    receiver per push mints items from nothing.
    """

    def _push_into_corner_state(self, state_factory) -> EnvState:
        """Build a miner at (0, 1) pushing COAL left into a pallet at (0, 0).

        The pallet occupies the corner tile that every inactive slot
        clips onto, so any ungated gather leaks the miner's ore into the
        inactive slots in lockstep with the real pallet.

        Args:
            state_factory: The shared ``state_factory`` fixture.

        Returns:
            A state with two active entities (pallet, miner) and the
            remaining slots inactive at (0, 0).
        """
        return state_factory(
            world_map=jnp.array([[BlockType.DIRT, BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[0, 50]], dtype=jnp.int16),
            machine_types=jnp.array(
                [[MachineType.PALLET, MachineType.MINER]], dtype=jnp.int32
            ),
            machine_direction=jnp.array(
                [[Direction.DOWN, Direction.LEFT]], dtype=jnp.int8
            ),
        )

    def test_inactive_slots_stay_empty_under_neighbour_push(
        self, state_factory
    ) -> None:
        """Inactive slots keep an empty buffer while the pallet fills."""
        state = self._push_into_corner_state(state_factory)
        params = EnvParams(map_width=2, map_height=1)

        for _ in range(5):
            state = run_miners(state, params)

        pallet_eid = _eid(state, 0, 0)
        assert state.ent_buf_count[pallet_eid] > 0
        assert state.ent_buf_type[pallet_eid] == ItemType.COAL

        inactive = state.ent_y < 0
        assert bool(jnp.all(state.ent_buf_count[inactive] == 0))
        assert bool(jnp.all(state.ent_buf_type[inactive] == 0))

    def test_push_conserves_items(self, state_factory) -> None:
        """Total buffered ore equals total mined ore (no duplication)."""
        state = self._push_into_corner_state(state_factory)
        params = EnvParams(map_width=2, map_height=1)

        for _ in range(5):
            state = run_miners(state, params)

        total_buffered = int(jnp.sum(state.ent_buf_count.astype(jnp.int32)))
        total_mined = int(jnp.sum(state.items_mined))
        assert total_buffered == total_mined


class TestBeltPushDoesNotLeakIntoInactiveSlots:
    """Regression: a belt push must reach only the placed receiver.

    ``run_conveyor_belts`` shares the gather pattern of
    :func:`run_miners`: every inactive slot clips onto tile (0, 0), so a
    belt next to (0, 0) and facing into it must not have its item
    credited to those phantom slots. The crossing-axis receiver track is
    already gated on ``is_crossing`` (active), but the buffer track is
    not, so belts and splitters leak without an explicit active gate.
    """

    def _belt_into_corner_state(self, state_factory) -> EnvState:
        """Build a belt at (0, 1) pushing COAL left into a pallet at (0, 0).

        Args:
            state_factory: The shared ``state_factory`` fixture.

        Returns:
            A state with two active entities (pallet, belt) and the
            remaining slots inactive at (0, 0). The belt holds one COAL.
        """
        return state_factory(
            world_map=jnp.array([[BlockType.DIRT, BlockType.DIRT]], dtype=jnp.int32),
            machine_types=jnp.array(
                [[MachineType.PALLET, MachineType.CONVEYOR_BELT]], dtype=jnp.int32
            ),
            machine_direction=jnp.array(
                [[Direction.DOWN, Direction.LEFT]], dtype=jnp.int8
            ),
            buffer_type=jnp.array([[0, int(ItemType.COAL)]], dtype=jnp.int8),
            buffer_count=jnp.array([[0, 1]], dtype=jnp.int16),
        )

    def test_inactive_slots_stay_empty_under_belt_push(self, state_factory) -> None:
        """Inactive slots keep an empty buffer while the pallet fills."""
        state = self._belt_into_corner_state(state_factory)
        params = EnvParams(map_width=2, map_height=1)

        state = run_conveyor_belts(state, params)

        pallet_eid = _eid(state, 0, 0)
        assert state.ent_buf_count[pallet_eid] == 1
        assert state.ent_buf_type[pallet_eid] == ItemType.COAL

        inactive = state.ent_y < 0
        assert bool(jnp.all(state.ent_buf_count[inactive] == 0))
        assert bool(jnp.all(state.ent_buf_type[inactive] == 0))

    def test_belt_push_conserves_items(self, state_factory) -> None:
        """The single COAL on the belt is moved, never duplicated."""
        state = self._belt_into_corner_state(state_factory)
        params = EnvParams(map_width=2, map_height=1)

        before = int(jnp.sum(state.ent_buf_count.astype(jnp.int32)))
        state = run_conveyor_belts(state, params)
        after = int(jnp.sum(state.ent_buf_count.astype(jnp.int32)))
        assert after == before


class TestUpdateAllMachines:
    """Tests for the combined machine update function."""

    def test_miner_runs_via_update_all(self, state_factory) -> None:
        """update_all_machines should run miners end-to-end."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[50]], dtype=jnp.int16),
            machine_types=jnp.array([[MachineType.MINER]], dtype=jnp.int32),
        )
        params = EnvParams(map_width=1, map_height=1)

        new_state = update_all_machines(state, params)

        eid = _eid(new_state, 0, 0)
        assert new_state.ent_buf_count[eid] == 3
        assert new_state.block_resources[0, 0] == 47
