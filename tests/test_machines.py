"""Tests for the machine system."""

import jax.numpy as jnp
from jax import random

from factoriax.engine.constants import (
    BlockType,
    Direction,
    ItemType,
    Machine,
)
from factoriax.engine.levels import generate_state
from factoriax.engine.machines import (
    _lookup_neighbor,
    _subtract_buffer,
    run_arms,
    run_assemblers,
    run_conveyor_belts,
    run_miners,
    update_all_machines,
)
from factoriax.engine.state import EnvParams, EnvState
from factoriax.engine.tables import MACHINE_MAX_STACK

# The miner's buffer capacity — a "full" buffer value for the stop tests.
_MINER_BUF_CAP = int(MACHINE_MAX_STACK[int(Machine.MINER)])


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

        assert jnp.all(state.machine_types == Machine.NONE)
        assert jnp.all(state.ent_power == 0)
        assert jnp.all(state.ent_buf_count == 0)

    def test_machine_arrays_match_map_shape(self) -> None:
        """Machine state arrays should have the expected shapes."""
        rng = random.PRNGKey(0)
        params = EnvParams()
        state = generate_state(rng, params, 16, 24)

        assert state.machine_types.shape == state.map.shape
        mm = max(64, 16 * 24 // 4)
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
            machine_types=jnp.array([[Machine.MINER]], dtype=jnp.int32),
        )
        params = EnvParams()

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
            machine_types=jnp.array([[Machine.MINER]], dtype=jnp.int32),
            buffer_type=jnp.array([[int(ItemType.COAL)]], dtype=jnp.int8),
            buffer_count=jnp.array(
                [[_MINER_BUF_CAP]],
                dtype=jnp.int16,
            ),
        )
        params = EnvParams()

        new_state = run_miners(state, params)

        eid = _eid(new_state, 0, 0)
        assert new_state.block_resources[0, 0] == 50
        assert new_state.ent_buf_count[eid] == _MINER_BUF_CAP

    def test_miner_stops_when_no_resources(self, state_factory) -> None:
        """Miner should stop when block has no resources."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[0]], dtype=jnp.int16),
            machine_types=jnp.array([[Machine.MINER]], dtype=jnp.int32),
        )
        params = EnvParams()

        new_state = run_miners(state, params)

        eid = _eid(new_state, 0, 0)
        assert new_state.ent_buf_count[eid] == 0

    def test_miner_depletes_block_to_dirt(self, state_factory) -> None:
        """Block should become dirt when fully depleted by miner."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[2]], dtype=jnp.int16),
            machine_types=jnp.array([[Machine.MINER]], dtype=jnp.int32),
            max_machines=1,
        )
        params = EnvParams()

        new_state = run_miners(state, params)

        eid = _eid(new_state, 0, 0)
        assert new_state.map[0, 0] == BlockType.DIRT
        assert new_state.block_resources[0, 0] == 0
        assert new_state.ent_buf_count[eid] == 2

    def test_miner_depletes_block_to_dirt_with_free_slots_present(
        self, state_factory
    ) -> None:
        """Depletion converts the tile even when free entity slots exist.

        ``run_miners`` writes the map with ``.at[ey, ex].set(...)``. Free slots
        clip onto tile (0, 0) and write the unchanged block type to that same
        index, so a miner standing on (0, 0) puts one DIRT write and many
        stale-value writes into one scatter. Duplicate scatter indices resolve
        in an unspecified order, so the depletion can be lost. The test above
        misses this because ``max_machines=1`` leaves no free slot.
        """
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[2]], dtype=jnp.int16),
            machine_types=jnp.array([[Machine.MINER]], dtype=jnp.int32),
        )

        new_state = run_miners(state, EnvParams())

        assert new_state.block_resources[0, 0] == 0
        assert new_state.map[0, 0] == BlockType.DIRT

    def test_miner_mines_partial_when_limited_by_resources(
        self,
        state_factory,
    ) -> None:
        """Miner should extract only available resources when less than rate."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[1]], dtype=jnp.int16),
            machine_types=jnp.array([[Machine.MINER]], dtype=jnp.int32),
        )
        params = EnvParams()

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
        from factoriax.engine.machines import MINER_OUTPUT_CAP

        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[50]], dtype=jnp.int16),
            machine_types=jnp.array([[Machine.MINER]], dtype=jnp.int32),
            buffer_type=jnp.array([[int(ItemType.COAL)]], dtype=jnp.int8),
            buffer_count=jnp.array([[MINER_OUTPUT_CAP - 1]], dtype=jnp.int16),
        )
        params = EnvParams()

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
            machine_types=jnp.array([[Machine.MINER]], dtype=jnp.int32),
        )
        params = EnvParams()

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
                    [Machine.MINER, Machine.MINER],
                    [Machine.MINER, Machine.NONE],
                ],
                dtype=jnp.int32,
            ),
        )
        params = EnvParams()

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
            machine_types=jnp.array([[Machine.PALLET, Machine.MINER]], dtype=jnp.int32),
            machine_direction=jnp.array(
                [[Direction.DOWN, Direction.LEFT]], dtype=jnp.int8
            ),
        )

    def test_inactive_slots_stay_empty_under_neighbour_push(
        self, state_factory
    ) -> None:
        """Inactive slots keep an empty buffer while the pallet fills."""
        state = self._push_into_corner_state(state_factory)
        params = EnvParams()

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
        params = EnvParams()

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
                [[Machine.PALLET, Machine.CONVEYOR_BELT]], dtype=jnp.int32
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
        params = EnvParams()

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
        params = EnvParams()

        before = int(jnp.sum(state.ent_buf_count.astype(jnp.int32)))
        state = run_conveyor_belts(state, params)
        after = int(jnp.sum(state.ent_buf_count.astype(jnp.int32)))
        assert after == before


class TestPushIntoZeroCapacityMachine:
    """Regression: a machine that holds nothing must receive nothing.

    ``MACHINE_MAX_STACK`` is 0 for ``ROCKET``. Every push path tests the
    destination with "empty, or same item with room left", and an empty
    destination passes that test whatever its capacity. The transfer then
    clamps to 0 in :func:`run_miners` and :func:`run_conveyor_belts`, which
    leaves the receiver holding an item type and no items, breaking the rule
    that a zero count means a cleared type. :func:`run_arms` does not clamp
    at all and moves the item outright.
    """

    def test_belt_does_not_stamp_its_item_on_a_rocket(self, state_factory) -> None:
        """A belt facing a rocket keeps its item and leaves the rocket clear."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT, BlockType.DIRT]], dtype=jnp.int32),
            machine_types=jnp.array(
                [[Machine.CONVEYOR_BELT, Machine.ROCKET]], dtype=jnp.int32
            ),
            machine_direction=jnp.array(
                [[Direction.RIGHT, Direction.DOWN]], dtype=jnp.int8
            ),
            buffer_type=jnp.array([[int(ItemType.COAL), 0]], dtype=jnp.int8),
            buffer_count=jnp.array([[1, 0]], dtype=jnp.int16),
        )

        state = run_conveyor_belts(state, EnvParams())

        rocket = _eid(state, 0, 1)
        assert state.ent_buf_count[_eid(state, 0, 0)] == 1
        assert state.ent_buf_count[rocket] == 0
        assert state.ent_buf_type[rocket] == 0

    def test_miner_does_not_stamp_its_ore_on_a_rocket(self, state_factory) -> None:
        """A miner facing a rocket holds its ore and leaves the rocket clear."""
        state = state_factory(
            world_map=jnp.array(
                [[BlockType.DIRT, BlockType.COAL, BlockType.DIRT]], dtype=jnp.int32
            ),
            block_resources=jnp.array([[0, 50, 0]], dtype=jnp.int16),
            machine_types=jnp.array(
                [[Machine.NONE, Machine.MINER, Machine.ROCKET]], dtype=jnp.int32
            ),
            machine_direction=jnp.array(
                [[Direction.DOWN, Direction.RIGHT, Direction.DOWN]], dtype=jnp.int8
            ),
        )

        state = run_miners(state, EnvParams())

        rocket = _eid(state, 0, 2)
        assert state.ent_buf_count[_eid(state, 0, 1)] > 0
        assert state.ent_buf_count[rocket] == 0
        assert state.ent_buf_type[rocket] == 0

    def test_arm_does_not_deliver_into_a_rocket(self, state_factory) -> None:
        """An arm facing a rocket leaves the item on its source."""
        state = state_factory(
            world_map=jnp.full((1, 3), int(BlockType.DIRT), dtype=jnp.int32),
            machine_types=jnp.array(
                [[Machine.PALLET, Machine.ARM, Machine.ROCKET]], dtype=jnp.int32
            ),
            machine_direction=jnp.array(
                [[Direction.DOWN, Direction.RIGHT, Direction.DOWN]], dtype=jnp.int8
            ),
            buffer_type=jnp.array([[int(ItemType.COAL), 0, 0]], dtype=jnp.int8),
            buffer_count=jnp.array([[1, 0, 0]], dtype=jnp.int16),
        )

        state = run_arms(state, EnvParams())

        rocket = _eid(state, 0, 2)
        assert state.ent_buf_count[_eid(state, 0, 0)] == 1
        assert state.ent_buf_count[rocket] == 0
        assert state.ent_buf_type[rocket] == 0


class TestAssemblerPullDoesNotLeakIntoInactiveSlots:
    """Regression: the combiner's belt pull must debit only the placed belt.

    ``run_assemblers`` opens with a directional pull: a combiner takes one
    item from each adjacent belt facing it. The pull side is gated on
    ``is_combiner``, which carries ``active``, but the side that pays for it
    is not. Every inactive slot clips onto tile (0, 0), so a combiner on
    (0, 1) pulling leftwards reads as pulling from all of them at once and
    drives each to ``-1``.
    """

    def _belt_feeding_assembler(self, state_factory) -> EnvState:
        """Build a belt on (0, 0) facing RIGHT into an assembler on (0, 1).

        The belt occupies the corner tile that every inactive slot clips
        onto, which is what puts the inactive slots on the paying side of
        the pull.

        Args:
            state_factory: The shared ``state_factory`` fixture.

        Returns:
            A state with two active entities and the remaining slots
            inactive at (0, 0). The belt holds one IRON_ORE.
        """
        return state_factory(
            world_map=jnp.array([[BlockType.DIRT, BlockType.DIRT]], dtype=jnp.int32),
            machine_types=jnp.array(
                [[Machine.CONVEYOR_BELT, Machine.ASSEMBLER]], dtype=jnp.int32
            ),
            machine_direction=jnp.array(
                [[Direction.RIGHT, Direction.DOWN]], dtype=jnp.int8
            ),
            buffer_type=jnp.array([[int(ItemType.IRON_ORE), 0]], dtype=jnp.int8),
            buffer_count=jnp.array([[1, 0]], dtype=jnp.int16),
        )

    def test_inactive_slots_keep_empty_buffers_under_pull(self, state_factory) -> None:
        """The belt pays for the pull and no inactive slot goes negative."""
        state = self._belt_feeding_assembler(state_factory)

        state = run_assemblers(state, EnvParams())

        assert state.ent_buf_count[_eid(state, 0, 0)] == 0
        assert state.ent_asm_in_count[_eid(state, 0, 1), 0] == 1

        inactive = state.ent_y < 0
        assert bool(jnp.all(state.ent_buf_count[inactive] == 0))

    def test_pull_conserves_items(self, state_factory) -> None:
        """The IRON_ORE moves from belt buffer to input slot, once."""
        state = self._belt_feeding_assembler(state_factory)

        before = int(jnp.sum(state.ent_buf_count.astype(jnp.int32))) + int(
            jnp.sum(state.ent_asm_in_count.astype(jnp.int32))
        )
        state = run_assemblers(state, EnvParams())
        after = int(jnp.sum(state.ent_buf_count.astype(jnp.int32))) + int(
            jnp.sum(state.ent_asm_in_count.astype(jnp.int32))
        )

        assert after == before


class TestArmTransferDoesNotLeakIntoInactiveSlots:
    """Regression: an arm transfer must touch only the two placed entities.

    ``run_arms`` gathers on both sides: a receiver checks whether the tile
    behind it holds an arm pushing its way, and a giver checks whether the
    tile in front of it does. Every inactive slot clips onto tile (0, 0), so
    an arm standing on (0, 1) is the neighbour of every inactive slot at once
    on one side or the other, depending on which way it faces. Both sides
    need the active gate: without it an arm mints one item into every
    inactive slot when it faces (0, 0), and drives every inactive slot to
    ``-1`` when it faces away.
    """

    def _arm_state(self, state_factory, arm_direction: int) -> EnvState:
        """Build a pallet, an arm, and a pallet in a row, with COAL upstream.

        The arm sits at (0, 1) so that tile (0, 0) is its destination when it
        faces LEFT and its source when it faces RIGHT. Tile (0, 0) is the one
        every inactive slot clips onto, so either facing puts the arm on the
        inactive slots' neighbour tile.

        Args:
            state_factory: The shared ``state_factory`` fixture.
            arm_direction: ``Direction.LEFT`` or ``Direction.RIGHT``.

        Returns:
            A state with three active entities and the remaining slots
            inactive at (0, 0). One COAL sits in the pallet the arm draws
            from.
        """
        src_x = 2 if arm_direction == Direction.LEFT else 0
        coal = int(ItemType.COAL)
        buf_type = jnp.zeros((1, 3), dtype=jnp.int8).at[0, src_x].set(coal)
        buf_count = jnp.zeros((1, 3), dtype=jnp.int16).at[0, src_x].set(1)
        return state_factory(
            world_map=jnp.full((1, 3), int(BlockType.DIRT), dtype=jnp.int32),
            machine_types=jnp.array(
                [[Machine.PALLET, Machine.ARM, Machine.PALLET]], dtype=jnp.int32
            ),
            machine_direction=jnp.array(
                [[Direction.DOWN, arm_direction, Direction.DOWN]], dtype=jnp.int8
            ),
            buffer_type=buf_type,
            buffer_count=buf_count,
        )

    def test_inactive_slots_stay_empty_when_arm_faces_corner(
        self, state_factory
    ) -> None:
        """An arm delivering into (0, 0) credits the pallet there and nothing else."""
        state = self._arm_state(state_factory, int(Direction.LEFT))

        state = run_arms(state, EnvParams())

        assert state.ent_buf_count[_eid(state, 0, 0)] == 1
        assert state.ent_buf_type[_eid(state, 0, 0)] == ItemType.COAL

        inactive = state.ent_y < 0
        assert bool(jnp.all(state.ent_buf_count[inactive] == 0))
        assert bool(jnp.all(state.ent_buf_type[inactive] == 0))

    def test_inactive_slots_stay_empty_when_arm_draws_from_corner(
        self, state_factory
    ) -> None:
        """An arm drawing out of (0, 0) debits the pallet there and nothing else."""
        state = self._arm_state(state_factory, int(Direction.RIGHT))

        state = run_arms(state, EnvParams())

        assert state.ent_buf_count[_eid(state, 0, 0)] == 0
        assert state.ent_buf_count[_eid(state, 0, 2)] == 1

        inactive = state.ent_y < 0
        assert bool(jnp.all(state.ent_buf_count[inactive] == 0))
        assert bool(jnp.all(state.ent_buf_type[inactive] == 0))

    def test_arm_transfer_conserves_items(self, state_factory) -> None:
        """The single COAL is moved, never duplicated and never destroyed."""
        for arm_direction in (Direction.LEFT, Direction.RIGHT):
            state = self._arm_state(state_factory, int(arm_direction))

            before = int(jnp.sum(state.ent_buf_count.astype(jnp.int32)))
            state = run_arms(state, EnvParams())
            after = int(jnp.sum(state.ent_buf_count.astype(jnp.int32)))

            assert after == before


class TestUpdateAllMachines:
    """Tests for the combined machine update function."""

    def test_miner_runs_via_update_all(self, state_factory) -> None:
        """update_all_machines should run miners end-to-end."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[50]], dtype=jnp.int16),
            machine_types=jnp.array([[Machine.MINER]], dtype=jnp.int32),
        )
        params = EnvParams()

        new_state = update_all_machines(state, params)

        eid = _eid(new_state, 0, 0)
        assert new_state.ent_buf_count[eid] == 3
        assert new_state.block_resources[0, 0] == 47


class TestMachineHelpers:
    """Unit tests for private machine helpers — JIT-free, no state_factory."""

    def test_subtract_buffer_decrements_by_amount(self) -> None:
        cond = jnp.array([True, True, False])
        bt = jnp.array([5, 5, 5], dtype=jnp.int8)
        bc = jnp.array([3, 2, 10], dtype=jnp.int16)
        new_bt, new_bc = _subtract_buffer(cond, bt, bc, jnp.int16(2))
        assert new_bc.tolist() == [1, 0, 10]
        assert int(new_bt[0]) == 5  # still positive
        assert int(new_bt[1]) == 0  # count hit zero — cleared
        assert int(new_bt[2]) == 5  # unconditioned

    def test_subtract_buffer_clears_type_exactly_at_zero(self) -> None:
        cond = jnp.array([True, True])
        bt = jnp.array([7, 7], dtype=jnp.int8)
        bc = jnp.array([1, 2], dtype=jnp.int16)
        new_bt, new_bc = _subtract_buffer(cond, bt, bc, jnp.int16(1))
        assert int(new_bc[0]) == 0 and int(new_bt[0]) == 0  # zero — cleared
        assert int(new_bc[1]) == 1 and int(new_bt[1]) == 7  # still positive

    def test_subtract_buffer_no_change_when_false(self) -> None:
        cond = jnp.array([False])
        bt = jnp.array([3], dtype=jnp.int8)
        bc = jnp.array([5], dtype=jnp.int16)
        new_bt, new_bc = _subtract_buffer(cond, bt, bc, jnp.int16(1))
        assert int(new_bc[0]) == 5
        assert int(new_bt[0]) == 3

    def test_lookup_neighbor_finds_entity_at_neighbor_tile(self) -> None:
        h, w = 3, 3
        ey = jnp.array([1], dtype=jnp.int16)
        ex = jnp.array([1], dtype=jnp.int16)
        tile_entity = jnp.full((h, w), -1, dtype=jnp.int16).at[1, 2].set(5)
        _, _, eidx, valid, diff, safe = _lookup_neighbor(
            ey, ex, 0, 1, h, w, tile_entity, 9
        )
        assert int(eidx[0]) == 5
        assert bool(valid[0])
        assert bool(diff[0])
        assert int(safe[0]) == 5

    def test_lookup_neighbor_invalid_on_empty_tile(self) -> None:
        h, w = 3, 3
        ey = jnp.array([1], dtype=jnp.int16)
        ex = jnp.array([1], dtype=jnp.int16)
        tile_entity = jnp.full((h, w), -1, dtype=jnp.int16)
        _, _, eidx, valid, _, safe = _lookup_neighbor(
            ey, ex, 0, 1, h, w, tile_entity, 9
        )
        assert int(eidx[0]) == -1
        assert not bool(valid[0])
        assert int(safe[0]) == 0  # -1 clipped to 0

    def test_lookup_neighbor_diff_false_at_grid_edge(self) -> None:
        h, w = 3, 3
        # Entity at row 0 moving UP (dy=-1): clamped ny==ey, so diff=False
        ey = jnp.array([0], dtype=jnp.int16)
        ex = jnp.array([1], dtype=jnp.int16)
        tile_entity = jnp.full((h, w), -1, dtype=jnp.int16)
        _, _, _, _, diff, _ = _lookup_neighbor(ey, ex, -1, 0, h, w, tile_entity, 9)
        assert not bool(diff[0])
