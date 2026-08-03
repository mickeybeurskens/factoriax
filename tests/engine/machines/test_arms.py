"""Tests for the arm pass in :mod:`factoriax.engine.machines`.

``run_arms`` moves one item from the tile behind an arm to the tile in
front. It is the only route into an assembler, a furnace, or a science lab,
because the belt pass refuses those. The tests cover the transfer, what a
full destination does, and that a transfer writes no slot the arm does not
own.
"""

from __future__ import annotations

import jax.numpy as jnp

from factoriax.engine.constants import (
    BlockType,
    Direction,
    ItemType,
    Machine,
)
from factoriax.engine.machines import run_arms
from factoriax.engine.state import EnvParams, EnvState
from tests.helpers.states import entity_at as _eid

_BELT_MAX = 3


_NOOP = 0  # Direction that does not push anywhere.


_PARAMS = EnvParams()


def _buf_grids(
    shape: tuple[int, int],
    entries: dict[tuple[int, int], tuple[int, int]] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build buffer_type and buffer_count grids.

    Parameters
    ----------
    shape
        Grid ``(H, W)``.
    entries
        Mapping of ``(y, x)`` to ``(item_type, count)``.

    Returns
    -------
    tuple of jnp.ndarray
        ``(buffer_type, buffer_count)``, each shaped ``(H, W)``.
    """
    bt = jnp.zeros(shape, dtype=jnp.int8)
    bc = jnp.zeros(shape, dtype=jnp.int16)
    for (y, x), (item_type, count) in (entries or {}).items():
        bt = bt.at[y, x].set(item_type)
        bc = bc.at[y, x].set(count)
    return bt, bc


def _make_state(
    state_factory,
    *,
    machine_types: jnp.ndarray,
    machine_direction: jnp.ndarray,
    buffer_type: jnp.ndarray | None = None,
    buffer_count: jnp.ndarray | None = None,
):
    """Build a state with the given machine grid and optional buffer.

    Uses max_machines equal to the actual entity count to avoid
    inactive-entity scatter collisions in JAX vectorised loops.

    Parameters
    ----------
    state_factory
        Pytest fixture that creates an ``EnvState``.
    machine_types
        Machine type per tile.
    machine_direction
        Direction per tile.
    buffer_type
        Buffer item type per tile. Optional.
    buffer_count
        Buffer item count per tile. Optional.

    Returns
    -------
    EnvState
        Configured state.
    """
    shape = machine_types.shape
    n_entities = int((machine_types != int(Machine.NONE)).sum())
    return state_factory(
        world_map=jnp.zeros(shape, dtype=jnp.int32),
        machine_types=machine_types,
        machine_direction=machine_direction,
        buffer_type=buffer_type,
        buffer_count=buffer_count,
        max_machines=n_entities,
    )


def _get_buf(state, y: int, x: int) -> tuple[int, int]:
    """Return (buf_type, buf_count) for the entity at tile (y, x).

    Parameters
    ----------
    state
        Environment state.
    y
        Tile row.
    x
        Tile column.

    Returns
    -------
    tuple of int
        ``(item_type, count)``, or ``(0, 0)`` when the tile is empty.
    """
    eid = int(state.tile_entity[y, x])
    if eid < 0:
        return (0, 0)
    return (int(state.ent_buf_type[eid]), int(state.ent_buf_count[eid]))


B = Machine.CONVEYOR_BELT


R = int(Direction.RIGHT)


D = int(Direction.DOWN)


class TestArm:
    """Pick-and-place arm tests.

    Arms have no internal buffer. Each tick, an arm transfers 1 item from
    the entity behind it to the entity in front (facing direction).
    Layout: [SOURCE, ARM(RIGHT), DESTINATION] means the arm picks from
    SOURCE (behind) and deposits into DESTINATION (in front).
    """

    def test_transfers_from_miner_to_belt(self, state_factory) -> None:
        """Arm transfers 1 item from miner behind to belt in front."""
        # [MINER, ARM(RIGHT), BELT]
        types = jnp.array([[Machine.MINER, Machine.ARM, B]])
        dirs = jnp.array([[_NOOP, R, _NOOP]])
        bt, bc = _buf_grids((1, 3), {(0, 0): (ItemType.COAL, 5)})
        state = _make_state(
            state_factory,
            machine_types=types,
            machine_direction=dirs,
            buffer_type=bt,
            buffer_count=bc,
        )
        result = run_arms(state, _PARAMS)
        belt_type, belt_count = _get_buf(result, 0, 2)
        assert belt_type == int(ItemType.COAL)
        assert belt_count == 1
        _, miner_count = _get_buf(result, 0, 0)
        assert miner_count == 4

    def test_deposits_to_pallet(self, state_factory) -> None:
        """Arm transfers 1 item from belt behind to pallet in front."""
        # [BELT, ARM(RIGHT), PALLET]
        types = jnp.array([[B, Machine.ARM, Machine.PALLET]])
        dirs = jnp.array([[_NOOP, R, _NOOP]])
        bt, bc = _buf_grids((1, 3), {(0, 0): (ItemType.IRON_ORE, 3)})
        state = _make_state(
            state_factory,
            machine_types=types,
            machine_direction=dirs,
            buffer_type=bt,
            buffer_count=bc,
        )
        result = run_arms(state, _PARAMS)
        pallet_type, pallet_count = _get_buf(result, 0, 2)
        assert pallet_type == int(ItemType.IRON_ORE)
        assert pallet_count == 1
        _, belt_count = _get_buf(result, 0, 0)
        assert belt_count == 2

    def test_delivers_pack_into_lab_input_slot(self, state_factory) -> None:
        """Arm feeds a science lab's ``ent_asm_in`` slot, not its buffer.

        ``run_labs`` only consumes from the input slots, so automated
        lab feeding depends on arms routing packs there.
        """
        # [BELT, ARM(RIGHT), LAB]
        types = jnp.array([[B, Machine.ARM, Machine.SCIENCE_LAB]])
        dirs = jnp.array([[_NOOP, R, _NOOP]])
        bt, bc = _buf_grids((1, 3), {(0, 0): (ItemType.TIER1_SCIENCE_PACK, 3)})
        state = _make_state(
            state_factory,
            machine_types=types,
            machine_direction=dirs,
            buffer_type=bt,
            buffer_count=bc,
        )
        result = run_arms(state, _PARAMS)
        eid = int(result.tile_entity[0, 2])
        assert int(result.ent_asm_in_type[eid, 0]) == int(ItemType.TIER1_SCIENCE_PACK)
        assert int(result.ent_asm_in_count[eid, 0]) == 1
        assert int(result.ent_buf_count[eid]) == 0
        _, belt_count = _get_buf(result, 0, 0)
        assert belt_count == 2

    def test_merges_same_item_into_destination(self, state_factory) -> None:
        """Arm merges item into destination holding the same type."""
        # [MINER, ARM(RIGHT), PALLET with existing COAL]
        types = jnp.array([[Machine.MINER, Machine.ARM, Machine.PALLET]])
        dirs = jnp.array([[_NOOP, R, _NOOP]])
        bt, bc = _buf_grids(
            (1, 3),
            {
                (0, 0): (ItemType.COAL, 5),
                (0, 2): (ItemType.COAL, 10),
            },
        )
        state = _make_state(
            state_factory,
            machine_types=types,
            machine_direction=dirs,
            buffer_type=bt,
            buffer_count=bc,
        )
        result = run_arms(state, _PARAMS)
        pallet_type, pallet_count = _get_buf(result, 0, 2)
        assert pallet_type == int(ItemType.COAL)
        assert pallet_count == 11
        _, miner_count = _get_buf(result, 0, 0)
        assert miner_count == 4

    def test_noop_when_dest_full(self, state_factory) -> None:
        """Arm does nothing when destination is at max stack."""
        # [MINER, ARM(RIGHT), BELT at max]
        types = jnp.array([[Machine.MINER, Machine.ARM, B]])
        dirs = jnp.array([[_NOOP, R, _NOOP]])
        bt, bc = _buf_grids(
            (1, 3),
            {
                (0, 0): (ItemType.COAL, 5),
                (0, 2): (ItemType.COAL, _BELT_MAX),
            },
        )
        state = _make_state(
            state_factory,
            machine_types=types,
            machine_direction=dirs,
            buffer_type=bt,
            buffer_count=bc,
        )
        result = run_arms(state, _PARAMS)
        _, miner_count = _get_buf(result, 0, 0)
        assert miner_count == 5
        _, belt_count = _get_buf(result, 0, 2)
        assert belt_count == _BELT_MAX

    def test_noop_when_source_empty(self, state_factory) -> None:
        """Arm with empty source does nothing."""
        # [empty MINER, ARM(RIGHT), BELT]
        types = jnp.array([[Machine.MINER, Machine.ARM, B]])
        dirs = jnp.array([[_NOOP, R, _NOOP]])
        state = _make_state(
            state_factory,
            machine_types=types,
            machine_direction=dirs,
        )
        result = run_arms(state, _PARAMS)
        assert _get_buf(result, 0, 0) == (0, 0)
        assert _get_buf(result, 0, 2) == (0, 0)

    def test_noop_when_no_dest_entity(self, state_factory) -> None:
        """Arm facing into empty tile does nothing."""
        # [MINER, ARM(RIGHT), NONE]
        types = jnp.array([[Machine.MINER, Machine.ARM, Machine.NONE]])
        dirs = jnp.array([[_NOOP, R, _NOOP]])
        bt, bc = _buf_grids((1, 3), {(0, 0): (ItemType.COAL, 3)})
        state = _make_state(
            state_factory,
            machine_types=types,
            machine_direction=dirs,
            buffer_type=bt,
            buffer_count=bc,
        )
        result = run_arms(state, _PARAMS)
        _, miner_count = _get_buf(result, 0, 0)
        assert miner_count == 3

    def test_does_not_corrupt_unrelated_entities(
        self,
        state_factory,
    ) -> None:
        """Arm transfer does not zero out unrelated entity buffers.

        Reproduces a bug where the pick phase's item-type clearing used
        a gather+where instead of a scatter, causing nearby entity
        buffers to be zeroed in the same tick.
        """
        # [PALLET, ARM(RIGHT), BELT]
        types = jnp.array([[Machine.PALLET, Machine.ARM, B]])
        dirs = jnp.array([[_NOOP, R, _NOOP]])
        bt, bc = _buf_grids(
            (1, 3),
            {(0, 0): (ItemType.COAL, 3)},
        )
        state = _make_state(
            state_factory,
            machine_types=types,
            machine_direction=dirs,
            buffer_type=bt,
            buffer_count=bc,
        )
        result = run_arms(state, _PARAMS)
        belt_type, belt_count = _get_buf(result, 0, 2)
        assert belt_type == int(ItemType.COAL), "belt item corrupted"
        assert belt_count == 1
        pallet_type, pallet_count = _get_buf(result, 0, 0)
        assert pallet_type == int(ItemType.COAL)
        assert pallet_count == 2


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

        Parameters
        ----------
        state_factory
            The shared ``state_factory`` fixture.
        arm_direction
            ``Direction.LEFT`` or ``Direction.RIGHT``.

        Returns
        -------
        EnvState
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
