"""Tests for the deposit action in :mod:`factoriax.engine.step`.

A deposit moves items from the player into the machine the player faces.
Where they land depends on the machine kind: a pallet and a miner take them
into ``ent_buf``, and an assembler, a furnace, and a science lab take them
into ``ent_asm_in``. A full destination refuses the move and leaves the
player's inventory alone.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from factoriax.engine.constants import BlockType, Direction, ItemType, Machine
from factoriax.engine.step import deposit_to_adjacent
from factoriax.engine.tables import MACHINE_MAX_STACK
from tests.helpers.states import (
    buffer_grids,
    entity_at,
    machine_type_grid,
    player_inventory_grid,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIRT_3X3 = jnp.full((3, 3), BlockType.DIRT, dtype=jnp.int32)


def _asm_out_grids(
    h: int,
    w: int,
    entries: dict[tuple[int, int], tuple[int, int]],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build asm_out_type and asm_out_count grids.

    Parameters
    ----------
    h
        Grid height.
    w
        Grid width.
    entries
        Map of ``(y, x)`` to ``(item_type, count)``.

    Returns
    -------
    tuple of jnp.ndarray
        ``(asm_out_type, asm_out_count)``.
    """
    ot = np.zeros((h, w), dtype=np.int8)
    oc = np.zeros((h, w), dtype=np.int16)
    for (y, x), (itype, count) in entries.items():
        ot[y, x] = itype
        oc[y, x] = count
    return jnp.array(ot), jnp.array(oc)


def _asm_in_grids(
    h: int,
    w: int,
    entries: dict[tuple[int, int], list[tuple[int, int]]],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build asm_in_type and asm_in_count grids.

    Parameters
    ----------
    h
        Grid height.
    w
        Grid width.
    entries
        Map of ``(y, x)`` to ``[(item_type, count), ...]``, up to 2 slots.

    Returns
    -------
    tuple of jnp.ndarray
        ``(asm_in_type, asm_in_count)``, each shaped ``(h, w, 2)``.
    """
    ait = np.zeros((h, w, 2), dtype=np.int8)
    aic = np.zeros((h, w, 2), dtype=np.int16)
    for (y, x), slots in entries.items():
        for s, (itype, count) in enumerate(slots):
            ait[y, x, s] = itype
            aic[y, x, s] = count
    return jnp.array(ait), jnp.array(aic)


class TestDepositToPallet:
    """Deposit items from player inventory into a pallet."""

    def test_deposit_coal_into_empty_pallet(self, state_factory) -> None:
        """Coal transfers one item into the pallet's buffer."""
        p_inv = player_inventory_grid(1, {ItemType.COAL: 10})
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
            player_inventory=p_inv,
            machine_types=machine_type_grid(3, 3, {(2, 1): Machine.PALLET}),
        )

        state = deposit_to_adjacent(state, 0, ItemType.COAL)

        eidx = entity_at(state, 1, 2)
        assert int(state.player_inventory[0, ItemType.COAL]) == 9
        assert int(state.ent_buf_type[eidx]) == int(ItemType.COAL)
        assert int(state.ent_buf_count[eidx]) == 1

    def test_deposit_stacks_into_matching_type(self, state_factory) -> None:
        """A coal deposit adds to the coal already in the buffer."""
        p_inv = player_inventory_grid(1, {ItemType.COAL: 5})
        bt, bc = buffer_grids(3, 3, {(1, 2): (ItemType.COAL, 10)})
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
            player_inventory=p_inv,
            machine_types=machine_type_grid(3, 3, {(2, 1): Machine.PALLET}),
            buffer_type=bt,
            buffer_count=bc,
        )

        state = deposit_to_adjacent(state, 0, ItemType.COAL)

        eidx = entity_at(state, 1, 2)
        assert int(state.player_inventory[0, ItemType.COAL]) == 4
        assert int(state.ent_buf_count[eidx]) == 11

    def test_deposit_noop_at_cap(self, state_factory) -> None:
        """Deposit is a no-op once the buffer is at the machine's capacity.

        The capacity is the pallet's own ``MACHINE_MAX_STACK`` entry. This
        used to assert 64, a literal the deposit path enforced for every
        machine kind regardless of the table.
        """
        cap = int(MACHINE_MAX_STACK[int(Machine.PALLET)])
        p_inv = player_inventory_grid(1, {ItemType.COAL: 20})
        bt, bc = buffer_grids(3, 3, {(1, 2): (ItemType.COAL, cap)})
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
            player_inventory=p_inv,
            machine_types=machine_type_grid(3, 3, {(2, 1): Machine.PALLET}),
            buffer_type=bt,
            buffer_count=bc,
        )

        state = deposit_to_adjacent(state, 0, ItemType.COAL)

        eidx = entity_at(state, 1, 2)
        assert int(state.ent_buf_count[eidx]) == cap
        assert int(state.player_inventory[0, ItemType.COAL]) == 20

    def test_deposit_noop_no_machine(self, state_factory) -> None:
        """A deposit into an empty tile is a no-op."""
        p_inv = player_inventory_grid(1, {ItemType.COAL: 10})
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
            player_inventory=p_inv,
        )

        state = deposit_to_adjacent(state, 0, ItemType.COAL)

        assert int(state.player_inventory[0, ItemType.COAL]) == 10

    def test_deposit_noop_empty_type(self, state_factory) -> None:
        """A deposit with a zero count of the item type is a no-op."""
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
            machine_types=machine_type_grid(3, 3, {(2, 1): Machine.PALLET}),
        )

        state = deposit_to_adjacent(state, 0, ItemType.COAL)

        eidx = entity_at(state, 1, 2)
        assert int(state.ent_buf_count[eidx]) == 0

    def test_deposit_noop_out_of_bounds(self, state_factory) -> None:
        """A deposit that faces out of bounds is a no-op."""
        p_inv = player_inventory_grid(1, {ItemType.COAL: 10})
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(2, 1),
            player_direction=Direction.RIGHT,
            player_inventory=p_inv,
        )

        state = deposit_to_adjacent(state, 0, ItemType.COAL)

        assert int(state.player_inventory[0, ItemType.COAL]) == 10


class TestDepositToMiner:
    """Miners have no input slot and reject all deposits."""

    def test_deposit_coal_rejected_by_miner(self, state_factory) -> None:
        """A miner rejects a coal deposit."""
        p_inv = player_inventory_grid(1, {ItemType.COAL: 5})
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
            player_inventory=p_inv,
            machine_types=machine_type_grid(3, 3, {(2, 1): Machine.MINER}),
        )

        state = deposit_to_adjacent(state, 0, ItemType.COAL)

        assert int(state.player_inventory[0, ItemType.COAL]) == 5

    def test_deposit_iron_rejected_by_miner(self, state_factory) -> None:
        """A miner rejects an iron deposit."""
        p_inv = player_inventory_grid(1, {ItemType.IRON_ORE: 5})
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
            player_inventory=p_inv,
            machine_types=machine_type_grid(3, 3, {(2, 1): Machine.MINER}),
        )

        state = deposit_to_adjacent(state, 0, ItemType.IRON_ORE)

        assert int(state.player_inventory[0, ItemType.IRON_ORE]) == 5


class TestDepositToAssembler:
    """Deposit items into an assembler's input slots."""

    def test_deposit_iron_into_assembler(self, state_factory) -> None:
        """An assembler input slot accepts iron."""
        p_inv = player_inventory_grid(1, {ItemType.IRON_ORE: 10})
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
            player_inventory=p_inv,
            machine_types=machine_type_grid(3, 3, {(2, 1): Machine.ASSEMBLER}),
        )

        state = deposit_to_adjacent(state, 0, ItemType.IRON_ORE)

        eidx = entity_at(state, 1, 2)
        assert int(state.ent_asm_in_type[eidx, 0]) == int(ItemType.IRON_ORE)
        assert int(state.ent_asm_in_count[eidx, 0]) == 1
        assert int(state.player_inventory[0, ItemType.IRON_ORE]) == 9

    def test_deposit_two_item_types_into_assembler(self, state_factory) -> None:
        """Two different item types go into separate input slots."""
        p_inv = player_inventory_grid(1, {ItemType.COPPER_ORE: 3})
        ait, aic = _asm_in_grids(3, 3, {(1, 2): [(ItemType.IRON_ORE, 2), (0, 0)]})
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
            player_inventory=p_inv,
            machine_types=machine_type_grid(3, 3, {(2, 1): Machine.ASSEMBLER}),
            asm_in_type=ait,
            asm_in_count=aic,
        )

        state = deposit_to_adjacent(state, 0, ItemType.COPPER_ORE)

        eidx = entity_at(state, 1, 2)
        # Iron stays in slot 0.
        assert int(state.ent_asm_in_type[eidx, 0]) == int(ItemType.IRON_ORE)
        assert int(state.ent_asm_in_count[eidx, 0]) == 2
        # Copper goes into slot 1.
        assert int(state.ent_asm_in_type[eidx, 1]) == int(ItemType.COPPER_ORE)
        assert int(state.ent_asm_in_count[eidx, 1]) == 1


class TestDepositToScienceLab:
    """Deposit science packs into a lab's input slots.

    ``run_labs`` consumes from ``ent_asm_in``. The deposit must land
    there, not in the buffer track, or hand-fed labs never consume.
    """

    def test_deposit_pack_into_lab_input_slot(self, state_factory) -> None:
        p_inv = player_inventory_grid(1, {ItemType.TIER1_SCIENCE_PACK: 2})
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
            player_inventory=p_inv,
            machine_types=machine_type_grid(3, 3, {(2, 1): Machine.SCIENCE_LAB}),
        )

        state = deposit_to_adjacent(state, 0, ItemType.TIER1_SCIENCE_PACK)

        eidx = entity_at(state, 1, 2)
        assert int(state.ent_asm_in_type[eidx, 0]) == int(ItemType.TIER1_SCIENCE_PACK)
        assert int(state.ent_asm_in_count[eidx, 0]) == 1
        assert int(state.ent_buf_count[eidx]) == 0
        assert int(state.player_inventory[0, ItemType.TIER1_SCIENCE_PACK]) == 1
