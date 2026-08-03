"""Tests for the withdraw action in :mod:`factoriax.engine.step`.

A withdraw moves items from the machine the player faces back into the
player's inventory. It must empty the slot it reads, merge into a stack the
player already holds, and leave every other entity untouched.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from factoriax.engine.constants import BlockType, Direction, ItemType, Machine
from factoriax.engine.step import withdraw_from_adjacent
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


class TestWithdrawFromPallet:
    """Withdraw items from a pallet into player inventory."""

    def test_withdraw_iron_from_pallet(self, state_factory) -> None:
        """WITHDRAW pulls the whole stack (up to player capacity).

        The pallet has 10 iron ore. The player starts empty, with a 1024-stack
        cap, so all 10 transfer in one action.
        """
        bt, bc = buffer_grids(3, 3, {(1, 2): (ItemType.IRON_ORE, 10)})
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
            machine_types=machine_type_grid(3, 3, {(2, 1): Machine.PALLET}),
            buffer_type=bt,
            buffer_count=bc,
        )

        state = withdraw_from_adjacent(state, 0)

        eidx = entity_at(state, 1, 2)
        assert int(state.player_inventory[0, ItemType.IRON_ORE]) == 10
        assert int(state.ent_buf_count[eidx]) == 0

    def test_withdraw_noop_empty_machine(self, state_factory) -> None:
        """A withdraw from an empty pallet is a no-op."""
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
            machine_types=machine_type_grid(3, 3, {(2, 1): Machine.PALLET}),
        )

        state = withdraw_from_adjacent(state, 0)

        assert int(state.player_inventory[0, ItemType.IRON_ORE]) == 0

    def test_withdraw_noop_no_machine(self, state_factory) -> None:
        """A withdraw with no machine in front is a no-op."""
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
        )

        state = withdraw_from_adjacent(state, 0)

        assert int(state.player_inventory[0, ItemType.IRON_ORE]) == 0


class TestWithdrawFromMiner:
    """Withdraw from miners."""

    def test_withdraw_ore_from_miner(self, state_factory) -> None:
        """WITHDRAW pulls the whole miner output in one action."""
        bt, bc = buffer_grids(3, 3, {(1, 2): (ItemType.IRON_ORE, 10)})
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
            machine_types=machine_type_grid(3, 3, {(2, 1): Machine.MINER}),
            buffer_type=bt,
            buffer_count=bc,
        )

        state = withdraw_from_adjacent(state, 0)

        eidx = entity_at(state, 1, 2)
        assert int(state.player_inventory[0, ItemType.IRON_ORE]) == 10
        assert int(state.ent_buf_count[eidx]) == 0


class TestWithdrawFromAssembler:
    """Withdraw from assemblers uses the output slot."""

    def test_withdraw_output_from_assembler(self, state_factory) -> None:
        """WITHDRAW pulls all available output in one action."""
        aot, aoc = _asm_out_grids(3, 3, {(1, 2): (ItemType.FRAME, 5)})
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
            machine_types=machine_type_grid(3, 3, {(2, 1): Machine.ASSEMBLER}),
            asm_out_type=aot,
            asm_out_count=aoc,
        )

        state = withdraw_from_adjacent(state, 0)

        eidx = entity_at(state, 1, 2)
        assert int(state.player_inventory[0, ItemType.FRAME]) == 5
        assert int(state.ent_asm_out_count[eidx]) == 0


class TestWithdrawDoesNotClobberOtherEntities:
    """Regression: withdraw must not zero unrelated entities' slot metadata.

    ``run_assemblers and furnaces`` writes ``ent_asm_out_type`` at Phase 3 (cycle start)
    so an in-flight cycle can be identified as ``type=recipe_output,
    count=0``. Phase 1 only completes the cycle when ``asm_out_type != 0``.

    Before this regression, ``withdraw_from_adjacent`` cleared
    ``ent_asm_out_type`` elementwise for every entity whose count was
    zero, including unrelated cooking assemblers. The next tick's Phase
    1 then saw ``type=0`` on those and refused to write the output,
    silently consuming inputs without producing anything.
    """

    def test_withdraw_preserves_other_cooking_assembler_type(
        self,
        state_factory,
    ) -> None:
        """Two assemblers. A has a finished wire. B is cooking (type set,
        count 0). Withdrawing from A must leave B's out_type intact so B's
        cycle can complete next tick.
        """
        # Machine A at (2, 1): cycle done → out=(WIRE, 1).
        # Machine B at (2, 2): cycle cooking → out=(WIRE, 0), power=1.
        aot, aoc = _asm_out_grids(
            4,
            4,
            {
                (1, 2): (ItemType.WIRE, 1),
                (2, 2): (ItemType.WIRE, 0),
            },
        )
        world = jnp.full((4, 4), BlockType.DIRT, dtype=jnp.int32)
        state = state_factory(
            world_map=world,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
            machine_types=machine_type_grid(
                4,
                4,
                {
                    (2, 1): Machine.ASSEMBLER,
                    (2, 2): Machine.ASSEMBLER,
                },
            ),
            asm_out_type=aot,
            asm_out_count=aoc,
        )

        state = withdraw_from_adjacent(state, 0)

        e_a = entity_at(state, 1, 2)
        e_b = entity_at(state, 2, 2)

        # The player got the wire. Slot A is empty and its type is cleared.
        assert int(state.player_inventory[0, ItemType.WIRE]) == 1
        assert int(state.ent_asm_out_count[e_a]) == 0
        assert int(state.ent_asm_out_type[e_a]) == 0

        # B is untouched and still cooking. Its count stays 0, but its
        # out_type MUST still mark the in-flight recipe as WIRE so
        # Phase 1 can complete the cycle next tick.
        assert int(state.ent_asm_out_count[e_b]) == 0
        assert int(state.ent_asm_out_type[e_b]) == int(ItemType.WIRE), (
            "withdraw_from_adjacent clobbered another assembler's out_type; "
            "its in-flight cycle will never complete"
        )


class TestWithdrawMergesIntoInventory:
    """Withdrawn items merge with the player stacks that already exist."""

    def test_withdraw_merges_with_existing_stack(self, state_factory) -> None:
        """Withdrawn items merge with the existing player stack and
        the whole pallet drains in one action."""
        p_inv = player_inventory_grid(1, {ItemType.IRON_ORE: 3})
        bt, bc = buffer_grids(3, 3, {(1, 2): (ItemType.IRON_ORE, 7)})
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
            player_inventory=p_inv,
            machine_types=machine_type_grid(3, 3, {(2, 1): Machine.PALLET}),
            buffer_type=bt,
            buffer_count=bc,
        )

        state = withdraw_from_adjacent(state, 0)

        eidx = entity_at(state, 1, 2)
        assert int(state.player_inventory[0, ItemType.IRON_ORE]) == 10
        assert int(state.ent_buf_count[eidx]) == 0
