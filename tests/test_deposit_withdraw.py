"""Tests for the DEPOSIT and WITHDRAW compound actions.

Covers deposit into pallets, miners (fuel), and assemblers (input slots),
as well as withdraw from various machine types. Uses the entity-based
state model where machines are addressed via tile_entity -> entity arrays.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from factoriax.engine.constants import Action, BlockType, Direction, ItemType, Machine
from factoriax.engine.step import deposit_to_adjacent, withdraw_from_adjacent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIRT_3X3 = jnp.full((3, 3), BlockType.DIRT, dtype=jnp.int32)


def _machine_types(
    w: int, h: int, placements: dict[tuple[int, int], int]
) -> jnp.ndarray:
    """Build a machine_types grid with specific placements.

    Args:
        w: Grid width.
        h: Grid height.
        placements: Map of (x, y) -> Machine.

    Returns:
        Machine types grid of shape (h, w).
    """
    arr = jnp.full((h, w), Machine.NONE, dtype=jnp.int32)
    for (x, y), mtype in placements.items():
        arr = arr.at[y, x].set(mtype)
    return arr


def _player_inv(num_players: int, entries: dict[int, int]) -> jnp.ndarray:
    """Build a player inventory array.

    Args:
        num_players: Number of players.
        entries: Mapping of item_type -> count for player 0.

    Returns:
        Player inventory of shape (num_players, NUM_ITEM_TYPES).
    """
    from factoriax.engine.constants import NUM_ITEM_TYPES

    inv = jnp.zeros((num_players, NUM_ITEM_TYPES), dtype=jnp.int32)
    for item_type, count in entries.items():
        inv = inv.at[0, item_type].set(count)
    return inv


def _buf_grid(
    h: int,
    w: int,
    entries: dict[tuple[int, int], tuple[int, int]],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build buffer_type and buffer_count grids.

    Args:
        h: Grid height.
        w: Grid width.
        entries: Map of (y, x) -> (item_type, count).

    Returns:
        Tuple of (buffer_type, buffer_count) arrays.
    """
    bt = np.zeros((h, w), dtype=np.int8)
    bc = np.zeros((h, w), dtype=np.int16)
    for (y, x), (itype, count) in entries.items():
        bt[y, x] = itype
        bc[y, x] = count
    return jnp.array(bt), jnp.array(bc)


def _asm_out_grids(
    h: int,
    w: int,
    entries: dict[tuple[int, int], tuple[int, int]],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build asm_out_type and asm_out_count grids.

    Args:
        h: Grid height.
        w: Grid width.
        entries: Map of (y, x) -> (item_type, count).

    Returns:
        Tuple of (asm_out_type, asm_out_count) arrays.
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

    Args:
        h: Grid height.
        w: Grid width.
        entries: Map of (y, x) -> [(item_type, count), ...] for up to 2 slots.

    Returns:
        Tuple of (asm_in_type, asm_in_count) arrays with shape (h, w, 2).
    """
    ait = np.zeros((h, w, 2), dtype=np.int8)
    aic = np.zeros((h, w, 2), dtype=np.int16)
    for (y, x), slots in entries.items():
        for s, (itype, count) in enumerate(slots):
            ait[y, x, s] = itype
            aic[y, x, s] = count
    return jnp.array(ait), jnp.array(aic)


def _ent_lookup(state, y: int, x: int) -> int:
    """Get entity index for a tile position.

    Args:
        state: Current environment state.
        y: Tile row.
        x: Tile column.

    Returns:
        Entity index.
    """
    return int(state.tile_entity[y, x])


# ===========================================================================
# DEPOSIT tests
# ===========================================================================


class TestDepositToPallet:
    """Deposit items from player inventory into a pallet."""

    def test_deposit_coal_into_empty_pallet(self, state_factory) -> None:
        """Coal should transfer one item into the pallet's buffer."""
        p_inv = _player_inv(1, {ItemType.COAL: 10})
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
            player_inventory=p_inv,
            machine_types=_machine_types(3, 3, {(2, 1): Machine.PALLET}),
        )

        state = deposit_to_adjacent(state, 0, ItemType.COAL)

        eidx = _ent_lookup(state, 1, 2)
        assert int(state.player_inventory[0, ItemType.COAL]) == 9
        assert int(state.ent_buf_type[eidx]) == int(ItemType.COAL)
        assert int(state.ent_buf_count[eidx]) == 1

    def test_deposit_stacks_into_matching_type(self, state_factory) -> None:
        """Depositing coal should add to existing coal in the buffer."""
        p_inv = _player_inv(1, {ItemType.COAL: 5})
        bt, bc = _buf_grid(3, 3, {(1, 2): (ItemType.COAL, 10)})
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
            player_inventory=p_inv,
            machine_types=_machine_types(3, 3, {(2, 1): Machine.PALLET}),
            buffer_type=bt,
            buffer_count=bc,
        )

        state = deposit_to_adjacent(state, 0, ItemType.COAL)

        eidx = _ent_lookup(state, 1, 2)
        assert int(state.player_inventory[0, ItemType.COAL]) == 4
        assert int(state.ent_buf_count[eidx]) == 11

    def test_deposit_noop_at_cap(self, state_factory) -> None:
        """Deposit should be a no-op when buffer is at capacity (64)."""
        p_inv = _player_inv(1, {ItemType.COAL: 20})
        bt, bc = _buf_grid(3, 3, {(1, 2): (ItemType.COAL, 64)})
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
            player_inventory=p_inv,
            machine_types=_machine_types(3, 3, {(2, 1): Machine.PALLET}),
            buffer_type=bt,
            buffer_count=bc,
        )

        state = deposit_to_adjacent(state, 0, ItemType.COAL)

        eidx = _ent_lookup(state, 1, 2)
        assert int(state.ent_buf_count[eidx]) == 64
        assert int(state.player_inventory[0, ItemType.COAL]) == 20

    def test_deposit_noop_no_machine(self, state_factory) -> None:
        """Deposit into empty tile should be a no-op."""
        p_inv = _player_inv(1, {ItemType.COAL: 10})
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
            player_inventory=p_inv,
        )

        state = deposit_to_adjacent(state, 0, ItemType.COAL)

        assert int(state.player_inventory[0, ItemType.COAL]) == 10

    def test_deposit_noop_empty_type(self, state_factory) -> None:
        """Deposit with zero count of the item type should be a no-op."""
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
            machine_types=_machine_types(3, 3, {(2, 1): Machine.PALLET}),
        )

        state = deposit_to_adjacent(state, 0, ItemType.COAL)

        eidx = _ent_lookup(state, 1, 2)
        assert int(state.ent_buf_count[eidx]) == 0

    def test_deposit_noop_out_of_bounds(self, state_factory) -> None:
        """Deposit facing out of bounds should be a no-op."""
        p_inv = _player_inv(1, {ItemType.COAL: 10})
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
        """Coal deposited into a miner should be rejected."""
        p_inv = _player_inv(1, {ItemType.COAL: 5})
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
            player_inventory=p_inv,
            machine_types=_machine_types(3, 3, {(2, 1): Machine.MINER}),
        )

        state = deposit_to_adjacent(state, 0, ItemType.COAL)

        assert int(state.player_inventory[0, ItemType.COAL]) == 5

    def test_deposit_iron_rejected_by_miner(self, state_factory) -> None:
        """Iron deposited into a miner should be rejected."""
        p_inv = _player_inv(1, {ItemType.IRON_ORE: 5})
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
            player_inventory=p_inv,
            machine_types=_machine_types(3, 3, {(2, 1): Machine.MINER}),
        )

        state = deposit_to_adjacent(state, 0, ItemType.IRON_ORE)

        assert int(state.player_inventory[0, ItemType.IRON_ORE]) == 5


class TestDepositToAssembler:
    """Deposit items into an assembler's input slots."""

    def test_deposit_iron_into_assembler(self, state_factory) -> None:
        """Iron should be accepted into an assembler input slot."""
        p_inv = _player_inv(1, {ItemType.IRON_ORE: 10})
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
            player_inventory=p_inv,
            machine_types=_machine_types(3, 3, {(2, 1): Machine.ASSEMBLER}),
        )

        state = deposit_to_adjacent(state, 0, ItemType.IRON_ORE)

        eidx = _ent_lookup(state, 1, 2)
        assert int(state.ent_asm_in_type[eidx, 0]) == int(ItemType.IRON_ORE)
        assert int(state.ent_asm_in_count[eidx, 0]) == 1
        assert int(state.player_inventory[0, ItemType.IRON_ORE]) == 9

    def test_deposit_two_item_types_into_assembler(self, state_factory) -> None:
        """Two different item types should go into separate input slots."""
        p_inv = _player_inv(1, {ItemType.COPPER_ORE: 3})
        ait, aic = _asm_in_grids(3, 3, {(1, 2): [(ItemType.IRON_ORE, 2), (0, 0)]})
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
            player_inventory=p_inv,
            machine_types=_machine_types(3, 3, {(2, 1): Machine.ASSEMBLER}),
            asm_in_type=ait,
            asm_in_count=aic,
        )

        state = deposit_to_adjacent(state, 0, ItemType.COPPER_ORE)

        eidx = _ent_lookup(state, 1, 2)
        # Iron stays in slot 0.
        assert int(state.ent_asm_in_type[eidx, 0]) == int(ItemType.IRON_ORE)
        assert int(state.ent_asm_in_count[eidx, 0]) == 2
        # Copper goes into slot 1.
        assert int(state.ent_asm_in_type[eidx, 1]) == int(ItemType.COPPER_ORE)
        assert int(state.ent_asm_in_count[eidx, 1]) == 1


class TestDepositToScienceLab:
    """Deposit science packs into a lab's input slots.

    ``run_labs`` consumes from ``ent_asm_in`` — the deposit must land
    there (not in the buffer track) or hand-fed labs never consume.
    """

    def test_deposit_pack_into_lab_input_slot(self, state_factory) -> None:
        p_inv = _player_inv(1, {ItemType.TIER1_SCIENCE_PACK: 2})
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
            player_inventory=p_inv,
            machine_types=_machine_types(3, 3, {(2, 1): Machine.SCIENCE_LAB}),
        )

        state = deposit_to_adjacent(state, 0, ItemType.TIER1_SCIENCE_PACK)

        eidx = _ent_lookup(state, 1, 2)
        assert int(state.ent_asm_in_type[eidx, 0]) == int(
            ItemType.TIER1_SCIENCE_PACK
        )
        assert int(state.ent_asm_in_count[eidx, 0]) == 1
        assert int(state.ent_buf_count[eidx]) == 0
        assert int(state.player_inventory[0, ItemType.TIER1_SCIENCE_PACK]) == 1


# ===========================================================================
# WITHDRAW tests
# ===========================================================================


class TestWithdrawFromPallet:
    """Withdraw items from a pallet into player inventory."""

    def test_withdraw_iron_from_pallet(self, state_factory) -> None:
        """WITHDRAW pulls the whole stack (up to player capacity).

        Pallet has 10 iron ore; player starts empty with 1024-stack
        cap, so all 10 transfer in one action.
        """
        bt, bc = _buf_grid(3, 3, {(1, 2): (ItemType.IRON_ORE, 10)})
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
            machine_types=_machine_types(3, 3, {(2, 1): Machine.PALLET}),
            buffer_type=bt,
            buffer_count=bc,
        )

        state = withdraw_from_adjacent(state, 0)

        eidx = _ent_lookup(state, 1, 2)
        assert int(state.player_inventory[0, ItemType.IRON_ORE]) == 10
        assert int(state.ent_buf_count[eidx]) == 0

    def test_withdraw_noop_empty_machine(self, state_factory) -> None:
        """Withdraw from empty pallet should be a no-op."""
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
            machine_types=_machine_types(3, 3, {(2, 1): Machine.PALLET}),
        )

        state = withdraw_from_adjacent(state, 0)

        assert int(state.player_inventory[0, ItemType.IRON_ORE]) == 0

    def test_withdraw_noop_no_machine(self, state_factory) -> None:
        """Withdraw with no machine in front should be a no-op."""
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
        bt, bc = _buf_grid(3, 3, {(1, 2): (ItemType.IRON_ORE, 10)})
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
            machine_types=_machine_types(3, 3, {(2, 1): Machine.MINER}),
            buffer_type=bt,
            buffer_count=bc,
        )

        state = withdraw_from_adjacent(state, 0)

        eidx = _ent_lookup(state, 1, 2)
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
            machine_types=_machine_types(3, 3, {(2, 1): Machine.ASSEMBLER}),
            asm_out_type=aot,
            asm_out_count=aoc,
        )

        state = withdraw_from_adjacent(state, 0)

        eidx = _ent_lookup(state, 1, 2)
        assert int(state.player_inventory[0, ItemType.FRAME]) == 5
        assert int(state.ent_asm_out_count[eidx]) == 0


class TestWithdrawDoesNotClobberOtherEntities:
    """Regression: withdraw must not zero unrelated entities' slot metadata.

    ``run_combiners`` writes ``ent_asm_out_type`` at Phase 3 (cycle start)
    so an in-flight cycle can be identified as ``type=recipe_output,
    count=0``. Phase 1 only completes the cycle when ``asm_out_type != 0``.

    Before this regression, ``withdraw_from_adjacent`` cleared
    ``ent_asm_out_type`` elementwise for every entity whose count was
    zero — including unrelated cooking assemblers. The next tick's Phase
    1 then saw ``type=0`` on those and refused to write the output,
    silently consuming inputs without producing anything.
    """

    def test_withdraw_preserves_other_cooking_assembler_type(
        self,
        state_factory,
    ) -> None:
        """Two assemblers. A has a finished wire; B is cooking (type set,
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
            machine_types=_machine_types(
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

        e_a = _ent_lookup(state, 1, 2)
        e_b = _ent_lookup(state, 2, 2)

        # Player got the wire; A's slot is empty and its type cleared.
        assert int(state.player_inventory[0, ItemType.WIRE]) == 1
        assert int(state.ent_asm_out_count[e_a]) == 0
        assert int(state.ent_asm_out_type[e_a]) == 0

        # B is untouched — still cooking. Its count stays 0, but its
        # out_type MUST still mark the in-flight recipe as WIRE so
        # Phase 1 can complete the cycle next tick.
        assert int(state.ent_asm_out_count[e_b]) == 0
        assert int(state.ent_asm_out_type[e_b]) == int(ItemType.WIRE), (
            "withdraw_from_adjacent clobbered another assembler's out_type; "
            "its in-flight cycle will never complete"
        )


class TestWithdrawMergesIntoInventory:
    """Withdrawn items should merge with existing player stacks."""

    def test_withdraw_merges_with_existing_stack(self, state_factory) -> None:
        """Withdrawn items merge with the existing player stack and
        the whole pallet drains in one action."""
        p_inv = _player_inv(1, {ItemType.IRON_ORE: 3})
        bt, bc = _buf_grid(3, 3, {(1, 2): (ItemType.IRON_ORE, 7)})
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
            player_inventory=p_inv,
            machine_types=_machine_types(3, 3, {(2, 1): Machine.PALLET}),
            buffer_type=bt,
            buffer_count=bc,
        )

        state = withdraw_from_adjacent(state, 0)

        eidx = _ent_lookup(state, 1, 2)
        assert int(state.player_inventory[0, ItemType.IRON_ORE]) == 10
        assert int(state.ent_buf_count[eidx]) == 0


class TestDepositWithdrawViaStep:
    """End-to-end tests dispatched through factoriax_step."""

    def test_deposit_via_step(self, state_factory) -> None:
        """DEPOSIT_COAL action through the full step pipeline."""
        import jax

        from factoriax.engine.state import EnvParams
        from factoriax.engine.step import factoriax_step

        p_inv = _player_inv(1, {ItemType.COAL: 5})
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
            player_inventory=p_inv,
            machine_types=_machine_types(3, 3, {(2, 1): Machine.PALLET}),
        )
        params = EnvParams()
        rng = jax.random.PRNGKey(0)

        state = factoriax_step(rng, state, Action.DEPOSIT_COAL, params)

        eidx = _ent_lookup(state, 1, 2)
        assert int(state.ent_buf_count[eidx]) == 1
        assert int(state.player_inventory[0, ItemType.COAL]) == 4

    def test_withdraw_via_step(self, state_factory) -> None:
        """WITHDRAW action pulls from the buffer slot regardless of item."""
        import jax

        from factoriax.engine.state import EnvParams
        from factoriax.engine.step import factoriax_step

        bt, bc = _buf_grid(3, 3, {(1, 2): (ItemType.IRON_ORE, 10)})
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
            machine_types=_machine_types(3, 3, {(2, 1): Machine.PALLET}),
            buffer_type=bt,
            buffer_count=bc,
        )
        params = EnvParams()
        rng = jax.random.PRNGKey(0)

        state = factoriax_step(rng, state, Action.WITHDRAW, params)

        eidx = _ent_lookup(state, 1, 2)
        assert int(state.player_inventory[0, ItemType.IRON_ORE]) == 10
        assert int(state.ent_buf_count[eidx]) == 0
