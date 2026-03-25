"""Tests for the DEPOSIT and WITHDRAW RL actions.

Covers deposit into chests, miners (fuel slot), and assemblers (recipe-filtered
inputs), as well as withdraw with OUTPUT > STORAGE > INPUT priority ordering.
"""

from __future__ import annotations

import jax.numpy as jnp

from factoriax import Action, BlockType, ItemType
from factoriax.constants import (
    MAX_MACHINE_INVENTORY_SLOTS,
    MAX_MACHINE_STACK_SIZE,
    NUM_INVENTORY_SLOTS,
    MachineType,
)
from factoriax.game_logic import deposit_to_adjacent, withdraw_from_adjacent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIRT_3X3 = jnp.full((3, 3), BlockType.DIRT, dtype=jnp.int32)


def _machine_types(w: int, h: int, placements: dict[tuple[int, int], int]):
    """Build a machine_types grid with specific placements."""
    arr = jnp.full((h, w), MachineType.NONE, dtype=jnp.int32)
    for (x, y), mtype in placements.items():
        arr = arr.at[y, x].set(mtype)
    return arr


def _machine_inv(
    w: int,
    h: int,
    items: dict[tuple[int, int, int], int] | None = None,
    counts: dict[tuple[int, int, int], int] | None = None,
):
    """Build machine inventory arrays. Keys are (x, y, slot)."""
    item_arr = jnp.zeros((h, w, MAX_MACHINE_INVENTORY_SLOTS), dtype=jnp.int32)
    count_arr = jnp.zeros((h, w, MAX_MACHINE_INVENTORY_SLOTS), dtype=jnp.int16)
    for (x, y, slot), val in (items or {}).items():
        item_arr = item_arr.at[y, x, slot].set(val)
    for (x, y, slot), val in (counts or {}).items():
        count_arr = count_arr.at[y, x, slot].set(val)
    return item_arr, count_arr


def _player_inv(slot_data: dict[int, tuple[int, int]]):
    """Build player inventory arrays from {slot: (item_type, count)}.

    Returns (items, counts) each of shape (1, NUM_INVENTORY_SLOTS).
    """
    items = jnp.zeros((1, NUM_INVENTORY_SLOTS), dtype=jnp.int32)
    counts = jnp.zeros((1, NUM_INVENTORY_SLOTS), dtype=jnp.int32)
    for slot, (item, count) in slot_data.items():
        items = items.at[0, slot].set(item)
        counts = counts.at[0, slot].set(count)
    return items, counts


# ===========================================================================
# DEPOSIT tests
# ===========================================================================


class TestDepositToChest:
    """Deposit items from player inventory into a chest."""

    def test_deposit_coal_into_empty_chest(self, state_factory) -> None:
        """Full coal stack should transfer into the first chest slot."""
        inv_items, inv_counts = _player_inv({0: (ItemType.COAL, 10)})
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Action.RIGHT,
            inventory_items=inv_items,
            inventory_counts=inv_counts,
            machine_types=_machine_types(3, 3, {(2, 1): MachineType.CHEST}),
        )

        state = deposit_to_adjacent(state, 0)

        assert int(state.inventory_counts[0, 0]) == 0
        assert int(state.inventory_items[0, 0]) == ItemType.EMPTY
        assert int(state.machine_inventory_items[1, 2, 0]) == ItemType.COAL
        assert int(state.machine_inventory_counts[1, 2, 0]) == 10

    def test_deposit_stacks_into_matching_slot(self, state_factory) -> None:
        """Depositing coal should merge with an existing coal stack."""
        inv_items, inv_counts = _player_inv({0: (ItemType.COAL, 5)})
        m_items, m_counts = _machine_inv(
            3,
            3,
            items={(2, 1, 0): ItemType.COAL},
            counts={(2, 1, 0): 10},
        )
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Action.RIGHT,
            inventory_items=inv_items,
            inventory_counts=inv_counts,
            machine_types=_machine_types(3, 3, {(2, 1): MachineType.CHEST}),
            machine_inventory_items=m_items,
            machine_inventory_counts=m_counts,
        )

        state = deposit_to_adjacent(state, 0)

        assert int(state.inventory_counts[0, 0]) == 0
        assert int(state.machine_inventory_counts[1, 2, 0]) == 15

    def test_deposit_respects_stack_cap(self, state_factory) -> None:
        """Deposit should cap at MAX_MACHINE_STACK_SIZE, leaving remainder."""
        inv_items, inv_counts = _player_inv({0: (ItemType.COAL, 20)})
        m_items, m_counts = _machine_inv(
            3,
            3,
            items={(2, 1, 0): ItemType.COAL},
            counts={(2, 1, 0): MAX_MACHINE_STACK_SIZE - 5},
        )
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Action.RIGHT,
            inventory_items=inv_items,
            inventory_counts=inv_counts,
            machine_types=_machine_types(3, 3, {(2, 1): MachineType.CHEST}),
            machine_inventory_items=m_items,
            machine_inventory_counts=m_counts,
        )

        state = deposit_to_adjacent(state, 0)

        assert int(state.machine_inventory_counts[1, 2, 0]) == (MAX_MACHINE_STACK_SIZE)
        assert int(state.inventory_counts[0, 0]) == 15

    def test_deposit_noop_no_machine(self, state_factory) -> None:
        """Deposit into empty tile should be a no-op."""
        inv_items, inv_counts = _player_inv({0: (ItemType.COAL, 10)})
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Action.RIGHT,
            inventory_items=inv_items,
            inventory_counts=inv_counts,
        )

        state = deposit_to_adjacent(state, 0)

        assert int(state.inventory_counts[0, 0]) == 10

    def test_deposit_noop_empty_slot(self, state_factory) -> None:
        """Deposit with empty selected slot should be a no-op."""
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Action.RIGHT,
            machine_types=_machine_types(3, 3, {(2, 1): MachineType.CHEST}),
        )

        state = deposit_to_adjacent(state, 0)

        assert int(state.machine_inventory_counts[1, 2, 0]) == 0

    def test_deposit_noop_out_of_bounds(self, state_factory) -> None:
        """Deposit facing out of bounds should be a no-op."""
        inv_items, inv_counts = _player_inv({0: (ItemType.COAL, 10)})
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(2, 1),
            player_direction=Action.RIGHT,
            inventory_items=inv_items,
            inventory_counts=inv_counts,
        )

        state = deposit_to_adjacent(state, 0)

        assert int(state.inventory_counts[0, 0]) == 10


class TestDepositToMiner:
    """Deposit coal into a miner's fuel slot (INPUT, slot 0)."""

    def test_deposit_coal_as_fuel(self, state_factory) -> None:
        """Coal deposited into a miner should go to slot 0 (fuel INPUT)."""
        inv_items, inv_counts = _player_inv({0: (ItemType.COAL, 5)})
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Action.RIGHT,
            inventory_items=inv_items,
            inventory_counts=inv_counts,
            machine_types=_machine_types(3, 3, {(2, 1): MachineType.MINER}),
        )

        state = deposit_to_adjacent(state, 0)

        assert int(state.machine_inventory_items[1, 2, 0]) == ItemType.COAL
        assert int(state.machine_inventory_counts[1, 2, 0]) == 5
        assert int(state.inventory_counts[0, 0]) == 0

    def test_deposit_cannot_go_to_output_slot(self, state_factory) -> None:
        """Iron deposited into a miner should not go to slot 1 (OUTPUT).

        The miner only has slot 0 (INPUT) and slot 1 (OUTPUT). Depositing
        iron makes no sense (it's not fuel), but even if slot 0 is full,
        the deposit must not route to the OUTPUT slot.
        """
        inv_items, inv_counts = _player_inv({0: (ItemType.IRON, 5)})
        m_items, m_counts = _machine_inv(
            3,
            3,
            items={(2, 1, 0): ItemType.COAL},
            counts={(2, 1, 0): MAX_MACHINE_STACK_SIZE},
        )
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Action.RIGHT,
            inventory_items=inv_items,
            inventory_counts=inv_counts,
            machine_types=_machine_types(3, 3, {(2, 1): MachineType.MINER}),
            machine_inventory_items=m_items,
            machine_inventory_counts=m_counts,
        )

        state = deposit_to_adjacent(state, 0)

        # Iron stays in player inventory, miner output slot untouched.
        assert int(state.inventory_counts[0, 0]) == 5
        assert int(state.machine_inventory_counts[1, 2, 1]) == 0


class TestDepositToAssembler:
    """Deposit items into an assembler with recipe filtering."""

    def test_deposit_iron_into_hull_assembler(self, state_factory) -> None:
        """Iron should go into slot 0 (Hull recipe needs 5 iron)."""
        inv_items, inv_counts = _player_inv({0: (ItemType.IRON, 10)})
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Action.RIGHT,
            inventory_items=inv_items,
            inventory_counts=inv_counts,
            machine_types=_machine_types(3, 3, {(2, 1): MachineType.ASSEMBLER}),
            machine_selected_recipe=jnp.zeros((3, 3), dtype=jnp.int32),
        )

        state = deposit_to_adjacent(state, 0)

        assert int(state.machine_inventory_items[1, 2, 0]) == ItemType.IRON
        assert int(state.machine_inventory_counts[1, 2, 0]) == 10
        assert int(state.inventory_counts[0, 0]) == 0

    def test_deposit_wrong_item_rejected(self, state_factory) -> None:
        """Copper should be rejected by a Hull assembler (needs only iron)."""
        inv_items, inv_counts = _player_inv({0: (ItemType.COPPER, 10)})
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Action.RIGHT,
            inventory_items=inv_items,
            inventory_counts=inv_counts,
            machine_types=_machine_types(3, 3, {(2, 1): MachineType.ASSEMBLER}),
            machine_selected_recipe=jnp.zeros((3, 3), dtype=jnp.int32),
        )

        state = deposit_to_adjacent(state, 0)

        assert int(state.inventory_counts[0, 0]) == 10
        assert int(state.machine_inventory_counts[1, 2, 0]) == 0

    def test_deposit_fuel_pack_recipe_both_inputs(self, state_factory) -> None:
        """Fuel Pack recipe needs copper (slot 0) and coal (slot 1)."""
        # Deposit copper first.
        inv_items, inv_counts = _player_inv({0: (ItemType.COPPER, 3)})
        recipe = jnp.zeros((3, 3), dtype=jnp.int32).at[1, 2].set(1)
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Action.RIGHT,
            inventory_items=inv_items,
            inventory_counts=inv_counts,
            machine_types=_machine_types(3, 3, {(2, 1): MachineType.ASSEMBLER}),
            machine_selected_recipe=recipe,
        )
        state = deposit_to_adjacent(state, 0)
        assert int(state.machine_inventory_items[1, 2, 0]) == ItemType.COPPER
        assert int(state.machine_inventory_counts[1, 2, 0]) == 3

        # Now deposit coal (should go to slot 1).
        state = state.replace(
            inventory_items=state.inventory_items.at[0, 0].set(ItemType.COAL),
            inventory_counts=state.inventory_counts.at[0, 0].set(2),
        )
        state = deposit_to_adjacent(state, 0)
        assert int(state.machine_inventory_items[1, 2, 1]) == ItemType.COAL
        assert int(state.machine_inventory_counts[1, 2, 1]) == 2


class TestDepositSelectedSlot:
    """Deposit uses the player's currently selected inventory slot."""

    def test_deposit_from_non_zero_slot(self, state_factory) -> None:
        """Selecting slot 2 should deposit from slot 2, not slot 0."""
        inv_items, inv_counts = _player_inv(
            {0: (ItemType.IRON, 5), 2: (ItemType.COAL, 8)}
        )
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Action.RIGHT,
            inventory_items=inv_items,
            inventory_counts=inv_counts,
            selected_slots=jnp.array([2], dtype=jnp.int32),
            machine_types=_machine_types(3, 3, {(2, 1): MachineType.CHEST}),
        )

        state = deposit_to_adjacent(state, 0)

        # Slot 2 (coal) should be deposited, slot 0 (iron) untouched.
        assert int(state.inventory_counts[0, 0]) == 5
        assert int(state.inventory_counts[0, 2]) == 0
        assert int(state.machine_inventory_items[1, 2, 0]) == ItemType.COAL
        assert int(state.machine_inventory_counts[1, 2, 0]) == 8


# ===========================================================================
# WITHDRAW tests
# ===========================================================================


class TestWithdrawFromChest:
    """Withdraw items from a chest into player inventory."""

    def test_withdraw_from_chest(self, state_factory) -> None:
        """Should take items from the first occupied chest slot."""
        m_items, m_counts = _machine_inv(
            3,
            3,
            items={(2, 1, 0): ItemType.IRON},
            counts={(2, 1, 0): 10},
        )
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Action.RIGHT,
            machine_types=_machine_types(3, 3, {(2, 1): MachineType.CHEST}),
            machine_inventory_items=m_items,
            machine_inventory_counts=m_counts,
        )

        state = withdraw_from_adjacent(state, 0)

        assert int(state.inventory_items[0, 0]) == ItemType.IRON
        assert int(state.inventory_counts[0, 0]) == 10
        assert int(state.machine_inventory_counts[1, 2, 0]) == 0

    def test_withdraw_noop_empty_machine(self, state_factory) -> None:
        """Withdraw from empty chest should be a no-op."""
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Action.RIGHT,
            machine_types=_machine_types(3, 3, {(2, 1): MachineType.CHEST}),
        )

        state = withdraw_from_adjacent(state, 0)

        assert int(state.inventory_items[0, 0]) == ItemType.EMPTY

    def test_withdraw_noop_no_machine(self, state_factory) -> None:
        """Withdraw with no machine in front should be a no-op."""
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Action.RIGHT,
        )

        state = withdraw_from_adjacent(state, 0)

        assert int(state.inventory_items[0, 0]) == ItemType.EMPTY


class TestWithdrawPriority:
    """Withdraw scans OUTPUT before STORAGE before INPUT."""

    def test_miner_output_before_fuel(self, state_factory) -> None:
        """Miner slot 1 (OUTPUT) should be taken before slot 0 (INPUT)."""
        m_items, m_counts = _machine_inv(
            3,
            3,
            items={(2, 1, 0): ItemType.COAL, (2, 1, 1): ItemType.IRON},
            counts={(2, 1, 0): 5, (2, 1, 1): 3},
        )
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Action.RIGHT,
            machine_types=_machine_types(3, 3, {(2, 1): MachineType.MINER}),
            machine_inventory_items=m_items,
            machine_inventory_counts=m_counts,
        )

        state = withdraw_from_adjacent(state, 0)

        # Should take iron (OUTPUT slot 1), not coal (INPUT slot 0).
        assert int(state.inventory_items[0, 0]) == ItemType.IRON
        assert int(state.inventory_counts[0, 0]) == 3
        # Coal in fuel slot should be untouched.
        assert int(state.machine_inventory_counts[1, 2, 0]) == 5

    def test_assembler_output_before_input(self, state_factory) -> None:
        """Assembler slot 3 (OUTPUT) should be taken before slots 0-2 (INPUT)."""
        m_items, m_counts = _machine_inv(
            3,
            3,
            items={(2, 1, 0): ItemType.IRON, (2, 1, 3): ItemType.HULL},
            counts={(2, 1, 0): 5, (2, 1, 3): 2},
        )
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Action.RIGHT,
            machine_types=_machine_types(3, 3, {(2, 1): MachineType.ASSEMBLER}),
            machine_inventory_items=m_items,
            machine_inventory_counts=m_counts,
        )

        state = withdraw_from_adjacent(state, 0)

        # Should take hulls (OUTPUT slot 3), not iron (INPUT slot 0).
        assert int(state.inventory_items[0, 0]) == ItemType.HULL
        assert int(state.inventory_counts[0, 0]) == 2
        assert int(state.machine_inventory_counts[1, 2, 0]) == 5


class TestWithdrawMergesIntoInventory:
    """Withdrawn items should merge with existing player stacks."""

    def test_withdraw_merges_with_existing_stack(self, state_factory) -> None:
        """Items withdrawn should stack with matching items in inventory."""
        inv_items, inv_counts = _player_inv({0: (ItemType.IRON, 3)})
        m_items, m_counts = _machine_inv(
            3,
            3,
            items={(2, 1, 0): ItemType.IRON},
            counts={(2, 1, 0): 7},
        )
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Action.RIGHT,
            inventory_items=inv_items,
            inventory_counts=inv_counts,
            machine_types=_machine_types(3, 3, {(2, 1): MachineType.CHEST}),
            machine_inventory_items=m_items,
            machine_inventory_counts=m_counts,
        )

        state = withdraw_from_adjacent(state, 0)

        assert int(state.inventory_items[0, 0]) == ItemType.IRON
        assert int(state.inventory_counts[0, 0]) == 10
        assert int(state.machine_inventory_counts[1, 2, 0]) == 0


class TestDepositWithdrawViaStep:
    """End-to-end tests dispatched through factoriax_step."""

    def test_deposit_via_step(self, state_factory) -> None:
        """DEPOSIT action through the full step pipeline."""
        import jax

        from factoriax.game_logic import factoriax_step
        from factoriax.state import EnvParams

        inv_items, inv_counts = _player_inv({0: (ItemType.COAL, 5)})
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Action.RIGHT,
            inventory_items=inv_items,
            inventory_counts=inv_counts,
            machine_types=_machine_types(3, 3, {(2, 1): MachineType.CHEST}),
        )
        params = EnvParams(map_width=3, map_height=3, num_players=1)
        rng = jax.random.PRNGKey(0)

        state = factoriax_step(rng, state, Action.DEPOSIT, params)

        assert int(state.machine_inventory_items[1, 2, 0]) == ItemType.COAL
        assert int(state.machine_inventory_counts[1, 2, 0]) == 5
        assert int(state.inventory_counts[0, 0]) == 0

    def test_withdraw_via_step(self, state_factory) -> None:
        """WITHDRAW action through the full step pipeline."""
        import jax

        from factoriax.game_logic import factoriax_step
        from factoriax.state import EnvParams

        m_items, m_counts = _machine_inv(
            3,
            3,
            items={(2, 1, 0): ItemType.IRON},
            counts={(2, 1, 0): 10},
        )
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Action.RIGHT,
            machine_types=_machine_types(3, 3, {(2, 1): MachineType.CHEST}),
            machine_inventory_items=m_items,
            machine_inventory_counts=m_counts,
        )
        params = EnvParams(map_width=3, map_height=3, num_players=1)
        rng = jax.random.PRNGKey(0)

        state = factoriax_step(rng, state, Action.WITHDRAW, params)

        assert int(state.inventory_items[0, 0]) == ItemType.IRON
        assert int(state.inventory_counts[0, 0]) == 10
        assert int(state.machine_inventory_counts[1, 2, 0]) == 0
