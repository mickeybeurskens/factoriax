"""Tests for factoriax.play.transfer — item transfer between player and machine."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from factoriax.constants import (
    MAX_MACHINE_STACK_SIZE,
    MAX_STACK_SIZE,
    Action,
    ItemType,
    MachineType,
)
from factoriax.play.transfer import (
    deposit_to_machine,
    rotate_machine,
    swap_inventory_slots,
    withdraw_from_machine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state_with_machine(
    state_factory,
    machine_type: MachineType,
    *,
    m_items: list[int] | None = None,
    m_counts: list[int] | None = None,
    p_items: list[int] | None = None,
    p_counts: list[int] | None = None,
):
    """Build a 4×4 state with the given machine at (0, 0)."""
    from factoriax.constants import MAX_MACHINE_INVENTORY_SLOTS

    shape = (4, 4)
    machine_types = jnp.full(shape, int(machine_type), dtype=jnp.int32)

    inv_items = jnp.zeros((*shape, MAX_MACHINE_INVENTORY_SLOTS), dtype=jnp.int32)
    inv_counts = jnp.zeros((*shape, MAX_MACHINE_INVENTORY_SLOTS), dtype=jnp.int16)
    if m_items:
        for i, v in enumerate(m_items):
            inv_items = inv_items.at[0, 0, i].set(v)
    if m_counts:
        for i, v in enumerate(m_counts):
            inv_counts = inv_counts.at[0, 0, i].set(v)

    num_slots = 10
    pi = [0] * num_slots
    pc = [0] * num_slots
    if p_items:
        for i, v in enumerate(p_items):
            pi[i] = v
    if p_counts:
        for i, v in enumerate(p_counts):
            pc[i] = v

    return state_factory(
        world_map=jnp.zeros(shape, dtype=jnp.int32),
        machine_types=machine_types,
        machine_inventory_items=inv_items,
        machine_inventory_counts=inv_counts,
        inventory_items=jnp.array([pi], dtype=jnp.int32),
        inventory_counts=jnp.array([pc], dtype=jnp.int32),
    )


# ---------------------------------------------------------------------------
# withdraw_from_machine
# ---------------------------------------------------------------------------


class TestWithdraw:
    def test_noop_on_empty_slot(self, state_factory) -> None:
        """Withdraw from an empty slot returns state unchanged."""
        state = _state_with_machine(state_factory, MachineType.MINER)
        result = withdraw_from_machine(state, 0, 0, 0, 1)
        assert result is state

    def test_moves_stack_to_empty_player_inventory(self, state_factory) -> None:
        """Full machine stack transfers into the first empty player slot."""
        state = _state_with_machine(
            state_factory,
            MachineType.MINER,
            m_items=[0, int(ItemType.COAL)],
            m_counts=[0, 10],
        )
        result = withdraw_from_machine(state, 0, 0, 0, 1)
        assert int(result.machine_inventory_items[0, 0, 1]) == 0
        assert int(result.machine_inventory_counts[0, 0, 1]) == 0
        assert int(result.inventory_items[0, 0]) == int(ItemType.COAL)
        assert int(result.inventory_counts[0, 0]) == 10

    def test_merges_into_existing_matching_stack(self, state_factory) -> None:
        """Withdrawn items merge into a matching player stack before using empty slots."""
        state = _state_with_machine(
            state_factory,
            MachineType.MINER,
            m_items=[0, int(ItemType.COAL)],
            m_counts=[0, 5],
            p_items=[int(ItemType.COAL), 0],
            p_counts=[10, 0],
        )
        result = withdraw_from_machine(state, 0, 0, 0, 1)
        assert int(result.machine_inventory_counts[0, 0, 1]) == 0
        assert int(result.inventory_counts[0, 0]) == 15
        assert int(result.inventory_items[0, 1]) == 0

    def test_remainder_stays_in_machine_when_inventory_full(
        self, state_factory
    ) -> None:
        """When the player inventory is full, the remainder is returned to the machine."""
        full_items = [int(ItemType.IRON)] * 10
        full_counts = [MAX_STACK_SIZE] * 10
        state = _state_with_machine(
            state_factory,
            MachineType.MINER,
            m_items=[0, int(ItemType.COAL)],
            m_counts=[0, 7],
            p_items=full_items,
            p_counts=full_counts,
        )
        result = withdraw_from_machine(state, 0, 0, 0, 1)
        assert int(result.machine_inventory_items[0, 0, 1]) == int(ItemType.COAL)
        assert int(result.machine_inventory_counts[0, 0, 1]) == 7

    def test_partial_merge_uses_empty_slot_for_overflow(self, state_factory) -> None:
        """Items first merge into matching stack, then overflow into an empty slot."""
        state = _state_with_machine(
            state_factory,
            MachineType.MINER,
            m_items=[0, int(ItemType.COAL)],
            m_counts=[0, 20],
            p_items=[int(ItemType.COAL), 0],
            p_counts=[MAX_STACK_SIZE - 5, 0],
        )
        result = withdraw_from_machine(state, 0, 0, 0, 1)
        assert int(result.machine_inventory_counts[0, 0, 1]) == 0
        assert int(result.inventory_counts[0, 0]) == MAX_STACK_SIZE
        assert int(result.inventory_items[0, 1]) == int(ItemType.COAL)
        assert int(result.inventory_counts[0, 1]) == 15

    def test_clears_item_type_on_full_withdrawal(self, state_factory) -> None:
        """Machine slot item type is reset to 0 after a full withdrawal."""
        state = _state_with_machine(
            state_factory,
            MachineType.MINER,
            m_items=[0, int(ItemType.IRON)],
            m_counts=[0, 3],
        )
        result = withdraw_from_machine(state, 0, 0, 0, 1)
        assert int(result.machine_inventory_items[0, 0, 1]) == 0


# ---------------------------------------------------------------------------
# deposit_to_machine
# ---------------------------------------------------------------------------


class TestDeposit:
    def test_noop_on_empty_player_slot(self, state_factory) -> None:
        """Depositing from an empty player slot returns state unchanged."""
        state = _state_with_machine(state_factory, MachineType.MINER)
        result = deposit_to_machine(state, 0, 0, 0, 0, 0)
        assert result is state

    def test_deposits_into_focused_input_slot(self, state_factory) -> None:
        """Player items move into the focused INPUT slot (miner slot 0 = fuel)."""
        state = _state_with_machine(
            state_factory,
            MachineType.MINER,
            p_items=[int(ItemType.COAL)],
            p_counts=[5],
        )
        result = deposit_to_machine(state, 0, 0, 0, 0, 0)
        assert int(result.machine_inventory_items[0, 0, 0]) == int(ItemType.COAL)
        assert int(result.machine_inventory_counts[0, 0, 0]) == 5
        assert int(result.inventory_counts[0, 0]) == 0
        assert int(result.inventory_items[0, 0]) == 0

    def test_cannot_deposit_into_output_slot(self, state_factory) -> None:
        """Depositing into an OUTPUT slot is blocked; auto-route finds INPUT."""
        state = _state_with_machine(
            state_factory,
            MachineType.MINER,
            p_items=[int(ItemType.COAL)],
            p_counts=[5],
        )
        # Miner slot 1 is OUTPUT; should auto-route to slot 0 (INPUT).
        result = deposit_to_machine(state, 0, 0, 0, 1, 0)
        assert int(result.machine_inventory_items[0, 0, 0]) == int(ItemType.COAL)
        assert int(result.machine_inventory_counts[0, 0, 0]) == 5

    def test_merges_into_matching_focused_slot(self, state_factory) -> None:
        """Items merge into a focused slot already holding the same item."""
        state = _state_with_machine(
            state_factory,
            MachineType.CHEST,
            m_items=[int(ItemType.IRON)],
            m_counts=[10],
            p_items=[int(ItemType.IRON)],
            p_counts=[15],
        )
        result = deposit_to_machine(state, 0, 0, 0, 0, 0)
        assert int(result.machine_inventory_counts[0, 0, 0]) == 25
        assert int(result.inventory_counts[0, 0]) == 0

    def test_auto_routes_to_matching_slot_when_focused_has_wrong_item(
        self, state_factory
    ) -> None:
        """Auto-routes to a compatible slot holding the same item."""
        state = _state_with_machine(
            state_factory,
            MachineType.CHEST,
            m_items=[int(ItemType.IRON), int(ItemType.COAL)],
            m_counts=[5, 3],
            p_items=[int(ItemType.COAL)],
            p_counts=[8],
        )
        # Focused on slot 0 (holds IRON); auto-routes to slot 1 (holds COAL).
        result = deposit_to_machine(state, 0, 0, 0, 0, 0)
        assert int(result.machine_inventory_counts[0, 0, 1]) == 11
        assert int(result.inventory_counts[0, 0]) == 0

    def test_auto_routes_to_empty_slot_when_no_match(self, state_factory) -> None:
        """Auto-routes to the first empty INPUT/STORAGE slot if no match found."""
        state = _state_with_machine(
            state_factory,
            MachineType.CHEST,
            m_items=[int(ItemType.IRON)],
            m_counts=[5],
            p_items=[int(ItemType.COPPER)],
            p_counts=[4],
        )
        # Focused on slot 0 (IRON); slot 1 is empty STORAGE — auto-route there.
        result = deposit_to_machine(state, 0, 0, 0, 0, 0)
        assert int(result.machine_inventory_items[0, 0, 1]) == int(ItemType.COPPER)
        assert int(result.machine_inventory_counts[0, 0, 1]) == 4
        assert int(result.inventory_counts[0, 0]) == 0

    def test_noop_when_no_compatible_slot(self, state_factory) -> None:
        """Returns state unchanged when no machine slot can accept the item."""
        # Fill all CHEST slots with a different item.
        from factoriax.constants import MACHINE_NUM_SLOTS

        num = int(MACHINE_NUM_SLOTS[MachineType.CHEST])
        state = _state_with_machine(
            state_factory,
            MachineType.CHEST,
            m_items=[int(ItemType.IRON)] * num,
            m_counts=[MAX_MACHINE_STACK_SIZE] * num,
            p_items=[int(ItemType.COAL)],
            p_counts=[5],
        )
        result = deposit_to_machine(state, 0, 0, 0, 0, 0)
        assert result is state

    def test_partial_transfer_when_slot_nearly_full(self, state_factory) -> None:
        """Only fills the available space; remainder stays in player slot."""
        state = _state_with_machine(
            state_factory,
            MachineType.CHEST,
            m_items=[int(ItemType.COAL)],
            m_counts=[MAX_MACHINE_STACK_SIZE - 3],
            p_items=[int(ItemType.COAL)],
            p_counts=[10],
        )
        result = deposit_to_machine(state, 0, 0, 0, 0, 0)
        assert int(result.machine_inventory_counts[0, 0, 0]) == MAX_MACHINE_STACK_SIZE
        assert int(result.inventory_counts[0, 0]) == 7


# ---------------------------------------------------------------------------
# swap_inventory_slots
# ---------------------------------------------------------------------------


class TestSwapInventorySlots:
    """Tests for the swap_inventory_slots helper."""

    def test_swap_two_occupied_slots(self, state_factory) -> None:
        """Swapping two slots exchanges their items and counts."""
        state = _state_with_machine(
            state_factory,
            MachineType.CHEST,
            p_items=[int(ItemType.COAL), int(ItemType.IRON)],
            p_counts=[5, 10],
        )
        result = swap_inventory_slots(state, 0, 0, 1)
        assert int(result.inventory_items[0, 0]) == int(ItemType.IRON)
        assert int(result.inventory_counts[0, 0]) == 10
        assert int(result.inventory_items[0, 1]) == int(ItemType.COAL)
        assert int(result.inventory_counts[0, 1]) == 5

    def test_swap_occupied_with_empty(self, state_factory) -> None:
        """Swapping an occupied slot with an empty one moves the item."""
        state = _state_with_machine(
            state_factory,
            MachineType.CHEST,
            p_items=[int(ItemType.COPPER)],
            p_counts=[3],
        )
        result = swap_inventory_slots(state, 0, 0, 4)
        assert int(result.inventory_items[0, 0]) == 0
        assert int(result.inventory_counts[0, 0]) == 0
        assert int(result.inventory_items[0, 4]) == int(ItemType.COPPER)
        assert int(result.inventory_counts[0, 4]) == 3

    def test_same_slot_is_noop(self, state_factory) -> None:
        """Swapping a slot with itself returns the same state object."""
        state = _state_with_machine(
            state_factory,
            MachineType.CHEST,
            p_items=[int(ItemType.COAL)],
            p_counts=[5],
        )
        result = swap_inventory_slots(state, 0, 0, 0)
        assert result is state

    def test_swap_two_empty_slots(self, state_factory) -> None:
        """Swapping two empty slots leaves both empty."""
        state = _state_with_machine(
            state_factory, MachineType.CHEST,
        )
        result = swap_inventory_slots(state, 0, 2, 7)
        assert int(result.inventory_items[0, 2]) == 0
        assert int(result.inventory_items[0, 7]) == 0

    def test_merge_same_item_type(self, state_factory) -> None:
        """Moving onto the same item type merges stacks."""
        state = _state_with_machine(
            state_factory,
            MachineType.CHEST,
            p_items=[int(ItemType.COAL), int(ItemType.COAL)],
            p_counts=[10, 20],
        )
        result = swap_inventory_slots(state, 0, 0, 1)
        assert int(result.inventory_items[0, 1]) == int(ItemType.COAL)
        assert int(result.inventory_counts[0, 1]) == 30
        assert int(result.inventory_items[0, 0]) == 0
        assert int(result.inventory_counts[0, 0]) == 0

    def test_merge_overflow_stays_in_source(self, state_factory) -> None:
        """When the destination is nearly full, overflow stays in source."""
        state = _state_with_machine(
            state_factory,
            MachineType.CHEST,
            p_items=[int(ItemType.IRON), int(ItemType.IRON)],
            p_counts=[30, MAX_STACK_SIZE - 10],
        )
        result = swap_inventory_slots(state, 0, 0, 1)
        assert int(result.inventory_counts[0, 1]) == MAX_STACK_SIZE
        assert int(result.inventory_items[0, 0]) == int(ItemType.IRON)
        assert int(result.inventory_counts[0, 0]) == 20

    def test_merge_destination_already_full(self, state_factory) -> None:
        """Merging onto a full stack transfers nothing."""
        state = _state_with_machine(
            state_factory,
            MachineType.CHEST,
            p_items=[int(ItemType.COAL), int(ItemType.COAL)],
            p_counts=[5, MAX_STACK_SIZE],
        )
        result = swap_inventory_slots(state, 0, 0, 1)
        assert int(result.inventory_counts[0, 0]) == 5
        assert int(result.inventory_counts[0, 1]) == MAX_STACK_SIZE


# ---------------------------------------------------------------------------
# rotate_machine
# ---------------------------------------------------------------------------


class TestRotateMachine:
    """Tests for the rotate_machine helper."""

    def test_cycles_direction_clockwise(self, state_factory) -> None:
        """Each call advances direction one step in the clockwise cycle."""
        shape = (4, 4)
        dirs = jnp.full(shape, int(Action.DOWN), dtype=jnp.int32)
        state = state_factory(
            world_map=jnp.zeros(shape, dtype=jnp.int32),
            machine_types=jnp.full(
                shape, int(MachineType.CONVEYOR_BELT), dtype=jnp.int32
            ),
            machine_direction=dirs,
        )
        # DOWN -> RIGHT -> UP -> LEFT -> DOWN
        expected = [
            int(Action.RIGHT),
            int(Action.UP),
            int(Action.LEFT),
            int(Action.DOWN),
        ]
        for exp in expected:
            state = rotate_machine(state, 0, 0)
            assert int(state.machine_direction[0, 0]) == exp

    def test_noop_on_empty_tile(self, state_factory) -> None:
        """Rotating a tile with no machine returns state unchanged."""
        shape = (4, 4)
        state = state_factory(
            world_map=jnp.zeros(shape, dtype=jnp.int32),
        )
        result = rotate_machine(state, 1, 1)
        assert result is state

    def test_does_not_affect_other_tiles(self, state_factory) -> None:
        """Rotation only changes the targeted tile's direction."""
        shape = (4, 4)
        dirs = jnp.full(shape, int(Action.DOWN), dtype=jnp.int32)
        state = state_factory(
            world_map=jnp.zeros(shape, dtype=jnp.int32),
            machine_types=jnp.full(
                shape, int(MachineType.CONVEYOR_BELT), dtype=jnp.int32
            ),
            machine_direction=dirs,
        )
        result = rotate_machine(state, 1, 1)
        assert int(result.machine_direction[1, 1]) == int(Action.RIGHT)
        assert int(result.machine_direction[0, 0]) == int(Action.DOWN)
        assert int(result.machine_direction[2, 3]) == int(Action.DOWN)
