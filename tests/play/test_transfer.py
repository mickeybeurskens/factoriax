"""Tests for factoriax.play.transfer — inventory slot swapping."""

from __future__ import annotations

import jax.numpy as jnp

from factoriax.constants import (
    MAX_STACK_SIZE,
    ItemType,
    MachineType,
)
from factoriax.play.transfer import swap_inventory_slots

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state_with_machine(
    state_factory,
    machine_type: MachineType,
    *,
    p_items: list[int] | None = None,
    p_counts: list[int] | None = None,
):
    """Build a 4x4 state with a machine at every tile (for test convenience)."""
    from factoriax.constants import MAX_MACHINE_INVENTORY_SLOTS

    shape = (4, 4)
    machine_types = jnp.full(shape, int(machine_type), dtype=jnp.int32)

    inv_items = jnp.zeros((*shape, MAX_MACHINE_INVENTORY_SLOTS), dtype=jnp.int32)
    inv_counts = jnp.zeros((*shape, MAX_MACHINE_INVENTORY_SLOTS), dtype=jnp.int16)

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
