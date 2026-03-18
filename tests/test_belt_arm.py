"""Tests for conveyor belt and pick-and-place arm machine logic."""

from __future__ import annotations

import jax.numpy as jnp

from factoriax.constants import (
    MAX_MACHINE_INVENTORY_SLOTS,
    MAX_MACHINE_STACK_SIZE,
    Action,
    ItemType,
    MachineType,
)
from factoriax.machines import run_arms, run_conveyor_belts

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _belt_state(
    state_factory,
    *,
    machine_types: jnp.ndarray,
    machine_direction: jnp.ndarray,
    slot_items: jnp.ndarray | None = None,
    slot_counts: jnp.ndarray | None = None,
):
    """Build a state with the given belt grid and optional slot 0 contents."""
    shape = machine_types.shape
    inv_items = jnp.zeros((*shape, MAX_MACHINE_INVENTORY_SLOTS), dtype=jnp.int32)
    inv_counts = jnp.zeros((*shape, MAX_MACHINE_INVENTORY_SLOTS), dtype=jnp.int16)
    if slot_items is not None:
        inv_items = inv_items.at[..., 0].set(slot_items)
    if slot_counts is not None:
        inv_counts = inv_counts.at[..., 0].set(
            jnp.asarray(slot_counts, dtype=jnp.int16)
        )
    return state_factory(
        world_map=jnp.zeros(shape, dtype=jnp.int32),
        machine_types=machine_types,
        machine_direction=machine_direction,
        machine_inventory_items=inv_items,
        machine_inventory_counts=inv_counts,
    )


def _arm_state(
    state_factory,
    *,
    machine_types: jnp.ndarray,
    machine_direction: jnp.ndarray,
    inv_items: jnp.ndarray,
    inv_counts: jnp.ndarray,
):
    """Build a state with the given machine grid and full inventory arrays."""
    return state_factory(
        world_map=jnp.zeros(machine_types.shape, dtype=jnp.int32),
        machine_types=machine_types,
        machine_direction=machine_direction,
        machine_inventory_items=inv_items,
        machine_inventory_counts=inv_counts,
    )


def _inv(
    shape,
    items_slot0=None,
    counts_slot0=None,
    items_slot1=None,
    counts_slot1=None,
):
    """Build machine inventory arrays with values in specific slots."""
    inv_items = jnp.zeros((*shape, MAX_MACHINE_INVENTORY_SLOTS), dtype=jnp.int32)
    inv_counts = jnp.zeros((*shape, MAX_MACHINE_INVENTORY_SLOTS), dtype=jnp.int16)
    if items_slot0 is not None:
        inv_items = inv_items.at[..., 0].set(items_slot0)
    if counts_slot0 is not None:
        inv_counts = inv_counts.at[..., 0].set(
            jnp.asarray(counts_slot0, dtype=jnp.int16)
        )
    if items_slot1 is not None:
        inv_items = inv_items.at[..., 1].set(items_slot1)
    if counts_slot1 is not None:
        inv_counts = inv_counts.at[..., 1].set(
            jnp.asarray(counts_slot1, dtype=jnp.int16)
        )
    return inv_items, inv_counts


# ---------------------------------------------------------------------------
# Conveyor belt tests
# ---------------------------------------------------------------------------


class TestConveyorBelt:
    def test_noop_when_no_items(self, state_factory) -> None:
        """Belt with empty slot does nothing."""
        types = jnp.array([[MachineType.CONVEYOR_BELT, MachineType.CONVEYOR_BELT]])
        dirs = jnp.array([[Action.RIGHT, Action.RIGHT]])
        state = _belt_state(state_factory, machine_types=types, machine_direction=dirs)
        result = run_conveyor_belts(state)
        assert int(result.machine_inventory_counts[0, 0, 0]) == 0
        assert int(result.machine_inventory_counts[0, 1, 0]) == 0

    def test_pushes_item_right(self, state_factory) -> None:
        """Belt facing right moves items from slot (0,0) to (0,1)."""
        types = jnp.array([[MachineType.CONVEYOR_BELT, MachineType.CONVEYOR_BELT]])
        dirs = jnp.array([[Action.RIGHT, Action.RIGHT]])
        items = jnp.array([[int(ItemType.COAL), 0]])
        counts = jnp.array([[10, 0]])
        state = _belt_state(
            state_factory,
            machine_types=types,
            machine_direction=dirs,
            slot_items=items,
            slot_counts=counts,
        )
        result = run_conveyor_belts(state)
        assert int(result.machine_inventory_counts[0, 0, 0]) == 0
        assert int(result.machine_inventory_counts[0, 1, 0]) == 10
        assert int(result.machine_inventory_items[0, 1, 0]) == int(ItemType.COAL)

    def test_pushes_item_down(self, state_factory) -> None:
        """Belt facing down moves items from row 0 to row 1."""
        types = jnp.array(
            [[MachineType.CONVEYOR_BELT], [MachineType.CONVEYOR_BELT]]
        )
        dirs = jnp.array([[Action.DOWN], [Action.DOWN]])
        items = jnp.array([[int(ItemType.IRON)], [0]])
        counts = jnp.array([[5], [0]])
        state = _belt_state(
            state_factory,
            machine_types=types,
            machine_direction=dirs,
            slot_items=items,
            slot_counts=counts,
        )
        result = run_conveyor_belts(state)
        assert int(result.machine_inventory_counts[0, 0, 0]) == 0
        assert int(result.machine_inventory_counts[1, 0, 0]) == 5

    def test_does_not_push_to_non_belt(self, state_factory) -> None:
        """Belt adjacent to a non-belt machine does not transfer items."""
        types = jnp.array(
            [[MachineType.CONVEYOR_BELT, MachineType.CHEST]]
        )
        dirs = jnp.array([[Action.RIGHT, Action.RIGHT]])
        items = jnp.array([[int(ItemType.COAL), 0]])
        counts = jnp.array([[8, 0]])
        state = _belt_state(
            state_factory,
            machine_types=types,
            machine_direction=dirs,
            slot_items=items,
            slot_counts=counts,
        )
        result = run_conveyor_belts(state)
        assert int(result.machine_inventory_counts[0, 0, 0]) == 8
        assert int(result.machine_inventory_counts[0, 1, 0]) == 0

    def test_does_not_push_to_blocked_target(self, state_factory) -> None:
        """Belt does not push when target holds a different item at full count."""
        types = jnp.array([[MachineType.CONVEYOR_BELT, MachineType.CONVEYOR_BELT]])
        dirs = jnp.array([[Action.RIGHT, Action.RIGHT]])
        items = jnp.array([[int(ItemType.COAL), int(ItemType.IRON)]])
        counts = jnp.array([[5, MAX_MACHINE_STACK_SIZE]])
        state = _belt_state(
            state_factory,
            machine_types=types,
            machine_direction=dirs,
            slot_items=items,
            slot_counts=counts,
        )
        result = run_conveyor_belts(state)
        assert int(result.machine_inventory_counts[0, 0, 0]) == 5

    def test_merges_same_item_into_target(self, state_factory) -> None:
        """Items of the same type merge into the target's existing stack."""
        types = jnp.array([[MachineType.CONVEYOR_BELT, MachineType.CONVEYOR_BELT]])
        dirs = jnp.array([[Action.RIGHT, Action.RIGHT]])
        items = jnp.array([[int(ItemType.COAL), int(ItemType.COAL)]])
        counts = jnp.array([[3, 7]])
        state = _belt_state(
            state_factory,
            machine_types=types,
            machine_direction=dirs,
            slot_items=items,
            slot_counts=counts,
        )
        result = run_conveyor_belts(state)
        assert int(result.machine_inventory_counts[0, 1, 0]) == 10
        assert int(result.machine_inventory_counts[0, 0, 0]) == 0

    def test_noop_with_zero_direction(self, state_factory) -> None:
        """Belt with NOOP direction (0) does not push to self."""
        types = jnp.array([[MachineType.CONVEYOR_BELT]])
        dirs = jnp.array([[0]])  # NOOP
        items = jnp.array([[int(ItemType.COAL)]])
        counts = jnp.array([[5]])
        state = _belt_state(
            state_factory,
            machine_types=types,
            machine_direction=dirs,
            slot_items=items,
            slot_counts=counts,
        )
        result = run_conveyor_belts(state)
        assert int(result.machine_inventory_counts[0, 0, 0]) == 5


# ---------------------------------------------------------------------------
# Pick-and-place arm tests
# ---------------------------------------------------------------------------


class TestArm:
    def test_picks_from_miner_output(self, state_factory) -> None:
        """Arm picks item from miner's output slot (slot 1, OUTPUT role)."""
        # Layout: [MINER, ARM] — arm faces LEFT (picks from (0,0) miner)
        # Miner slot 1 = OUTPUT, has COAL
        shape = (1, 2)
        types = jnp.array([[MachineType.MINER, MachineType.ARM]])
        # Arm facing RIGHT: bwd=(0,0) MINER, fwd=(0,2) clipped.
        dirs = jnp.array([[0, int(Action.RIGHT)]])
        inv_items, inv_counts = _inv(
            shape,
            items_slot1=jnp.array([[int(ItemType.COAL), 0]]),
            counts_slot1=jnp.array([[15, 0]]),
        )
        state = _arm_state(
            state_factory,
            machine_types=types,
            machine_direction=dirs,
            inv_items=inv_items,
            inv_counts=inv_counts,
        )
        result = run_arms(state)
        # ARM buffer should now hold COAL
        assert int(result.machine_inventory_items[0, 1, 0]) == int(ItemType.COAL)
        assert int(result.machine_inventory_counts[0, 1, 0]) == 15
        # Miner output slot should be empty
        assert int(result.machine_inventory_counts[0, 0, 1]) == 0

    def test_deposits_to_chest(self, state_factory) -> None:
        """Arm with full buffer deposits items into adjacent chest."""
        # Layout: [CHEST, ARM] — arm faces RIGHT (fwd=OOB→self, bwd=CHEST)
        # Actually arm facing LEFT: fwd=(0,0) CHEST, bwd=(0,2)→clipped=(0,1)=self
        # So arm at (0,1) facing LEFT: forward=(0,0), not_self=True → deposits to chest
        shape = (1, 2)
        types = jnp.array([[MachineType.CHEST, MachineType.ARM]])
        dirs = jnp.array([[0, int(Action.LEFT)]])
        inv_items, inv_counts = _inv(
            shape,
            items_slot0=jnp.array([[0, int(ItemType.IRON)]]),
            counts_slot0=jnp.array([[0, 20]]),
        )
        state = _arm_state(
            state_factory,
            machine_types=types,
            machine_direction=dirs,
            inv_items=inv_items,
            inv_counts=inv_counts,
        )
        result = run_arms(state)
        # ARM buffer cleared
        assert int(result.machine_inventory_counts[0, 1, 0]) == 0
        # Chest slot 0 (STORAGE) received items
        assert int(result.machine_inventory_items[0, 0, 0]) == int(ItemType.IRON)
        assert int(result.machine_inventory_counts[0, 0, 0]) == 20

    def test_deposit_then_pick_same_tick(self, state_factory) -> None:
        """Arm deposits first, then picks in same tick if buffer is now empty."""
        # Layout: [CHEST_with_IRON, ARM, MINER_with_COAL]
        # Arm at (0,1) facing LEFT: fwd=(0,0) CHEST, bwd=(0,2) MINER
        # ARM buffer has items → deposits to CHEST first,
        # then buffer empty → picks from MINER output
        shape = (1, 3)
        types = jnp.array(
            [[MachineType.CHEST, MachineType.ARM, MachineType.MINER]]
        )
        dirs = jnp.array([[0, int(Action.LEFT), 0]])
        inv_items, inv_counts = _inv(
            shape,
            # ARM buffer slot 0 has IRON
            items_slot0=jnp.array([[0, int(ItemType.IRON), 0]]),
            counts_slot0=jnp.array([[0, 8, 0]]),
            # MINER output slot 1 has COAL
            items_slot1=jnp.array([[0, 0, int(ItemType.COAL)]]),
            counts_slot1=jnp.array([[0, 0, 12]]),
        )
        state = _arm_state(
            state_factory,
            machine_types=types,
            machine_direction=dirs,
            inv_items=inv_items,
            inv_counts=inv_counts,
        )
        result = run_arms(state)
        # IRON deposited into chest
        assert int(result.machine_inventory_counts[0, 0, 0]) == 8
        assert int(result.machine_inventory_items[0, 0, 0]) == int(ItemType.IRON)
        # COAL picked into arm buffer
        assert int(result.machine_inventory_counts[0, 1, 0]) == 12
        assert int(result.machine_inventory_items[0, 1, 0]) == int(ItemType.COAL)
        # MINER output now empty
        assert int(result.machine_inventory_counts[0, 2, 1]) == 0

    def test_noop_when_buffer_full_and_no_deposit_slot(self, state_factory) -> None:
        """Arm with buffer full but no compatible deposit slot does nothing."""
        # ARM faces right into empty space (no machine forward)
        shape = (1, 2)
        types = jnp.array([[MachineType.ARM, MachineType.NONE]])
        dirs = jnp.array([[int(Action.RIGHT), 0]])
        inv_items, inv_counts = _inv(
            shape,
            items_slot0=jnp.array([[int(ItemType.COAL), 0]]),
            counts_slot0=jnp.array([[5, 0]]),
        )
        state = _arm_state(
            state_factory,
            machine_types=types,
            machine_direction=dirs,
            inv_items=inv_items,
            inv_counts=inv_counts,
        )
        result = run_arms(state)
        assert int(result.machine_inventory_counts[0, 0, 0]) == 5

    def test_noop_when_buffer_empty_and_no_pick_source(self, state_factory) -> None:
        """Arm with empty buffer and empty backward neighbour does nothing."""
        shape = (1, 2)
        types = jnp.array([[MachineType.NONE, MachineType.ARM]])
        dirs = jnp.array([[0, int(Action.RIGHT)]])  # bwd = (0,0) NONE
        inv_items = jnp.zeros((*shape, MAX_MACHINE_INVENTORY_SLOTS), dtype=jnp.int32)
        inv_counts = jnp.zeros((*shape, MAX_MACHINE_INVENTORY_SLOTS), dtype=jnp.int16)
        state = _arm_state(
            state_factory,
            machine_types=types,
            machine_direction=dirs,
            inv_items=inv_items,
            inv_counts=inv_counts,
        )
        result = run_arms(state)
        assert int(result.machine_inventory_counts[0, 1, 0]) == 0

    def test_chest_to_belt_no_item_corruption(self, state_factory) -> None:
        """Arm depositing to belt then picking from chest must not corrupt belt items.

        Reproduces a bug where the pick phase's item-type clearing used a
        gather+where instead of a scatter, causing the belt's item type to
        be zeroed when the arm depleted the chest slot in the same tick.
        """
        # Layout: [CHEST, ARM, BELT]
        # ARM faces RIGHT: forward=BELT, backward=CHEST.
        # ARM buffer holds COAL (will deposit to belt), chest has more
        # COAL (will be picked after deposit clears the buffer).
        shape = (1, 3)
        types = jnp.array([[
            MachineType.CHEST,
            MachineType.ARM,
            MachineType.CONVEYOR_BELT,
        ]])
        dirs = jnp.array([[0, int(Action.RIGHT), int(Action.RIGHT)]])
        inv_items, inv_counts = _inv(
            shape,
            # Chest slot 0: 10 COAL.  ARM buffer: 5 COAL.  Belt: empty.
            items_slot0=jnp.array([
                [int(ItemType.COAL), int(ItemType.COAL), 0],
            ]),
            counts_slot0=jnp.array([[10, 5, 0]]),
        )
        state = _arm_state(
            state_factory,
            machine_types=types,
            machine_direction=dirs,
            inv_items=inv_items,
            inv_counts=inv_counts,
        )
        result = run_arms(state)

        # Deposit phase: ARM deposits 5 COAL to belt slot 0.
        # Pick phase: ARM (now empty) picks 10 COAL from chest slot 0.
        # Belt must retain COAL item type — not be zeroed.
        assert int(result.machine_inventory_items[0, 2, 0]) == int(
            ItemType.COAL
        ), "belt item type corrupted to EMPTY"
        assert int(result.machine_inventory_counts[0, 2, 0]) == 5

        # Chest should be empty (count decremented to 0).
        assert int(result.machine_inventory_counts[0, 0, 0]) == 0

        # ARM buffer should hold the picked COAL from chest.
        assert int(result.machine_inventory_items[0, 1, 0]) == int(
            ItemType.COAL
        )
        assert int(result.machine_inventory_counts[0, 1, 0]) == 10

    def test_does_not_pick_from_input_only_slot(self, state_factory) -> None:
        """Arm cannot pick from INPUT-role slots (miner fuel slot 0)."""
        # MINER slot 0 = INPUT (fuel); arm should not pick from it.
        # Arm at (0,1) facing RIGHT: bwd=(0,0) MINER.
        shape = (1, 2)
        types = jnp.array([[MachineType.MINER, MachineType.ARM]])
        dirs = jnp.array([[0, int(Action.RIGHT)]])
        inv_items, inv_counts = _inv(
            shape,
            # MINER fuel (slot 0, INPUT role) has COAL — should NOT be picked
            items_slot0=jnp.array([[int(ItemType.COAL), 0]]),
            counts_slot0=jnp.array([[10, 0]]),
        )
        state = _arm_state(
            state_factory,
            machine_types=types,
            machine_direction=dirs,
            inv_items=inv_items,
            inv_counts=inv_counts,
        )
        result = run_arms(state)
        # ARM buffer must remain empty (cannot pick from INPUT slot)
        assert int(result.machine_inventory_counts[0, 1, 0]) == 0
        # MINER fuel untouched
        assert int(result.machine_inventory_counts[0, 0, 0]) == 10
