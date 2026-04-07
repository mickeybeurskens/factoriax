"""Tests for conveyor belt and pick-and-place arm machine logic.

Uses the pouch inventory model where machine_inventory has shape
(H, W, NUM_ITEM_TYPES) and items are indexed by ItemType.
"""

from __future__ import annotations

import jax.numpy as jnp

from factoriax.constants import (
    DEFAULT_MACHINE_MAX_HEALTH,
    MACHINE_INVENTORY_COUNT_DTYPE,
    MAX_MACHINE_STACK_SIZE,
    NUM_ITEM_TYPES,
    Direction,
    ItemType,
    MachineType,
)
from factoriax.machines import run_arms, run_conveyor_belts

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_machine_inv(
    shape: tuple[int, int],
    entries: dict[tuple[int, int, int], int] | None = None,
) -> jnp.ndarray:
    """Build a machine inventory pouch with specific item counts.

    Args:
        shape: Grid (H, W).
        entries: Mapping of (y, x, item_type) -> count.

    Returns:
        Machine inventory array of shape (H, W, NUM_ITEM_TYPES).
    """
    inv = jnp.zeros(
        (*shape, NUM_ITEM_TYPES),
        dtype=MACHINE_INVENTORY_COUNT_DTYPE,
    )
    for (y, x, item_type), count in (entries or {}).items():
        inv = inv.at[y, x, item_type].set(count)
    return inv


def _belt_state(
    state_factory,
    *,
    machine_types: jnp.ndarray,
    machine_direction: jnp.ndarray,
    machine_inventory: jnp.ndarray | None = None,
):
    """Build a state with the given belt grid and optional inventory."""
    shape = machine_types.shape
    if machine_inventory is None:
        machine_inventory = jnp.zeros(
            (*shape, NUM_ITEM_TYPES),
            dtype=MACHINE_INVENTORY_COUNT_DTYPE,
        )
    health = jnp.where(
        machine_types != int(MachineType.NONE),
        DEFAULT_MACHINE_MAX_HEALTH,
        0,
    ).astype(jnp.int32)
    return state_factory(
        world_map=jnp.zeros(shape, dtype=jnp.int32),
        machine_types=machine_types,
        machine_direction=machine_direction,
        machine_inventory=machine_inventory,
        machine_health=health,
    )


def _arm_state(
    state_factory,
    *,
    machine_types: jnp.ndarray,
    machine_direction: jnp.ndarray,
    machine_inventory: jnp.ndarray,
):
    """Build a state with the given machine grid and full inventory."""
    health = jnp.where(
        machine_types != int(MachineType.NONE),
        DEFAULT_MACHINE_MAX_HEALTH,
        0,
    ).astype(jnp.int32)
    return state_factory(
        world_map=jnp.zeros(machine_types.shape, dtype=jnp.int32),
        machine_types=machine_types,
        machine_direction=machine_direction,
        machine_inventory=machine_inventory,
        machine_health=health,
    )


# ---------------------------------------------------------------------------
# Conveyor belt tests
# ---------------------------------------------------------------------------


class TestConveyorBelt:
    """Conveyor belt item transport tests."""

    def test_noop_when_no_items(self, state_factory) -> None:
        """Belt with empty inventory does nothing."""
        types = jnp.array([[MachineType.CONVEYOR_BELT, MachineType.CONVEYOR_BELT]])
        dirs = jnp.array([[Direction.RIGHT, Direction.RIGHT]])
        state = _belt_state(state_factory, machine_types=types, machine_direction=dirs)
        result = run_conveyor_belts(state)
        assert int(result.machine_inventory[0, 0, ItemType.COAL]) == 0
        assert int(result.machine_inventory[0, 1, ItemType.COAL]) == 0

    def test_pushes_item_right(self, state_factory) -> None:
        """Belt facing right moves items from (0,0) to (0,1)."""
        types = jnp.array([[MachineType.CONVEYOR_BELT, MachineType.CONVEYOR_BELT]])
        dirs = jnp.array([[Direction.RIGHT, Direction.RIGHT]])
        inv = _make_machine_inv((1, 2), {(0, 0, ItemType.COAL): 10})
        state = _belt_state(
            state_factory,
            machine_types=types,
            machine_direction=dirs,
            machine_inventory=inv,
        )
        result = run_conveyor_belts(state)
        assert int(result.machine_inventory[0, 0, ItemType.COAL]) == 0
        assert int(result.machine_inventory[0, 1, ItemType.COAL]) == 10

    def test_pushes_item_down(self, state_factory) -> None:
        """Belt facing down moves items from row 0 to row 1."""
        types = jnp.array(
            [[MachineType.CONVEYOR_BELT], [MachineType.CONVEYOR_BELT]],
        )
        dirs = jnp.array([[Direction.DOWN], [Direction.DOWN]])
        inv = _make_machine_inv((2, 1), {(0, 0, ItemType.IRON): 5})
        state = _belt_state(
            state_factory,
            machine_types=types,
            machine_direction=dirs,
            machine_inventory=inv,
        )
        result = run_conveyor_belts(state)
        assert int(result.machine_inventory[0, 0, ItemType.IRON]) == 0
        assert int(result.machine_inventory[1, 0, ItemType.IRON]) == 5

    def test_does_not_push_to_non_belt(self, state_factory) -> None:
        """Belt adjacent to a non-belt machine does not transfer items."""
        types = jnp.array(
            [[MachineType.CONVEYOR_BELT, MachineType.CHEST]],
        )
        dirs = jnp.array([[Direction.RIGHT, Direction.RIGHT]])
        inv = _make_machine_inv((1, 2), {(0, 0, ItemType.COAL): 8})
        state = _belt_state(
            state_factory,
            machine_types=types,
            machine_direction=dirs,
            machine_inventory=inv,
        )
        result = run_conveyor_belts(state)
        assert int(result.machine_inventory[0, 0, ItemType.COAL]) == 8
        assert int(result.machine_inventory[0, 1, ItemType.COAL]) == 0

    def test_does_not_push_to_blocked_target(self, state_factory) -> None:
        """Belt does not push when target holds a different item at max stack."""
        types = jnp.array([[MachineType.CONVEYOR_BELT, MachineType.CONVEYOR_BELT]])
        dirs = jnp.array([[Direction.RIGHT, Direction.RIGHT]])
        inv = _make_machine_inv(
            (1, 2),
            {
                (0, 0, ItemType.COAL): 5,
                (0, 1, ItemType.IRON): MAX_MACHINE_STACK_SIZE,
            },
        )
        state = _belt_state(
            state_factory,
            machine_types=types,
            machine_direction=dirs,
            machine_inventory=inv,
        )
        result = run_conveyor_belts(state)
        assert int(result.machine_inventory[0, 0, ItemType.COAL]) == 5

    def test_merges_same_item_into_target(self, state_factory) -> None:
        """Items of the same type merge into the target's existing stack."""
        types = jnp.array([[MachineType.CONVEYOR_BELT, MachineType.CONVEYOR_BELT]])
        dirs = jnp.array([[Direction.RIGHT, Direction.RIGHT]])
        inv = _make_machine_inv(
            (1, 2),
            {(0, 0, ItemType.COAL): 3, (0, 1, ItemType.COAL): 7},
        )
        state = _belt_state(
            state_factory,
            machine_types=types,
            machine_direction=dirs,
            machine_inventory=inv,
        )
        result = run_conveyor_belts(state)
        assert int(result.machine_inventory[0, 1, ItemType.COAL]) == 10
        assert int(result.machine_inventory[0, 0, ItemType.COAL]) == 0

    def test_noop_with_zero_direction(self, state_factory) -> None:
        """Belt with NOOP direction (0) does not push to self."""
        types = jnp.array([[MachineType.CONVEYOR_BELT]])
        dirs = jnp.array([[0]])  # NOOP
        inv = _make_machine_inv((1, 1), {(0, 0, ItemType.COAL): 5})
        state = _belt_state(
            state_factory,
            machine_types=types,
            machine_direction=dirs,
            machine_inventory=inv,
        )
        result = run_conveyor_belts(state)
        assert int(result.machine_inventory[0, 0, ItemType.COAL]) == 5


# ---------------------------------------------------------------------------
# Pick-and-place arm tests
# ---------------------------------------------------------------------------


class TestArm:
    """Pick-and-place arm tests."""

    def test_picks_from_miner_output(self, state_factory) -> None:
        """Arm picks item from miner's inventory (e.g. mined COAL)."""
        # Layout: [MINER, ARM] -- arm faces RIGHT: bwd=(0,0) MINER.
        shape = (1, 2)
        types = jnp.array([[MachineType.MINER, MachineType.ARM]])
        dirs = jnp.array([[0, int(Direction.RIGHT)]])
        inv = _make_machine_inv(shape, {(0, 0, ItemType.COAL): 15})
        state = _arm_state(
            state_factory,
            machine_types=types,
            machine_direction=dirs,
            machine_inventory=inv,
        )
        result = run_arms(state)
        # ARM buffer should now hold COAL.
        assert int(result.machine_inventory[0, 1, ItemType.COAL]) == 15
        # Miner should be empty.
        assert int(result.machine_inventory[0, 0, ItemType.COAL]) == 0

    def test_deposits_to_chest(self, state_factory) -> None:
        """Arm with full buffer deposits items into adjacent chest."""
        # Layout: [CHEST, ARM] -- arm at (0,1) facing LEFT: forward=(0,0).
        shape = (1, 2)
        types = jnp.array([[MachineType.CHEST, MachineType.ARM]])
        dirs = jnp.array([[0, int(Direction.LEFT)]])
        inv = _make_machine_inv(shape, {(0, 1, ItemType.IRON): 20})
        state = _arm_state(
            state_factory,
            machine_types=types,
            machine_direction=dirs,
            machine_inventory=inv,
        )
        result = run_arms(state)
        # ARM buffer cleared.
        assert int(result.machine_inventory[0, 1, ItemType.IRON]) == 0
        # Chest received items.
        assert int(result.machine_inventory[0, 0, ItemType.IRON]) == 20

    def test_deposit_then_pick_same_tick(self, state_factory) -> None:
        """Arm deposits first, then picks in same tick if buffer is now empty."""
        # Layout: [CHEST_with_IRON, ARM, MINER_with_COAL]
        # Arm at (0,1) facing LEFT: fwd=(0,0) CHEST, bwd=(0,2) MINER.
        shape = (1, 3)
        types = jnp.array(
            [[MachineType.CHEST, MachineType.ARM, MachineType.MINER]],
        )
        dirs = jnp.array([[0, int(Direction.LEFT), 0]])
        inv = _make_machine_inv(
            shape,
            {
                (0, 1, ItemType.IRON): 8,  # ARM buffer
                (0, 2, ItemType.COAL): 12,  # MINER output
            },
        )
        state = _arm_state(
            state_factory,
            machine_types=types,
            machine_direction=dirs,
            machine_inventory=inv,
        )
        result = run_arms(state)
        # IRON deposited into chest.
        assert int(result.machine_inventory[0, 0, ItemType.IRON]) == 8
        # COAL picked into arm buffer.
        assert int(result.machine_inventory[0, 1, ItemType.COAL]) == 12
        # MINER output now empty.
        assert int(result.machine_inventory[0, 2, ItemType.COAL]) == 0

    def test_noop_when_buffer_full_and_no_deposit_slot(
        self,
        state_factory,
    ) -> None:
        """Arm with buffer full but no compatible deposit target does nothing."""
        shape = (1, 2)
        types = jnp.array([[MachineType.ARM, MachineType.NONE]])
        dirs = jnp.array([[int(Direction.RIGHT), 0]])
        inv = _make_machine_inv(shape, {(0, 0, ItemType.COAL): 5})
        state = _arm_state(
            state_factory,
            machine_types=types,
            machine_direction=dirs,
            machine_inventory=inv,
        )
        result = run_arms(state)
        assert int(result.machine_inventory[0, 0, ItemType.COAL]) == 5

    def test_noop_when_buffer_empty_and_no_pick_source(
        self,
        state_factory,
    ) -> None:
        """Arm with empty buffer and empty backward neighbour does nothing."""
        shape = (1, 2)
        types = jnp.array([[MachineType.NONE, MachineType.ARM]])
        dirs = jnp.array([[0, int(Direction.RIGHT)]])
        inv = _make_machine_inv(shape)
        state = _arm_state(
            state_factory,
            machine_types=types,
            machine_direction=dirs,
            machine_inventory=inv,
        )
        result = run_arms(state)
        assert int(result.machine_inventory[0, 1, ItemType.COAL]) == 0

    def test_chest_to_belt_no_item_corruption(self, state_factory) -> None:
        """Arm depositing to belt then picking from chest must not corrupt items.

        Reproduces a bug where the pick phase's item-type clearing used a
        gather+where instead of a scatter, causing the belt's item type to
        be zeroed when the arm depleted the chest slot in the same tick.
        """
        # Layout: [CHEST, ARM, BELT]
        # ARM faces RIGHT: forward=BELT, backward=CHEST.
        shape = (1, 3)
        types = jnp.array(
            [
                [
                    MachineType.CHEST,
                    MachineType.ARM,
                    MachineType.CONVEYOR_BELT,
                ]
            ]
        )
        dirs = jnp.array([[0, int(Direction.RIGHT), int(Direction.RIGHT)]])
        inv = _make_machine_inv(
            shape,
            {
                (0, 0, ItemType.COAL): 10,  # Chest
                (0, 1, ItemType.COAL): 5,  # ARM buffer
            },
        )
        state = _arm_state(
            state_factory,
            machine_types=types,
            machine_direction=dirs,
            machine_inventory=inv,
        )
        result = run_arms(state)

        # Deposit phase: ARM deposits 5 COAL to belt.
        # Pick phase: ARM (now empty) picks 10 COAL from chest.
        # Belt must retain COAL -- not be zeroed.
        assert int(result.machine_inventory[0, 2, ItemType.COAL]) == 5, (
            "belt item corrupted to zero"
        )

        # Chest should be empty.
        assert int(result.machine_inventory[0, 0, ItemType.COAL]) == 0

        # ARM buffer should hold the picked COAL from chest.
        assert int(result.machine_inventory[0, 1, ItemType.COAL]) == 10
