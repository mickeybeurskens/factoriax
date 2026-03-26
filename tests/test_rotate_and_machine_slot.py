"""Tests for ROTATE, NEXT_MACHINE_SLOT, and PREV_MACHINE_SLOT actions.

Also covers the focused-slot behaviour of deposit and withdraw, where
``machine_selected_slot`` biases which machine slot is targeted.
"""

from __future__ import annotations

import jax.numpy as jnp

from factoriax import Action, BlockType, ItemType
from factoriax.constants import (
    MAX_MACHINE_INVENTORY_SLOTS,
    MachineType,
)
from factoriax.game_logic import (
    cycle_machine_slot,
    deposit_to_adjacent,
    rotate_adjacent,
    withdraw_from_adjacent,
)

_DIRT_3X3 = jnp.full((3, 3), BlockType.DIRT, dtype=jnp.int32)


def _machine_types(
    w: int, h: int, placements: dict[tuple[int, int], int]
) -> jnp.ndarray:
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
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build machine inventory arrays. Keys are (x, y, slot)."""
    item_arr = jnp.zeros((h, w, MAX_MACHINE_INVENTORY_SLOTS), dtype=jnp.int32)
    count_arr = jnp.zeros((h, w, MAX_MACHINE_INVENTORY_SLOTS), dtype=jnp.int16)
    for (x, y, slot), val in (items or {}).items():
        item_arr = item_arr.at[y, x, slot].set(val)
    for (x, y, slot), val in (counts or {}).items():
        count_arr = count_arr.at[y, x, slot].set(val)
    return item_arr, count_arr


# ---------------------------------------------------------------------------
# ROTATE tests
# ---------------------------------------------------------------------------


class TestRotateAdjacent:
    """Rotate the machine on the tile in front of the player."""

    def test_full_clockwise_cycle(self, state_factory) -> None:
        """Four rotations should cycle DOWN -> RIGHT -> UP -> LEFT -> DOWN."""
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Action.RIGHT,
            machine_types=_machine_types(3, 3, {(2, 1): MachineType.CHEST}),
            machine_direction=jnp.full((3, 3), Action.DOWN, dtype=jnp.int32),
        )

        expected = [Action.RIGHT, Action.UP, Action.LEFT, Action.DOWN]
        for exp in expected:
            state = rotate_adjacent(state, 0)
            assert int(state.machine_direction[1, 2]) == exp

    def test_noop_on_empty_tile(self, state_factory) -> None:
        """Rotation should be a no-op when no machine is in front."""
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Action.RIGHT,
        )
        original_dirs = state.machine_direction.copy()
        state = rotate_adjacent(state, 0)
        assert jnp.array_equal(state.machine_direction, original_dirs)

    def test_noop_out_of_bounds(self, state_factory) -> None:
        """Rotation targeting outside the map should be a no-op."""
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(2, 1),
            player_direction=Action.RIGHT,
        )
        original_dirs = state.machine_direction.copy()
        state = rotate_adjacent(state, 0)
        assert jnp.array_equal(state.machine_direction, original_dirs)


# ---------------------------------------------------------------------------
# CYCLE_MACHINE_SLOT tests
# ---------------------------------------------------------------------------


class TestCycleMachineSlot:
    """Advance or retreat the selected machine slot."""

    def test_advance_wraps_around(self, state_factory) -> None:
        """Advancing past the last slot wraps to slot 0."""
        # Chest has 8 slots. Set selected to 7, advance -> should wrap to 0.
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Action.RIGHT,
            machine_types=_machine_types(3, 3, {(2, 1): MachineType.CHEST}),
            machine_selected_slot=jnp.zeros((3, 3), dtype=jnp.int32).at[1, 2].set(7),
        )

        state = cycle_machine_slot(state, 0, 1)
        assert int(state.machine_selected_slot[1, 2]) == 0

    def test_retreat_wraps_around(self, state_factory) -> None:
        """Retreating past slot 0 wraps to the last slot."""
        # Chest has 8 slots. Start at 0, retreat -> should wrap to 7.
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Action.RIGHT,
            machine_types=_machine_types(3, 3, {(2, 1): MachineType.CHEST}),
        )

        state = cycle_machine_slot(state, 0, -1)
        assert int(state.machine_selected_slot[1, 2]) == 7

    def test_noop_no_machine(self, state_factory) -> None:
        """Cycling should be a no-op when no machine is in front."""
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Action.RIGHT,
        )
        original = state.machine_selected_slot.copy()
        state = cycle_machine_slot(state, 0, 1)
        assert jnp.array_equal(state.machine_selected_slot, original)


# ---------------------------------------------------------------------------
# Focused-slot deposit tests
# ---------------------------------------------------------------------------


class TestFocusedDeposit:
    """Deposit should prefer the machine_selected_slot when usable."""

    def test_deposit_into_focused_slot(self, state_factory) -> None:
        """When the focused slot can accept, deposit goes there."""
        inv_items = jnp.zeros((1, 10), dtype=jnp.int32).at[0, 0].set(ItemType.COAL)
        inv_counts = jnp.zeros((1, 10), dtype=jnp.int32).at[0, 0].set(5)
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Action.RIGHT,
            inventory_items=inv_items,
            inventory_counts=inv_counts,
            machine_types=_machine_types(3, 3, {(2, 1): MachineType.CHEST}),
            machine_selected_slot=jnp.zeros(
                (3, 3), dtype=jnp.int32
            ).at[1, 2].set(3),
        )

        state = deposit_to_adjacent(state, 0)

        # Should deposit into slot 3 (focused), not slot 0 (default).
        assert int(state.machine_inventory_items[1, 2, 3]) == ItemType.COAL
        assert int(state.machine_inventory_counts[1, 2, 3]) == 5
        assert int(state.machine_inventory_items[1, 2, 0]) == 0

    def test_deposit_fallback_when_focused_unusable(self, state_factory) -> None:
        """When the focused slot is full, deposit auto-routes elsewhere."""
        inv_items = jnp.zeros((1, 10), dtype=jnp.int32).at[0, 0].set(ItemType.IRON)
        inv_counts = jnp.zeros((1, 10), dtype=jnp.int32).at[0, 0].set(5)
        m_items, m_counts = _machine_inv(
            3, 3,
            items={(2, 1, 3): ItemType.COAL},
            counts={(2, 1, 3): 50},
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
            machine_selected_slot=jnp.zeros(
                (3, 3), dtype=jnp.int32
            ).at[1, 2].set(3),
        )

        state = deposit_to_adjacent(state, 0)

        # Focused slot 3 has coal, can't accept iron. Should go to slot 0.
        assert int(state.machine_inventory_items[1, 2, 0]) == ItemType.IRON
        assert int(state.machine_inventory_counts[1, 2, 0]) == 5


# ---------------------------------------------------------------------------
# Focused-slot withdraw tests
# ---------------------------------------------------------------------------


class TestFocusedWithdraw:
    """Withdraw should prefer the machine_selected_slot when it has items."""

    def test_withdraw_from_focused_slot(self, state_factory) -> None:
        """Focused slot with items should be withdrawn from first."""
        m_items, m_counts = _machine_inv(
            3, 3,
            items={(2, 1, 0): ItemType.COAL, (2, 1, 3): ItemType.IRON},
            counts={(2, 1, 0): 5, (2, 1, 3): 3},
        )
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Action.RIGHT,
            machine_types=_machine_types(3, 3, {(2, 1): MachineType.CHEST}),
            machine_inventory_items=m_items,
            machine_inventory_counts=m_counts,
            machine_selected_slot=jnp.zeros(
                (3, 3), dtype=jnp.int32
            ).at[1, 2].set(3),
        )

        state = withdraw_from_adjacent(state, 0)

        # Should take iron from slot 3 (focused), not coal from slot 0.
        assert int(state.inventory_items[0, 0]) == ItemType.IRON
        assert int(state.inventory_counts[0, 0]) == 3
        # Slot 3 should be cleared, slot 0 untouched.
        assert int(state.machine_inventory_counts[1, 2, 3]) == 0
        assert int(state.machine_inventory_counts[1, 2, 0]) == 5

    def test_withdraw_fallback_when_focused_empty(self, state_factory) -> None:
        """When the focused slot is empty, fall back to priority order."""
        m_items, m_counts = _machine_inv(
            3, 3,
            items={(2, 1, 0): ItemType.COAL},
            counts={(2, 1, 0): 5},
        )
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Action.RIGHT,
            machine_types=_machine_types(3, 3, {(2, 1): MachineType.CHEST}),
            machine_inventory_items=m_items,
            machine_inventory_counts=m_counts,
            machine_selected_slot=jnp.zeros(
                (3, 3), dtype=jnp.int32
            ).at[1, 2].set(5),
        )

        state = withdraw_from_adjacent(state, 0)

        # Focused slot 5 is empty, should fall back to slot 0.
        assert int(state.inventory_items[0, 0]) == ItemType.COAL
        assert int(state.inventory_counts[0, 0]) == 5
