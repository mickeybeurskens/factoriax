"""Tests for underground belt placement and item flow."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from factoriax.constants import (
    MACHINE_INVENTORY_COUNT_DTYPE,
    NUM_ITEM_TYPES,
    Action,
    Direction,
    ItemType,
    MachineType,
)
from factoriax.machines import run_conveyor_belts
from factoriax.placement import place_machine


def _belt_map(h: int, w: int) -> jnp.ndarray:
    """Create a dirt-only map."""
    return jnp.zeros((h, w), dtype=jnp.int32)


class TestUndergroundPlacement:
    """Placement determines entry vs exit automatically."""

    def test_first_placement_is_entry(self, state_factory) -> None:
        """First underground belt placed becomes an entry."""
        state = state_factory(
            world_map=_belt_map(5, 10),
            player_position=(2, 2),
            player_direction=Direction.RIGHT,
            player_inventory=jnp.zeros(
                (1, NUM_ITEM_TYPES),
                dtype=jnp.int32,
            )
            .at[0, ItemType.UNDERGROUND_BELT]
            .set(5),
        )
        state = place_machine(state, 0, ItemType.UNDERGROUND_BELT)
        assert int(state.machine_types[2, 3]) == MachineType.UNDERGROUND_ENTRY

    def test_second_placement_is_exit(self, state_factory) -> None:
        """Second placement within range and same direction is an exit."""
        mt = jnp.zeros((5, 10), dtype=jnp.int32)
        mt = mt.at[2, 3].set(MachineType.UNDERGROUND_ENTRY)
        md = jnp.zeros((5, 10), dtype=jnp.int32)
        md = md.at[2, 3].set(Direction.RIGHT)
        state = state_factory(
            world_map=_belt_map(5, 10),
            player_position=(5, 2),
            player_direction=Direction.RIGHT,
            player_inventory=jnp.zeros(
                (1, NUM_ITEM_TYPES),
                dtype=jnp.int32,
            )
            .at[0, ItemType.UNDERGROUND_BELT]
            .set(5),
            machine_types=mt,
            machine_direction=md,
        )
        state = place_machine(state, 0, ItemType.UNDERGROUND_BELT)
        assert int(state.machine_types[2, 6]) == MachineType.UNDERGROUND_EXIT

    def test_placement_beyond_range_is_entry(self, state_factory) -> None:
        """Placement beyond MAX_UNDERGROUND_RANGE creates a new entry."""
        mt = jnp.zeros((5, 15), dtype=jnp.int32)
        mt = mt.at[2, 1].set(MachineType.UNDERGROUND_ENTRY)
        md = jnp.zeros((5, 15), dtype=jnp.int32)
        md = md.at[2, 1].set(Direction.RIGHT)
        state = state_factory(
            world_map=_belt_map(5, 15),
            player_position=(7, 2),
            player_direction=Direction.RIGHT,
            player_inventory=jnp.zeros(
                (1, NUM_ITEM_TYPES),
                dtype=jnp.int32,
            )
            .at[0, ItemType.UNDERGROUND_BELT]
            .set(5),
            machine_types=mt,
            machine_direction=md,
        )
        # Place at (8, 2) which is 7 tiles away from entry at (1, 2).
        state = place_machine(state, 0, ItemType.UNDERGROUND_BELT)
        assert int(state.machine_types[2, 8]) == MachineType.UNDERGROUND_ENTRY


class TestUndergroundFlow:
    """Items flow from entry to paired exit via run_conveyor_belts."""

    def test_items_flow_entry_to_exit(self, state_factory) -> None:
        """Items on an entry move to the paired exit."""
        mt = jnp.zeros((3, 8), dtype=jnp.int32)
        mt = mt.at[1, 2].set(MachineType.UNDERGROUND_ENTRY)
        mt = mt.at[1, 5].set(MachineType.UNDERGROUND_EXIT)
        md = jnp.zeros((3, 8), dtype=jnp.int32)
        md = md.at[1, 2].set(Direction.RIGHT)
        md = md.at[1, 5].set(Direction.RIGHT)

        mi = jnp.zeros((3, 8, NUM_ITEM_TYPES), dtype=MACHINE_INVENTORY_COUNT_DTYPE)
        mi = mi.at[1, 2, ItemType.IRON].set(10)

        state = state_factory(
            world_map=_belt_map(3, 8),
            machine_types=mt,
            machine_direction=md,
            machine_inventory=mi,
        )
        state = run_conveyor_belts(state)

        # Entry should be empty, exit should have the items.
        assert int(state.machine_inventory[1, 2, ItemType.IRON]) == 0
        assert int(state.machine_inventory[1, 5, ItemType.IRON]) == 10

    def test_unpaired_entry_holds_items(self, state_factory) -> None:
        """Entry with no exit keeps its items."""
        mt = jnp.zeros((3, 8), dtype=jnp.int32)
        mt = mt.at[1, 2].set(MachineType.UNDERGROUND_ENTRY)
        md = jnp.zeros((3, 8), dtype=jnp.int32)
        md = md.at[1, 2].set(Direction.RIGHT)

        mi = jnp.zeros((3, 8, NUM_ITEM_TYPES), dtype=MACHINE_INVENTORY_COUNT_DTYPE)
        mi = mi.at[1, 2, ItemType.IRON].set(10)

        state = state_factory(
            world_map=_belt_map(3, 8),
            machine_types=mt,
            machine_direction=md,
            machine_inventory=mi,
        )
        state = run_conveyor_belts(state)

        assert int(state.machine_inventory[1, 2, ItemType.IRON]) == 10

    def test_exit_pushes_to_regular_belt(self, state_factory) -> None:
        """Exit pushes items to an adjacent regular belt."""
        mt = jnp.zeros((3, 8), dtype=jnp.int32)
        mt = mt.at[1, 3].set(MachineType.UNDERGROUND_EXIT)
        mt = mt.at[1, 4].set(MachineType.CONVEYOR_BELT)
        md = jnp.zeros((3, 8), dtype=jnp.int32)
        md = md.at[1, 3].set(Direction.RIGHT)
        md = md.at[1, 4].set(Direction.RIGHT)

        mi = jnp.zeros((3, 8, NUM_ITEM_TYPES), dtype=MACHINE_INVENTORY_COUNT_DTYPE)
        mi = mi.at[1, 3, ItemType.COAL].set(5)

        state = state_factory(
            world_map=_belt_map(3, 8),
            machine_types=mt,
            machine_direction=md,
            machine_inventory=mi,
        )
        state = run_conveyor_belts(state)

        assert int(state.machine_inventory[1, 3, ItemType.COAL]) == 0
        assert int(state.machine_inventory[1, 4, ItemType.COAL]) == 5

    def test_regular_belt_pushes_to_entry(self, state_factory) -> None:
        """Regular belt pushes items into an adjacent entry."""
        mt = jnp.zeros((3, 8), dtype=jnp.int32)
        mt = mt.at[1, 2].set(MachineType.CONVEYOR_BELT)
        mt = mt.at[1, 3].set(MachineType.UNDERGROUND_ENTRY)
        md = jnp.zeros((3, 8), dtype=jnp.int32)
        md = md.at[1, 2].set(Direction.RIGHT)
        md = md.at[1, 3].set(Direction.RIGHT)

        mi = jnp.zeros((3, 8, NUM_ITEM_TYPES), dtype=MACHINE_INVENTORY_COUNT_DTYPE)
        mi = mi.at[1, 2, ItemType.IRON].set(3)

        state = state_factory(
            world_map=_belt_map(3, 8),
            machine_types=mt,
            machine_direction=md,
            machine_inventory=mi,
        )
        state = run_conveyor_belts(state)

        assert int(state.machine_inventory[1, 2, ItemType.IRON]) == 0
        assert int(state.machine_inventory[1, 3, ItemType.IRON]) == 3

    def test_full_chain_belt_entry_exit_belt(self, state_factory) -> None:
        """Items flow through a complete belt->entry->exit->belt chain."""
        mt = jnp.zeros((3, 10), dtype=jnp.int32)
        mt = mt.at[1, 1].set(MachineType.CONVEYOR_BELT)  # source
        mt = mt.at[1, 2].set(MachineType.UNDERGROUND_ENTRY)
        mt = mt.at[1, 5].set(MachineType.UNDERGROUND_EXIT)
        mt = mt.at[1, 6].set(MachineType.CONVEYOR_BELT)  # dest
        md = jnp.zeros((3, 10), dtype=jnp.int32)
        for c in [1, 2, 5, 6]:
            md = md.at[1, c].set(Direction.RIGHT)

        mi = jnp.zeros(
            (3, 10, NUM_ITEM_TYPES),
            dtype=MACHINE_INVENTORY_COUNT_DTYPE,
        )
        mi = mi.at[1, 1, ItemType.COPPER].set(8)

        state = state_factory(
            world_map=_belt_map(3, 10),
            machine_types=mt,
            machine_direction=md,
            machine_inventory=mi,
        )

        # Tick 1: belt -> entry.
        state = run_conveyor_belts(state)
        # Tick 2: entry -> exit.
        state = run_conveyor_belts(state)
        # Tick 3: exit -> belt.
        state = run_conveyor_belts(state)

        assert int(state.machine_inventory[1, 6, ItemType.COPPER]) == 8

    def test_exit_does_not_accept_from_surface_belt(
        self,
        state_factory,
    ) -> None:
        """A regular belt cannot push directly into an exit."""
        mt = jnp.zeros((3, 8), dtype=jnp.int32)
        mt = mt.at[1, 3].set(MachineType.CONVEYOR_BELT)
        mt = mt.at[1, 4].set(MachineType.UNDERGROUND_EXIT)
        md = jnp.zeros((3, 8), dtype=jnp.int32)
        md = md.at[1, 3].set(Direction.RIGHT)
        md = md.at[1, 4].set(Direction.RIGHT)

        mi = jnp.zeros((3, 8, NUM_ITEM_TYPES), dtype=MACHINE_INVENTORY_COUNT_DTYPE)
        mi = mi.at[1, 3, ItemType.IRON].set(5)

        state = state_factory(
            world_map=_belt_map(3, 8),
            machine_types=mt,
            machine_direction=md,
            machine_inventory=mi,
        )
        state = run_conveyor_belts(state)

        # Belt should keep its items (exit rejects surface push).
        assert int(state.machine_inventory[1, 3, ItemType.IRON]) == 5
        assert int(state.machine_inventory[1, 4, ItemType.IRON]) == 0
