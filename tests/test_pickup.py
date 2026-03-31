"""Tests for the machine pickup system."""

import jax.numpy as jnp

from factoriax import BlockType, Direction, ItemType
from factoriax.constants import (
    MAX_MACHINE_INVENTORY_SLOTS,
    MAX_STACK_SIZE,
    NUM_INVENTORY_SLOTS,
    MachineType,
)
from factoriax.placement import can_fit_in_inventory, pickup_machine


class TestCanFitInInventory:
    """Tests for the inventory space simulation."""

    def test_empty_inventory_fits_single_item(self) -> None:
        """Should fit a single item into an empty inventory."""
        items = jnp.zeros(NUM_INVENTORY_SLOTS, dtype=jnp.int32)
        counts = jnp.zeros(NUM_INVENTORY_SLOTS, dtype=jnp.int32)
        to_add = jnp.array([ItemType.IRON], dtype=jnp.int32)
        to_add_counts = jnp.array([1], dtype=jnp.int32)
        assert can_fit_in_inventory(items, counts, to_add, to_add_counts)

    def test_full_inventory_rejects(self) -> None:
        """Should reject when all slots are full with different items."""
        items = jnp.array([ItemType.IRON] * NUM_INVENTORY_SLOTS, dtype=jnp.int32)
        counts = jnp.full(NUM_INVENTORY_SLOTS, MAX_STACK_SIZE, dtype=jnp.int32)
        to_add = jnp.array([ItemType.COPPER], dtype=jnp.int32)
        to_add_counts = jnp.array([1], dtype=jnp.int32)
        assert not can_fit_in_inventory(items, counts, to_add, to_add_counts)

    def test_stacking_fits(self) -> None:
        """Should fit when existing stack has room."""
        items = jnp.zeros(NUM_INVENTORY_SLOTS, dtype=jnp.int32)
        items = items.at[0].set(ItemType.IRON)
        counts = jnp.zeros(NUM_INVENTORY_SLOTS, dtype=jnp.int32)
        counts = counts.at[0].set(MAX_STACK_SIZE - 1)
        to_add = jnp.array([ItemType.IRON], dtype=jnp.int32)
        to_add_counts = jnp.array([1], dtype=jnp.int32)
        assert can_fit_in_inventory(items, counts, to_add, to_add_counts)

    def test_empty_items_ignored(self) -> None:
        """Should ignore EMPTY item entries in the items_to_add array."""
        items = jnp.zeros(NUM_INVENTORY_SLOTS, dtype=jnp.int32)
        counts = jnp.zeros(NUM_INVENTORY_SLOTS, dtype=jnp.int32)
        to_add = jnp.array([ItemType.EMPTY, ItemType.IRON], dtype=jnp.int32)
        to_add_counts = jnp.array([5, 1], dtype=jnp.int32)
        assert can_fit_in_inventory(items, counts, to_add, to_add_counts)


class TestPickupMachine:
    """Tests for the pickup_machine function."""

    def test_basic_pickup(self, state_factory) -> None:
        """Should remove machine from tile and return item to inventory."""
        state = state_factory(
            world_map=jnp.array(
                [[BlockType.DIRT, BlockType.DIRT]],
                dtype=jnp.int32,
            ),
            player_position=(0, 0),
            player_direction=Direction.RIGHT,
            machine_types=jnp.array(
                [[MachineType.NONE, MachineType.CHEST]], dtype=jnp.int32
            ),
        )
        new_state = pickup_machine(state, 0)

        assert int(new_state.machine_types[0, 1]) == MachineType.NONE
        assert int(new_state.inventory_items[0, 0]) == ItemType.CHEST
        assert int(new_state.inventory_counts[0, 0]) == 1

    def test_pickup_with_machine_inventory(self, state_factory) -> None:
        """Should transfer all machine inventory contents to player."""
        shape = (1, 2, MAX_MACHINE_INVENTORY_SLOTS)
        machine_inv_items = jnp.zeros(shape, dtype=jnp.int32)
        machine_inv_items = machine_inv_items.at[0, 1, 0].set(ItemType.IRON)
        machine_inv_items = machine_inv_items.at[0, 1, 1].set(ItemType.COPPER)
        machine_inv_counts = jnp.zeros(
            (1, 2, MAX_MACHINE_INVENTORY_SLOTS), dtype=jnp.int16
        )
        machine_inv_counts = machine_inv_counts.at[0, 1, 0].set(10)
        machine_inv_counts = machine_inv_counts.at[0, 1, 1].set(5)

        state = state_factory(
            world_map=jnp.array(
                [[BlockType.DIRT, BlockType.DIRT]],
                dtype=jnp.int32,
            ),
            player_position=(0, 0),
            player_direction=Direction.RIGHT,
            machine_types=jnp.array(
                [[MachineType.NONE, MachineType.CHEST]], dtype=jnp.int32
            ),
            machine_inventory_items=machine_inv_items,
            machine_inventory_counts=machine_inv_counts,
        )
        new_state = pickup_machine(state, 0)

        assert int(new_state.machine_types[0, 1]) == MachineType.NONE

        inv_items = new_state.inventory_items[0]
        inv_counts = new_state.inventory_counts[0]
        iron_mask = inv_items == ItemType.IRON
        copper_mask = inv_items == ItemType.COPPER
        chest_mask = inv_items == ItemType.CHEST

        assert int(jnp.sum(jnp.where(iron_mask, inv_counts, 0))) == 10
        assert int(jnp.sum(jnp.where(copper_mask, inv_counts, 0))) == 5
        assert int(jnp.sum(jnp.where(chest_mask, inv_counts, 0))) == 1

    def test_pickup_facing_empty_tile(self, state_factory) -> None:
        """Should be a no-op when facing a tile with no machine."""
        state = state_factory(
            world_map=jnp.array(
                [[BlockType.DIRT, BlockType.DIRT]],
                dtype=jnp.int32,
            ),
            player_position=(0, 0),
            player_direction=Direction.RIGHT,
        )
        new_state = pickup_machine(state, 0)

        assert int(new_state.machine_types[0, 1]) == MachineType.NONE
        assert int(new_state.inventory_items[0, 0]) == ItemType.EMPTY

    def test_pickup_out_of_bounds(self, state_factory) -> None:
        """Should be a no-op when facing out of map bounds."""
        state = state_factory(
            world_map=jnp.array(
                [[BlockType.DIRT]],
                dtype=jnp.int32,
            ),
            player_position=(0, 0),
            player_direction=Direction.RIGHT,
        )
        new_state = pickup_machine(state, 0)

        assert int(new_state.inventory_items[0, 0]) == ItemType.EMPTY

    def test_inventory_full_blocks_pickup(self, state_factory) -> None:
        """Should not pick up when inventory is completely full."""
        inv_items = jnp.full((1, NUM_INVENTORY_SLOTS), ItemType.COAL, dtype=jnp.int32)
        inv_counts = jnp.full((1, NUM_INVENTORY_SLOTS), MAX_STACK_SIZE, dtype=jnp.int32)

        state = state_factory(
            world_map=jnp.array(
                [[BlockType.DIRT, BlockType.DIRT]],
                dtype=jnp.int32,
            ),
            player_position=(0, 0),
            player_direction=Direction.RIGHT,
            inventory_items=inv_items,
            inventory_counts=inv_counts,
            machine_types=jnp.array(
                [[MachineType.NONE, MachineType.CHEST]], dtype=jnp.int32
            ),
        )
        new_state = pickup_machine(state, 0)

        assert int(new_state.machine_types[0, 1]) == MachineType.CHEST
        assert jnp.array_equal(new_state.inventory_items, inv_items)
        assert jnp.array_equal(new_state.inventory_counts, inv_counts)

    def test_inventory_partially_full_blocks_pickup(self, state_factory) -> None:
        """Should block pickup if space exists for machine but not contents."""
        inv_items = jnp.full((1, NUM_INVENTORY_SLOTS), ItemType.COAL, dtype=jnp.int32)
        inv_counts = jnp.full((1, NUM_INVENTORY_SLOTS), MAX_STACK_SIZE, dtype=jnp.int32)
        inv_items = inv_items.at[0, 9].set(ItemType.EMPTY)
        inv_counts = inv_counts.at[0, 9].set(0)

        machine_inv_items = jnp.zeros(
            (1, 2, MAX_MACHINE_INVENTORY_SLOTS), dtype=jnp.int32
        )
        machine_inv_items = machine_inv_items.at[0, 1, 0].set(ItemType.IRON)
        machine_inv_items = machine_inv_items.at[0, 1, 1].set(ItemType.COPPER)
        machine_inv_counts = jnp.zeros(
            (1, 2, MAX_MACHINE_INVENTORY_SLOTS), dtype=jnp.int16
        )
        machine_inv_counts = machine_inv_counts.at[0, 1, 0].set(10)
        machine_inv_counts = machine_inv_counts.at[0, 1, 1].set(5)

        state = state_factory(
            world_map=jnp.array(
                [[BlockType.DIRT, BlockType.DIRT]],
                dtype=jnp.int32,
            ),
            player_position=(0, 0),
            player_direction=Direction.RIGHT,
            inventory_items=inv_items,
            inventory_counts=inv_counts,
            machine_types=jnp.array(
                [[MachineType.NONE, MachineType.CHEST]], dtype=jnp.int32
            ),
            machine_inventory_items=machine_inv_items,
            machine_inventory_counts=machine_inv_counts,
        )
        new_state = pickup_machine(state, 0)

        assert int(new_state.machine_types[0, 1]) == MachineType.CHEST
        assert jnp.array_equal(new_state.inventory_items, inv_items)

    def test_pickup_clears_machine_state(self, state_factory) -> None:
        """Should clear all machine state fields on the tile after pickup."""
        state = state_factory(
            world_map=jnp.array(
                [[BlockType.DIRT, BlockType.DIRT]],
                dtype=jnp.int32,
            ),
            player_position=(0, 0),
            player_direction=Direction.RIGHT,
            machine_types=jnp.array(
                [[MachineType.NONE, MachineType.MINER]], dtype=jnp.int32
            ),
            machine_power=jnp.array([[0, 50]], dtype=jnp.int32),
            machine_direction=jnp.array([[0, Direction.DOWN]], dtype=jnp.int32),
        )
        new_state = pickup_machine(state, 0)

        assert int(new_state.machine_types[0, 1]) == MachineType.NONE
        assert int(new_state.machine_power[0, 1]) == 0
        assert int(new_state.machine_direction[0, 1]) == 0
        assert int(new_state.machine_selected_recipe[0, 1]) == 0
        assert int(new_state.machine_selected_slot[0, 1]) == 0
        assert jnp.all(new_state.machine_inventory_items[0, 1] == 0)
        assert jnp.all(new_state.machine_inventory_counts[0, 1] == 0)
