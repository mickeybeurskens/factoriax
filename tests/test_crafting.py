"""Tests for the crafting system."""

import jax.numpy as jnp
import pytest

from factoriax import BlockType, ItemType
from factoriax.constants import NUM_INVENTORY_SLOTS
from factoriax.crafting import (
    add_item_to_inventory,
    can_afford_recipe,
    consume_recipe_materials,
    count_item_in_inventory,
    cycle_slot,
    start_crafting,
    update_crafting,
)



class TestInventoryHelpers:
    """Tests for inventory helper functions."""

    def test_count_item_empty_inventory(self, state_factory) -> None:
        """Empty inventory should have zero count for all items."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
        )
        assert count_item_in_inventory(state, 0, ItemType.COAL) == 0
        assert count_item_in_inventory(state, 0, ItemType.IRON) == 0

    def test_count_item_single_stack(self, state_factory) -> None:
        """Should count items in a single stack."""
        inv_items = jnp.zeros((1, NUM_INVENTORY_SLOTS), dtype=jnp.int32)
        inv_items = inv_items.at[0, 0].set(ItemType.COAL)
        inv_counts = jnp.zeros((1, NUM_INVENTORY_SLOTS), dtype=jnp.int32)
        inv_counts = inv_counts.at[0, 0].set(10)

        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            inventory_items=inv_items,
            inventory_counts=inv_counts,
        )
        assert count_item_in_inventory(state, 0, ItemType.COAL) == 10

    def test_count_item_multiple_stacks(self, state_factory) -> None:
        """Should sum items across multiple stacks."""
        inv_items = jnp.zeros((1, NUM_INVENTORY_SLOTS), dtype=jnp.int32)
        inv_items = inv_items.at[0, 0].set(ItemType.IRON)
        inv_items = inv_items.at[0, 3].set(ItemType.IRON)
        inv_counts = jnp.zeros((1, NUM_INVENTORY_SLOTS), dtype=jnp.int32)
        inv_counts = inv_counts.at[0, 0].set(5)
        inv_counts = inv_counts.at[0, 3].set(7)

        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            inventory_items=inv_items,
            inventory_counts=inv_counts,
        )
        assert count_item_in_inventory(state, 0, ItemType.IRON) == 12

    def test_add_item_to_empty_inventory(self, state_factory) -> None:
        """Should add item to first empty slot."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
        )
        new_state = add_item_to_inventory(state, 0, ItemType.COAL, 5)

        assert new_state.inventory_items[0, 0] == ItemType.COAL
        assert new_state.inventory_counts[0, 0] == 5

    def test_add_item_stacks_with_existing(self, state_factory) -> None:
        """Should stack with existing items of same type."""
        inv_items = jnp.zeros((1, NUM_INVENTORY_SLOTS), dtype=jnp.int32)
        inv_items = inv_items.at[0, 0].set(ItemType.COAL)
        inv_counts = jnp.zeros((1, NUM_INVENTORY_SLOTS), dtype=jnp.int32)
        inv_counts = inv_counts.at[0, 0].set(10)

        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            inventory_items=inv_items,
            inventory_counts=inv_counts,
        )
        new_state = add_item_to_inventory(state, 0, ItemType.COAL, 5)

        assert new_state.inventory_items[0, 0] == ItemType.COAL
        assert new_state.inventory_counts[0, 0] == 15


class TestRecipeAffordability:
    """Tests for recipe affordability checking."""

    def test_cannot_afford_without_materials(self, state_factory) -> None:
        """Should not afford recipe without materials."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
        )
        assert not can_afford_recipe(state, 0, 0)

    def test_cannot_afford_partial_materials(self, state_factory) -> None:
        """Should not afford recipe with only some materials."""
        inv_items = jnp.zeros((1, NUM_INVENTORY_SLOTS), dtype=jnp.int32)
        inv_items = inv_items.at[0, 0].set(ItemType.COPPER)
        inv_counts = jnp.zeros((1, NUM_INVENTORY_SLOTS), dtype=jnp.int32)
        inv_counts = inv_counts.at[0, 0].set(5)

        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            inventory_items=inv_items,
            inventory_counts=inv_counts,
        )
        assert not can_afford_recipe(state, 0, 0)

    def test_can_afford_with_exact_materials(self, state_factory) -> None:
        """Should afford recipe with exact required materials."""
        inv_items = jnp.zeros((1, NUM_INVENTORY_SLOTS), dtype=jnp.int32)
        inv_items = inv_items.at[0, 0].set(ItemType.COPPER)
        inv_items = inv_items.at[0, 1].set(ItemType.IRON)
        inv_counts = jnp.zeros((1, NUM_INVENTORY_SLOTS), dtype=jnp.int32)
        inv_counts = inv_counts.at[0, 0].set(5)
        inv_counts = inv_counts.at[0, 1].set(5)

        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            inventory_items=inv_items,
            inventory_counts=inv_counts,
        )
        assert can_afford_recipe(state, 0, 0)

    def test_can_afford_with_excess_materials(self, state_factory) -> None:
        """Should afford recipe with more than required materials."""
        inv_items = jnp.zeros((1, NUM_INVENTORY_SLOTS), dtype=jnp.int32)
        inv_items = inv_items.at[0, 0].set(ItemType.COPPER)
        inv_items = inv_items.at[0, 1].set(ItemType.IRON)
        inv_counts = jnp.zeros((1, NUM_INVENTORY_SLOTS), dtype=jnp.int32)
        inv_counts = inv_counts.at[0, 0].set(20)
        inv_counts = inv_counts.at[0, 1].set(10)

        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            inventory_items=inv_items,
            inventory_counts=inv_counts,
        )
        assert can_afford_recipe(state, 0, 0)


class TestMaterialConsumption:
    """Tests for material consumption."""

    def test_consume_removes_exact_amounts(self, state_factory) -> None:
        """Should remove exact amounts required by recipe."""
        inv_items = jnp.zeros((1, NUM_INVENTORY_SLOTS), dtype=jnp.int32)
        inv_items = inv_items.at[0, 0].set(ItemType.COPPER)
        inv_items = inv_items.at[0, 1].set(ItemType.IRON)
        inv_counts = jnp.zeros((1, NUM_INVENTORY_SLOTS), dtype=jnp.int32)
        inv_counts = inv_counts.at[0, 0].set(10)
        inv_counts = inv_counts.at[0, 1].set(10)

        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            inventory_items=inv_items,
            inventory_counts=inv_counts,
        )
        new_state = consume_recipe_materials(state, 0, 0)

        assert new_state.inventory_counts[0, 0] == 5
        assert new_state.inventory_counts[0, 1] == 5


class TestCraftingProgress:
    """Tests for crafting progress."""

    def test_instant_craft_produces_output(self, state_factory) -> None:
        """With ticks=0, crafting should instantly produce the output."""
        inv_items = jnp.zeros((1, NUM_INVENTORY_SLOTS), dtype=jnp.int32)
        inv_items = inv_items.at[0, 0].set(ItemType.COPPER)
        inv_items = inv_items.at[0, 1].set(ItemType.IRON)
        inv_counts = jnp.zeros((1, NUM_INVENTORY_SLOTS), dtype=jnp.int32)
        inv_counts = inv_counts.at[0, 0].set(5)
        inv_counts = inv_counts.at[0, 1].set(5)

        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            inventory_items=inv_items,
            inventory_counts=inv_counts,
        )
        new_state = start_crafting(state, 0, 0)  # recipe 0 = miner

        # Instant: output appears immediately, no craft_progress.
        assert new_state.craft_progress[0] == 0
        assert int(new_state.inventory_items[0, 0]) == ItemType.MINER
        assert new_state.inventory_counts[0, 0] == 1

    def test_instant_craft_consumes_materials(self, state_factory) -> None:
        """Instant crafting should consume all required materials."""
        inv_items = jnp.zeros((1, NUM_INVENTORY_SLOTS), dtype=jnp.int32)
        inv_items = inv_items.at[0, 0].set(ItemType.COPPER)
        inv_items = inv_items.at[0, 1].set(ItemType.IRON)
        inv_counts = jnp.zeros((1, NUM_INVENTORY_SLOTS), dtype=jnp.int32)
        inv_counts = inv_counts.at[0, 0].set(5)
        inv_counts = inv_counts.at[0, 1].set(5)

        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            inventory_items=inv_items,
            inventory_counts=inv_counts,
        )
        new_state = start_crafting(state, 0, 0)  # recipe 0 = miner

        # Copper and iron consumed.
        assert new_state.inventory_counts[0, 1] == 0

    def test_cannot_start_while_crafting(self, state_factory) -> None:
        """Should not start new craft while already crafting."""
        inv_items = jnp.zeros((1, NUM_INVENTORY_SLOTS), dtype=jnp.int32)
        inv_items = inv_items.at[0, 0].set(ItemType.COPPER)
        inv_items = inv_items.at[0, 1].set(ItemType.IRON)
        inv_counts = jnp.zeros((1, NUM_INVENTORY_SLOTS), dtype=jnp.int32)
        inv_counts = inv_counts.at[0, 0].set(10)
        inv_counts = inv_counts.at[0, 1].set(10)
        craft_progress = jnp.array([2], dtype=jnp.int32)

        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            inventory_items=inv_items,
            inventory_counts=inv_counts,
            craft_progress=craft_progress,
        )
        new_state = start_crafting(state, 0, 0)

        assert new_state.craft_progress[0] == 2
        assert new_state.inventory_counts[0, 0] == 10

    def test_update_decrements_progress(self, state_factory) -> None:
        """Update should decrement crafting progress."""
        craft_progress = jnp.array([3], dtype=jnp.int32)

        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            craft_progress=craft_progress,
        )
        new_state = update_crafting(state)

        assert new_state.craft_progress[0] == 2

    def test_update_completes_craft(self, state_factory) -> None:
        """Update should complete craft and add item when progress reaches 0."""
        craft_progress = jnp.array([1], dtype=jnp.int32)
        crafting_recipe = jnp.array([0], dtype=jnp.int32)

        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            craft_progress=craft_progress,
            crafting_recipe=crafting_recipe,
        )
        new_state = update_crafting(state)

        assert new_state.craft_progress[0] == 0
        assert new_state.inventory_items[0, 0] == ItemType.MINER
        assert new_state.inventory_counts[0, 0] == 1

    def test_craft_specific_recipe(self, state_factory) -> None:
        """Direct craft action should produce the specified recipe output."""
        inv_items = jnp.zeros((1, NUM_INVENTORY_SLOTS), dtype=jnp.int32)
        inv_items = inv_items.at[0, 0].set(ItemType.IRON)
        inv_counts = jnp.zeros((1, NUM_INVENTORY_SLOTS), dtype=jnp.int32)
        inv_counts = inv_counts.at[0, 0].set(5)

        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            inventory_items=inv_items,
            inventory_counts=inv_counts,
        )
        new_state = start_crafting(state, 0, 1)  # recipe 1 = chest

        assert int(new_state.inventory_items[0, 0]) == ItemType.CHEST


class TestSlotCycling:
    """Tests for slot cycling."""

    @pytest.mark.parametrize(
        "direction, expected",
        [(1, 1), (-1, NUM_INVENTORY_SLOTS - 1)],
        ids=["forward", "backward"],
    )
    def test_cycle_slot(self, state_factory, direction, expected) -> None:
        """Should cycle slot in the given direction with wrap."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
        )
        new_state = cycle_slot(state, 0, direction)
        assert new_state.selected_slots[0] == expected

