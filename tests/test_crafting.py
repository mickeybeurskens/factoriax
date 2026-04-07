"""Tests for the crafting system (pouch inventory model)."""

import jax.numpy as jnp

from factoriax import BlockType, ItemType
from factoriax.constants import NUM_ITEM_TYPES
from factoriax.crafting import (
    add_item_to_player,
    can_afford_recipe,
    consume_recipe_materials,
    start_crafting,
    update_crafting,
)


def _inv_with(item_type: int, count: int) -> jnp.ndarray:
    """Build a single-player pouch inventory with one item type set."""
    inv = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int32)
    return inv.at[0, item_type].set(count)


def _inv_with_multi(**items: int) -> jnp.ndarray:
    """Build a single-player pouch with multiple item types."""
    inv = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int32)
    name_to_type = {m.name: int(m) for m in ItemType}
    for name, count in items.items():
        inv = inv.at[0, name_to_type[name]].set(count)
    return inv


class TestInventoryHelpers:
    """Tests for pouch inventory helper functions."""

    def test_count_item_empty_inventory(self, state_factory) -> None:
        """Empty inventory has zero count for all items."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
        )
        assert state.player_inventory[0, ItemType.COAL] == 0
        assert state.player_inventory[0, ItemType.IRON] == 0

    def test_add_item_to_empty_inventory(self, state_factory) -> None:
        """Should add item count to the correct type slot."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
        )
        new_state = add_item_to_player(state, 0, ItemType.COAL, 5)
        assert new_state.player_inventory[0, ItemType.COAL] == 5

    def test_add_item_stacks_with_existing(self, state_factory) -> None:
        """Should add to existing count of same type."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            player_inventory=_inv_with(ItemType.COAL, 10),
        )
        new_state = add_item_to_player(state, 0, ItemType.COAL, 5)
        assert new_state.player_inventory[0, ItemType.COAL] == 15


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
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            player_inventory=_inv_with(ItemType.COPPER, 5),
        )
        # Recipe 0 (miner) needs 5 copper + 5 iron.
        assert not can_afford_recipe(state, 0, 0)

    def test_can_afford_with_exact_materials(self, state_factory) -> None:
        """Should afford recipe with exact required materials."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            player_inventory=_inv_with_multi(COPPER=5, IRON=5),
        )
        assert can_afford_recipe(state, 0, 0)

    def test_can_afford_with_excess_materials(self, state_factory) -> None:
        """Should afford recipe with more than required materials."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            player_inventory=_inv_with_multi(COPPER=20, IRON=10),
        )
        assert can_afford_recipe(state, 0, 0)


class TestMaterialConsumption:
    """Tests for material consumption."""

    def test_consume_removes_exact_amounts(self, state_factory) -> None:
        """Should remove exact amounts required by recipe."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            player_inventory=_inv_with_multi(COPPER=10, IRON=10),
        )
        # Recipe 0 (miner): 5 copper + 5 iron.
        new_state = consume_recipe_materials(state, 0, 0)
        assert new_state.player_inventory[0, ItemType.COPPER] == 5
        assert new_state.player_inventory[0, ItemType.IRON] == 5


class TestCraftingProgress:
    """Tests for crafting progress."""

    def test_instant_craft_produces_output(self, state_factory) -> None:
        """With ticks=0, crafting should instantly produce the output."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            player_inventory=_inv_with_multi(COPPER=5, IRON=5),
        )
        new_state = start_crafting(state, 0, 0)  # recipe 0 = miner
        assert new_state.craft_progress[0] == 0
        assert new_state.player_inventory[0, ItemType.MINER] == 1

    def test_instant_craft_consumes_materials(
        self, state_factory,
    ) -> None:
        """Instant crafting should consume all required materials."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            player_inventory=_inv_with_multi(COPPER=5, IRON=5),
        )
        new_state = start_crafting(state, 0, 0)
        assert new_state.player_inventory[0, ItemType.COPPER] == 0
        assert new_state.player_inventory[0, ItemType.IRON] == 0

    def test_cannot_start_while_crafting(self, state_factory) -> None:
        """Should not start new craft while already crafting."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            player_inventory=_inv_with_multi(COPPER=10, IRON=10),
            craft_progress=jnp.array([2], dtype=jnp.int32),
        )
        new_state = start_crafting(state, 0, 0)
        assert new_state.craft_progress[0] == 2
        assert new_state.player_inventory[0, ItemType.COPPER] == 10

    def test_update_decrements_progress(self, state_factory) -> None:
        """Update should decrement crafting progress."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            craft_progress=jnp.array([3], dtype=jnp.int32),
        )
        new_state = update_crafting(state)
        assert new_state.craft_progress[0] == 2

    def test_update_completes_craft(self, state_factory) -> None:
        """Update should complete craft and add item when done."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            craft_progress=jnp.array([1], dtype=jnp.int32),
            crafting_recipe=jnp.array([0], dtype=jnp.int32),
        )
        new_state = update_crafting(state)
        assert new_state.craft_progress[0] == 0
        assert new_state.player_inventory[0, ItemType.MINER] == 1

    def test_craft_specific_recipe(self, state_factory) -> None:
        """Direct craft action should produce the specified output."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            player_inventory=_inv_with(ItemType.IRON, 5),
        )
        new_state = start_crafting(state, 0, 1)  # recipe 1 = chest
        assert new_state.player_inventory[0, ItemType.CHEST] == 1
