"""Tests for factoriax.play.transfer -- inventory type swap."""

from __future__ import annotations

import jax.numpy as jnp

from factoriax.constants import NUM_ITEM_TYPES, ItemType
from factoriax.play.transfer import swap_inventory_slots

# ---------------------------------------------------------------------------
# swap_inventory_slots
# ---------------------------------------------------------------------------


class TestSwapInventorySlots:
    """Tests for the swap_inventory_slots helper (pouch model)."""

    def test_swap_two_occupied_types(self, state_factory) -> None:
        """Swapping two item types exchanges their counts."""
        inv = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int32)
        inv = inv.at[0, int(ItemType.COAL)].set(5)
        inv = inv.at[0, int(ItemType.IRON)].set(10)
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            player_inventory=inv,
        )
        result = swap_inventory_slots(state, 0, int(ItemType.COAL), int(ItemType.IRON))
        assert int(result.player_inventory[0, int(ItemType.COAL)]) == 10
        assert int(result.player_inventory[0, int(ItemType.IRON)]) == 5

    def test_swap_occupied_with_empty(self, state_factory) -> None:
        """Swapping an occupied type with an empty one moves the count."""
        inv = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int32)
        inv = inv.at[0, int(ItemType.COPPER)].set(3)
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            player_inventory=inv,
        )
        result = swap_inventory_slots(
            state, 0, int(ItemType.COPPER), int(ItemType.MINER),
        )
        assert int(result.player_inventory[0, int(ItemType.COPPER)]) == 0
        assert int(result.player_inventory[0, int(ItemType.MINER)]) == 3

    def test_same_type_is_noop(self, state_factory) -> None:
        """Swapping a type with itself returns the same state object."""
        inv = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int32)
        inv = inv.at[0, int(ItemType.COAL)].set(5)
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            player_inventory=inv,
        )
        result = swap_inventory_slots(state, 0, int(ItemType.COAL), int(ItemType.COAL))
        assert result is state

    def test_swap_two_empty_types(self, state_factory) -> None:
        """Swapping two zero-count types leaves both at zero."""
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
        )
        result = swap_inventory_slots(
            state, 0, int(ItemType.HULL), int(ItemType.ROCKET),
        )
        assert int(result.player_inventory[0, int(ItemType.HULL)]) == 0
        assert int(result.player_inventory[0, int(ItemType.ROCKET)]) == 0

    def test_swap_preserves_total(self, state_factory) -> None:
        """Swapping two types preserves the total inventory count."""
        inv = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int32)
        inv = inv.at[0, int(ItemType.COAL)].set(10)
        inv = inv.at[0, int(ItemType.IRON)].set(20)
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            player_inventory=inv,
        )
        total_before = int(jnp.sum(state.player_inventory))
        result = swap_inventory_slots(state, 0, int(ItemType.COAL), int(ItemType.IRON))
        total_after = int(jnp.sum(result.player_inventory))
        assert total_before == total_after

    def test_swap_nonadjacent_types(self, state_factory) -> None:
        """Swapping non-adjacent item types works correctly."""
        inv = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int32)
        inv = inv.at[0, int(ItemType.MINER)].set(7)
        inv = inv.at[0, int(ItemType.ASSEMBLER)].set(15)
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            player_inventory=inv,
        )
        result = swap_inventory_slots(
            state, 0, int(ItemType.MINER), int(ItemType.ASSEMBLER),
        )
        assert int(result.player_inventory[0, int(ItemType.MINER)]) == 15
        assert int(result.player_inventory[0, int(ItemType.ASSEMBLER)]) == 7

    def test_swap_does_not_affect_other_types(self, state_factory) -> None:
        """Swapping two types does not modify any other type counts."""
        inv = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int32)
        inv = inv.at[0, int(ItemType.COAL)].set(5)
        inv = inv.at[0, int(ItemType.IRON)].set(10)
        inv = inv.at[0, int(ItemType.COPPER)].set(99)
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            player_inventory=inv,
        )
        result = swap_inventory_slots(state, 0, int(ItemType.COAL), int(ItemType.IRON))
        assert int(result.player_inventory[0, int(ItemType.COPPER)]) == 99
