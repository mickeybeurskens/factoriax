"""Tests for the machine placement system."""

import jax.numpy as jnp
import pytest

from factoriax import BlockType, Direction, ItemType
from factoriax.constants import NUM_INVENTORY_SLOTS, MachineType
from factoriax.placement import (
    get_tile_in_front,
    is_placeable_item,
    is_valid_placement_tile,
    place_machine,
)


class TestDirectionOffsets:
    """Tests for direction-based tile lookup."""

    @pytest.mark.parametrize(
        "direction, expected_x, expected_y",
        [
            (Direction.UP, 1, 0),
            (Direction.DOWN, 1, 2),
            (Direction.LEFT, 0, 1),
            (Direction.RIGHT, 2, 1),
        ],
        ids=["up", "down", "left", "right"],
    )
    def test_tile_in_front(
        self, state_factory, direction, expected_x, expected_y
    ) -> None:
        """Should return the correct adjacent tile for the given direction."""
        state = state_factory(
            world_map=jnp.full((3, 3), BlockType.DIRT, dtype=jnp.int32),
            player_position=(1, 1),
            player_direction=direction,
        )
        x, y = get_tile_in_front(state, 0)
        assert int(x) == expected_x
        assert int(y) == expected_y


class TestPlacementValidation:
    """Tests for placement validation."""

    def test_valid_placement_on_dirt(self, state_factory) -> None:
        """Should allow placement on dirt."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
        )
        assert is_valid_placement_tile(state, 0, 0)

    def test_invalid_placement_on_water(self, state_factory) -> None:
        """Should not allow placement on water."""
        state = state_factory(
            world_map=jnp.array([[BlockType.WATER]], dtype=jnp.int32),
        )
        assert not is_valid_placement_tile(state, 0, 0)

    def test_invalid_placement_out_of_bounds(self, state_factory) -> None:
        """Should not allow placement out of bounds."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
        )
        assert not is_valid_placement_tile(state, -1, 0)
        assert not is_valid_placement_tile(state, 1, 0)

    def test_invalid_placement_on_existing_machine(self, state_factory) -> None:
        """Should not allow placement where machine exists."""
        machine_types = jnp.array([[MachineType.MINER]], dtype=jnp.int32)
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            machine_types=machine_types,
        )
        assert not is_valid_placement_tile(state, 0, 0)


class TestPlaceableItems:
    """Tests for placeable item checking."""

    @pytest.mark.parametrize(
        "item, expected",
        [
            (ItemType.MINER, True),
            (ItemType.COAL, False),
            (ItemType.IRON, False),
            (ItemType.EMPTY, False),
        ],
        ids=["miner", "coal", "iron", "empty"],
    )
    def test_is_placeable(self, item, expected) -> None:
        """Item placeability should match the expected value."""
        assert is_placeable_item(item) == expected


class TestMachinePlacement:
    """Tests for machine placement."""

    def test_place_machine_from_inventory(self, state_factory) -> None:
        """Should place machine and remove from inventory."""
        inv_items = jnp.zeros((1, NUM_INVENTORY_SLOTS), dtype=jnp.int32)
        inv_items = inv_items.at[0, 0].set(ItemType.MINER)
        inv_counts = jnp.zeros((1, NUM_INVENTORY_SLOTS), dtype=jnp.int32)
        inv_counts = inv_counts.at[0, 0].set(1)

        state = state_factory(
            world_map=jnp.array(
                [[BlockType.DIRT, BlockType.DIRT], [BlockType.DIRT, BlockType.DIRT]],
                dtype=jnp.int32,
            ),
            player_position=(0, 0),
            player_direction=Direction.RIGHT,
            inventory_items=inv_items,
            inventory_counts=inv_counts,
        )
        new_state = place_machine(state, 0)

        assert new_state.machine_types[0, 1] == MachineType.MINER
        assert new_state.inventory_counts[0, 0] == 0
        assert new_state.inventory_items[0, 0] == ItemType.EMPTY

    def test_place_machine_decrements_stack(self, state_factory) -> None:
        """Should decrement stack count when placing."""
        inv_items = jnp.zeros((1, NUM_INVENTORY_SLOTS), dtype=jnp.int32)
        inv_items = inv_items.at[0, 0].set(ItemType.MINER)
        inv_counts = jnp.zeros((1, NUM_INVENTORY_SLOTS), dtype=jnp.int32)
        inv_counts = inv_counts.at[0, 0].set(3)

        state = state_factory(
            world_map=jnp.array(
                [[BlockType.DIRT, BlockType.DIRT], [BlockType.DIRT, BlockType.DIRT]],
                dtype=jnp.int32,
            ),
            player_position=(0, 0),
            player_direction=Direction.RIGHT,
            inventory_items=inv_items,
            inventory_counts=inv_counts,
        )
        new_state = place_machine(state, 0)

        assert new_state.inventory_counts[0, 0] == 2
        assert new_state.inventory_items[0, 0] == ItemType.MINER

    def test_cannot_place_on_water(self, state_factory) -> None:
        """Should not place machine on water."""
        inv_items = jnp.zeros((1, NUM_INVENTORY_SLOTS), dtype=jnp.int32)
        inv_items = inv_items.at[0, 0].set(ItemType.MINER)
        inv_counts = jnp.zeros((1, NUM_INVENTORY_SLOTS), dtype=jnp.int32)
        inv_counts = inv_counts.at[0, 0].set(1)

        state = state_factory(
            world_map=jnp.array(
                [[BlockType.DIRT, BlockType.WATER], [BlockType.DIRT, BlockType.DIRT]],
                dtype=jnp.int32,
            ),
            player_position=(0, 0),
            player_direction=Direction.RIGHT,
            inventory_items=inv_items,
            inventory_counts=inv_counts,
        )
        new_state = place_machine(state, 0)

        assert new_state.machine_types[0, 1] == MachineType.NONE
        assert new_state.inventory_counts[0, 0] == 1

    def test_cannot_place_without_item(self, state_factory) -> None:
        """Should not place machine without item in inventory."""
        state = state_factory(
            world_map=jnp.array(
                [[BlockType.DIRT, BlockType.DIRT], [BlockType.DIRT, BlockType.DIRT]],
                dtype=jnp.int32,
            ),
            player_position=(0, 0),
            player_direction=Direction.RIGHT,
        )
        new_state = place_machine(state, 0)

        assert new_state.machine_types[0, 1] == MachineType.NONE

    def test_cannot_place_non_placeable_item(self, state_factory) -> None:
        """Should not place non-placeable items."""
        inv_items = jnp.zeros((1, NUM_INVENTORY_SLOTS), dtype=jnp.int32)
        inv_items = inv_items.at[0, 0].set(ItemType.COAL)
        inv_counts = jnp.zeros((1, NUM_INVENTORY_SLOTS), dtype=jnp.int32)
        inv_counts = inv_counts.at[0, 0].set(5)

        state = state_factory(
            world_map=jnp.array(
                [[BlockType.DIRT, BlockType.DIRT], [BlockType.DIRT, BlockType.DIRT]],
                dtype=jnp.int32,
            ),
            player_position=(0, 0),
            player_direction=Direction.RIGHT,
            inventory_items=inv_items,
            inventory_counts=inv_counts,
        )
        new_state = place_machine(state, 0)

        assert new_state.machine_types[0, 1] == MachineType.NONE
        assert new_state.inventory_counts[0, 0] == 5

    def test_uses_selected_slot(self, state_factory) -> None:
        """Should use the selected inventory slot."""
        inv_items = jnp.zeros((1, NUM_INVENTORY_SLOTS), dtype=jnp.int32)
        inv_items = inv_items.at[0, 2].set(ItemType.MINER)
        inv_counts = jnp.zeros((1, NUM_INVENTORY_SLOTS), dtype=jnp.int32)
        inv_counts = inv_counts.at[0, 2].set(1)
        selected_slots = jnp.array([2], dtype=jnp.int32)

        state = state_factory(
            world_map=jnp.array(
                [[BlockType.DIRT, BlockType.DIRT], [BlockType.DIRT, BlockType.DIRT]],
                dtype=jnp.int32,
            ),
            player_position=(0, 0),
            player_direction=Direction.RIGHT,
            inventory_items=inv_items,
            inventory_counts=inv_counts,
            selected_slots=selected_slots,
        )
        new_state = place_machine(state, 0)

        assert new_state.machine_types[0, 1] == MachineType.MINER
        assert new_state.inventory_counts[0, 2] == 0
