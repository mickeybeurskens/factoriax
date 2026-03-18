"""Tests for the machine placement system."""

import jax.numpy as jnp

from factoriax import Action, BlockType, ItemType
from factoriax.constants import NUM_INVENTORY_SLOTS, MachineType
from factoriax.placement import (
    get_tile_in_front,
    is_placeable_item,
    is_valid_placement_tile,
    place_machine,
)


class TestDirectionOffsets:
    """Tests for direction-based tile lookup."""

    def test_tile_in_front_facing_up(self, state_factory) -> None:
        """Should return tile above player when facing up."""
        state = state_factory(
            world_map=jnp.array(
                [
                    [BlockType.DIRT, BlockType.DIRT, BlockType.DIRT],
                    [BlockType.DIRT, BlockType.DIRT, BlockType.DIRT],
                    [BlockType.DIRT, BlockType.DIRT, BlockType.DIRT],
                ],
                dtype=jnp.int32,
            ),
            player_position=(1, 1),
            player_direction=Action.UP,
        )
        x, y = get_tile_in_front(state, 0)
        assert int(x) == 1
        assert int(y) == 0

    def test_tile_in_front_facing_down(self, state_factory) -> None:
        """Should return tile below player when facing down."""
        state = state_factory(
            world_map=jnp.array(
                [
                    [BlockType.DIRT, BlockType.DIRT, BlockType.DIRT],
                    [BlockType.DIRT, BlockType.DIRT, BlockType.DIRT],
                    [BlockType.DIRT, BlockType.DIRT, BlockType.DIRT],
                ],
                dtype=jnp.int32,
            ),
            player_position=(1, 1),
            player_direction=Action.DOWN,
        )
        x, y = get_tile_in_front(state, 0)
        assert int(x) == 1
        assert int(y) == 2

    def test_tile_in_front_facing_left(self, state_factory) -> None:
        """Should return tile left of player when facing left."""
        state = state_factory(
            world_map=jnp.array(
                [
                    [BlockType.DIRT, BlockType.DIRT, BlockType.DIRT],
                    [BlockType.DIRT, BlockType.DIRT, BlockType.DIRT],
                    [BlockType.DIRT, BlockType.DIRT, BlockType.DIRT],
                ],
                dtype=jnp.int32,
            ),
            player_position=(1, 1),
            player_direction=Action.LEFT,
        )
        x, y = get_tile_in_front(state, 0)
        assert int(x) == 0
        assert int(y) == 1

    def test_tile_in_front_facing_right(self, state_factory) -> None:
        """Should return tile right of player when facing right."""
        state = state_factory(
            world_map=jnp.array(
                [
                    [BlockType.DIRT, BlockType.DIRT, BlockType.DIRT],
                    [BlockType.DIRT, BlockType.DIRT, BlockType.DIRT],
                    [BlockType.DIRT, BlockType.DIRT, BlockType.DIRT],
                ],
                dtype=jnp.int32,
            ),
            player_position=(1, 1),
            player_direction=Action.RIGHT,
        )
        x, y = get_tile_in_front(state, 0)
        assert int(x) == 2
        assert int(y) == 1


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

    def test_miner_is_placeable(self) -> None:
        """MINER item should be placeable."""
        assert is_placeable_item(ItemType.MINER)

    def test_coal_is_not_placeable(self) -> None:
        """COAL item should not be placeable."""
        assert not is_placeable_item(ItemType.COAL)

    def test_iron_is_not_placeable(self) -> None:
        """IRON item should not be placeable."""
        assert not is_placeable_item(ItemType.IRON)

    def test_empty_is_not_placeable(self) -> None:
        """EMPTY item should not be placeable."""
        assert not is_placeable_item(ItemType.EMPTY)


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
            player_direction=Action.RIGHT,
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
            player_direction=Action.RIGHT,
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
            player_direction=Action.RIGHT,
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
            player_direction=Action.RIGHT,
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
            player_direction=Action.RIGHT,
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
            player_direction=Action.RIGHT,
            inventory_items=inv_items,
            inventory_counts=inv_counts,
            selected_slots=selected_slots,
        )
        new_state = place_machine(state, 0)

        assert new_state.machine_types[0, 1] == MachineType.MINER
        assert new_state.inventory_counts[0, 2] == 0
