"""Tests for the machine placement system."""

import jax
import jax.numpy as jnp
import pytest

from factoriax import BlockType, Direction, ItemType
from factoriax.constants import NUM_ITEM_TYPES, MachineType
from factoriax.envs.factoriax_env import FactoriaXEnv
from factoriax.placement import (
    get_tile_in_front,
    is_placeable_item,
    is_valid_placement_tile,
    place_machine,
)
from factoriax.state import EnvParams


class TestEntHealth:
    """Tests for the per-entity health field."""

    def test_reset_env_initializes_ent_health_zeros(self) -> None:
        """After reset, ``ent_health`` exists with the right shape and
        every slot is zero (no entities placed yet)."""
        env = FactoriaXEnv()
        params = EnvParams(map_width=8, map_height=8, num_players=1)
        _, state = env.reset_env(jax.random.key(0), params)
        mm = params.resolved_max_machines()
        assert state.ent_health.shape == (mm,)
        assert state.ent_health.dtype == jnp.int16
        assert bool(jnp.all(state.ent_health == 0))

    def test_state_factory_initializes_ent_health_zeros(self, state_factory) -> None:
        """``state_factory`` exposes the new field so existing tests
        keep building EnvState without modification."""
        state = state_factory(
            world_map=jnp.full((4, 4), BlockType.DIRT, dtype=jnp.int32),
            max_machines=16,
        )
        assert state.ent_health.shape == (16,)
        assert state.ent_health.dtype == jnp.int16
        assert bool(jnp.all(state.ent_health == 0))


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
            (ItemType.IRON_ORE, False),
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
        inv = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int32)
        inv = inv.at[0, ItemType.MINER].set(1)

        state = state_factory(
            world_map=jnp.array(
                [[BlockType.DIRT, BlockType.DIRT], [BlockType.DIRT, BlockType.DIRT]],
                dtype=jnp.int32,
            ),
            player_position=(0, 0),
            player_direction=Direction.RIGHT,
            player_inventory=inv,
        )
        new_state = place_machine(state, 0, int(ItemType.MINER))

        assert new_state.machine_types[0, 1] == MachineType.MINER
        assert new_state.player_inventory[0, ItemType.MINER] == 0

    def test_place_machine_decrements_stack(self, state_factory) -> None:
        """Should decrement stack count when placing."""
        inv = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int32)
        inv = inv.at[0, ItemType.MINER].set(3)

        state = state_factory(
            world_map=jnp.array(
                [[BlockType.DIRT, BlockType.DIRT], [BlockType.DIRT, BlockType.DIRT]],
                dtype=jnp.int32,
            ),
            player_position=(0, 0),
            player_direction=Direction.RIGHT,
            player_inventory=inv,
        )
        new_state = place_machine(state, 0, int(ItemType.MINER))

        assert new_state.player_inventory[0, ItemType.MINER] == 2

    def test_cannot_place_on_water(self, state_factory) -> None:
        """Should not place machine on water."""
        inv = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int32)
        inv = inv.at[0, ItemType.MINER].set(1)

        state = state_factory(
            world_map=jnp.array(
                [[BlockType.DIRT, BlockType.WATER], [BlockType.DIRT, BlockType.DIRT]],
                dtype=jnp.int32,
            ),
            player_position=(0, 0),
            player_direction=Direction.RIGHT,
            player_inventory=inv,
        )
        new_state = place_machine(state, 0, int(ItemType.MINER))

        assert new_state.machine_types[0, 1] == MachineType.NONE
        assert new_state.player_inventory[0, ItemType.MINER] == 1

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
        new_state = place_machine(state, 0, int(ItemType.MINER))

        assert new_state.machine_types[0, 1] == MachineType.NONE

    def test_cannot_place_non_placeable_item(self, state_factory) -> None:
        """Should not place non-placeable items."""
        inv = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int32)
        inv = inv.at[0, ItemType.COAL].set(5)

        state = state_factory(
            world_map=jnp.array(
                [[BlockType.DIRT, BlockType.DIRT], [BlockType.DIRT, BlockType.DIRT]],
                dtype=jnp.int32,
            ),
            player_position=(0, 0),
            player_direction=Direction.RIGHT,
            player_inventory=inv,
        )
        new_state = place_machine(state, 0, int(ItemType.COAL))

        assert new_state.machine_types[0, 1] == MachineType.NONE
        assert new_state.player_inventory[0, ItemType.COAL] == 5
