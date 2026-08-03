"""Tests for :mod:`factoriax.engine.placement`.

Placement writes a machine into the entity arrays. These tests cover which
items are placeable, which tiles accept them, what a fresh entity holds, and
the health gate that pickup and repair share.
"""

import jax
import jax.numpy as jnp
import pytest

from factoriax.engine.constants import (
    NUM_ACTIONS,
    NUM_ITEM_TYPES,
    PLACEABLE_ITEM_LIST,
    Action,
    BlockType,
    Direction,
    InteractAction,
    ItemType,
    Machine,
    MoveAction,
)
from factoriax.engine.envs.base import FactoriaxEnv
from factoriax.engine.placement import (
    apply_repair,
    get_tile_in_front,
    is_placeable_item,
    is_valid_placement_tile,
    pickup_machine,
    place_machine,
)
from factoriax.engine.state import EnvParams
from factoriax.engine.step import factoriax_step
from factoriax.engine.tables import (
    ITEM_TO_MACHINE_ARRAY,
    MACHINE_HEALTH,
    MACHINE_TO_ITEM_ARRAY,
)


def test_placeable_items_are_exactly_the_non_none_machines() -> None:
    """The placeable set is exactly the items of the non-NONE machines.

    Every placeable item maps to a real machine, and every real machine's
    item is placeable. This anchors the placeable definition on
    ``Machine`` so that the two cannot drift apart.
    """
    placeable = set(PLACEABLE_ITEM_LIST)
    for item in placeable:
        assert int(ITEM_TO_MACHINE_ARRAY[item]) != int(Machine.NONE), (
            f"placeable item {ItemType(item).name} maps to no machine"
        )
    machine_items = {
        int(MACHINE_TO_ITEM_ARRAY[int(m)]) for m in Machine if m != Machine.NONE
    }
    assert machine_items == placeable


def test_place_dispatch_covers_exactly_the_placeable_items() -> None:
    """The PLACE_* dispatch table holds exactly the placeable items.

    ``step.PLACE_ACTION_TO_ITEM`` (the PLACE_* action-offset -> item
    dispatch order) and ``constants.PLACEABLE_ITEM_LIST`` (the placeable
    definition) are maintained in separate modules. They carry the same
    items for different reasons -- one is interface ordering, one is the
    definition -- so we guard the *set*, not the order: dispatch order is an
    interface concern free to differ, but neither side can gain or drop a
    placeable item without the other.
    """
    from factoriax.engine.step import PLACE_ACTION_TO_ITEM

    dispatch_items = {int(i) for i in PLACE_ACTION_TO_ITEM.tolist()}
    assert dispatch_items == set(PLACEABLE_ITEM_LIST)


class TestEntHealth:
    """Tests for the per-entity health field."""

    def test_reset_env_initializes_ent_health_zeros(self) -> None:
        """After reset, ``ent_health`` exists with the right shape and
        every slot is zero (no entities placed yet)."""
        env = FactoriaxEnv(map_width=8, map_height=8)
        params = EnvParams()
        _, state = env.reset_env(jax.random.key(0), params)
        mm = max(64, 8 * 8 // 4)
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
        """The lookup returns the correct adjacent tile for a direction."""
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
        """Placement on dirt is allowed."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
        )
        assert is_valid_placement_tile(state, 0, 0)

    def test_invalid_placement_on_water(self, state_factory) -> None:
        """Placement on water is refused."""
        state = state_factory(
            world_map=jnp.array([[BlockType.WATER]], dtype=jnp.int32),
        )
        assert not is_valid_placement_tile(state, 0, 0)

    def test_invalid_placement_out_of_bounds(self, state_factory) -> None:
        """Placement out of bounds is refused."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
        )
        assert not is_valid_placement_tile(state, -1, 0)
        assert not is_valid_placement_tile(state, 1, 0)

    def test_invalid_placement_on_existing_machine(self, state_factory) -> None:
        """Placement on a tile that already holds a machine is refused."""
        machine_types = jnp.array([[Machine.MINER]], dtype=jnp.int32)
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
        """The placeability of an item matches the expected value."""
        assert is_placeable_item(item) == expected


class TestMachinePlacement:
    """Tests for machine placement."""

    def test_place_machine_from_inventory(self, state_factory) -> None:
        """The action places the machine and removes it from the inventory."""
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
        new_state = place_machine(state, EnvParams(), 0, int(ItemType.MINER))

        assert new_state.machine_types[0, 1] == Machine.MINER
        assert new_state.player_inventory[0, ItemType.MINER] == 0

    def test_place_machine_decrements_stack(self, state_factory) -> None:
        """A placement decrements the stack count."""
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
        new_state = place_machine(state, EnvParams(), 0, int(ItemType.MINER))

        assert new_state.player_inventory[0, ItemType.MINER] == 2

    def test_cannot_place_on_water(self, state_factory) -> None:
        """The action does not place a machine on water."""
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
        new_state = place_machine(state, EnvParams(), 0, int(ItemType.MINER))

        assert new_state.machine_types[0, 1] == Machine.NONE
        assert new_state.player_inventory[0, ItemType.MINER] == 1

    def test_cannot_place_without_item(self, state_factory) -> None:
        """The action does not place a machine without the item in the inventory."""
        state = state_factory(
            world_map=jnp.array(
                [[BlockType.DIRT, BlockType.DIRT], [BlockType.DIRT, BlockType.DIRT]],
                dtype=jnp.int32,
            ),
            player_position=(0, 0),
            player_direction=Direction.RIGHT,
        )
        new_state = place_machine(state, EnvParams(), 0, int(ItemType.MINER))

        assert new_state.machine_types[0, 1] == Machine.NONE

    def test_cannot_place_non_placeable_item(self, state_factory) -> None:
        """The action does not place non-placeable items."""
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
        new_state = place_machine(state, EnvParams(), 0, int(ItemType.COAL))

        assert new_state.machine_types[0, 1] == Machine.NONE
        assert new_state.player_inventory[0, ItemType.COAL] == 5


class TestPlacementClearsInheritedBuffer:
    """Regression: a freshly-placed machine must start with an empty buffer.

    ``place_machine`` reuses the first inactive entity slot. If that slot
    carries stale buffer contents, for example ore that leaked in while it was
    inactive), the new machine must not inherit it.
    """

    def test_place_machine_clears_polluted_slot(self, state_factory) -> None:
        """A machine placed into a polluted slot starts empty."""
        inv = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int32)
        inv = inv.at[0, ItemType.MINER].set(1)

        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT, BlockType.DIRT]], dtype=jnp.int32),
            player_position=(0, 0),
            player_direction=Direction.RIGHT,
            player_inventory=inv,
        )
        # No machines on the grid, so slot 0 is the first inactive slot
        # that place_machine will allocate. Pollute its buffer.
        state = state.replace(
            ent_buf_type=state.ent_buf_type.at[0].set(jnp.int8(int(ItemType.COAL))),
            ent_buf_count=state.ent_buf_count.at[0].set(jnp.int16(99)),
        )

        new_state = place_machine(state, EnvParams(), 0, int(ItemType.MINER))

        assert int(new_state.tile_entity[0, 1]) == 0
        assert new_state.machine_types[0, 1] == Machine.MINER
        assert int(new_state.ent_buf_count[0]) == 0
        assert int(new_state.ent_buf_type[0]) == 0


class TestPlacementInitializesHealth:
    """Placement must seed ent_health to the configured max for the type."""

    def _miner_state(self, state_factory):
        inv = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int32)
        inv = inv.at[0, ItemType.MINER].set(1)
        return state_factory(
            world_map=jnp.array(
                [[BlockType.DIRT, BlockType.DIRT], [BlockType.DIRT, BlockType.DIRT]],
                dtype=jnp.int32,
            ),
            player_position=(0, 0),
            player_direction=Direction.RIGHT,
            player_inventory=inv,
        )

    def test_place_initializes_full_health_default(self, state_factory) -> None:
        """A freshly placed machine has ``MACHINE_HEALTH`` HP under defaults."""
        state = self._miner_state(state_factory)
        params = EnvParams()
        new_state = place_machine(state, params, 0, int(ItemType.MINER))
        eidx = int(new_state.tile_entity[0, 1])
        assert eidx >= 0
        assert int(new_state.ent_health[eidx]) == MACHINE_HEALTH

    def test_failed_placement_leaves_ent_health_unchanged(self, state_factory) -> None:
        """If placement is rejected (no item), ent_health is untouched."""
        state = state_factory(
            world_map=jnp.array(
                [[BlockType.DIRT, BlockType.DIRT], [BlockType.DIRT, BlockType.DIRT]],
                dtype=jnp.int32,
            ),
            player_position=(0, 0),
            player_direction=Direction.RIGHT,
        )
        params = EnvParams()
        new_state = place_machine(state, params, 0, int(ItemType.MINER))
        assert bool(jnp.all(new_state.ent_health == 0))

    def test_build_state_initializes_pre_placed_to_full_health(self) -> None:
        """Levels with pre-placed machines start them at full HP."""
        from factoriax.engine.levels import LevelBuilder, build_state

        level = (
            LevelBuilder(4, 4)
            .place_machine(2, 2, int(Machine.FURNACE), int(Direction.UP))
            .build("hp_init_test")
        )
        state = build_state(level, num_players=1)
        eidx = int(state.tile_entity[2, 2])
        assert eidx >= 0
        assert int(state.ent_health[eidx]) == MACHINE_HEALTH


class TestActionRepair:
    """Tests for Action.REPAIR and apply_repair."""

    def test_action_repair_value(self) -> None:
        """REPAIR is a fixed interaction action, derived from its category."""
        assert "REPAIR" in Action.__members__
        assert 0 <= int(Action.REPAIR) < NUM_ACTIONS
        # Its flat value falls out of the InteractAction offset within the
        # composed layout (MoveAction block, then InteractAction block).
        assert int(Action.REPAIR) == len(MoveAction) + int(InteractAction.REPAIR)

    def _placed_state(self, state_factory):
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
        params = EnvParams()
        placed = place_machine(state, params, 0, int(ItemType.MINER))
        eidx = int(placed.tile_entity[0, 1])
        return placed, params, eidx

    def test_apply_repair_restores_full_health(self, state_factory) -> None:
        """Calling apply_repair on a damaged target restores full HP."""
        state, params, eidx = self._placed_state(state_factory)
        damaged = state.replace(ent_health=state.ent_health.at[eidx].set(10))
        repaired = apply_repair(damaged, params, 0)
        assert int(repaired.ent_health[eidx]) == MACHINE_HEALTH

    def test_apply_repair_noop_on_full_health(self, state_factory) -> None:
        """Repair on a full-HP entity leaves ent_health untouched."""
        state, params, _ = self._placed_state(state_factory)
        repaired = apply_repair(state, params, 0)
        assert bool(jnp.all(repaired.ent_health == state.ent_health))

    def test_apply_repair_noop_on_empty_tile(self, state_factory) -> None:
        """Repair facing an empty tile leaves ent_health untouched."""
        state = state_factory(
            world_map=jnp.array(
                [[BlockType.DIRT, BlockType.DIRT], [BlockType.DIRT, BlockType.DIRT]],
                dtype=jnp.int32,
            ),
            player_position=(0, 0),
            player_direction=Direction.RIGHT,
        )
        params = EnvParams()
        repaired = apply_repair(state, params, 0)
        assert bool(jnp.all(repaired.ent_health == state.ent_health))

    def test_apply_repair_noop_out_of_bounds(self, state_factory) -> None:
        """Repair facing OOB is a no-op (no exception, no state change)."""
        state = state_factory(
            world_map=jnp.array(
                [[BlockType.DIRT, BlockType.DIRT], [BlockType.DIRT, BlockType.DIRT]],
                dtype=jnp.int32,
            ),
            player_position=(0, 0),
            player_direction=Direction.UP,  # facing y=-1
        )
        params = EnvParams()
        repaired = apply_repair(state, params, 0)
        assert bool(jnp.all(repaired.ent_health == state.ent_health))

    def test_repair_action_dispatch_restores_health(self, state_factory) -> None:
        """A full step with Action.REPAIR routes to apply_repair."""
        state, params, eidx = self._placed_state(state_factory)
        damaged = state.replace(ent_health=state.ent_health.at[eidx].set(7))
        new_state = factoriax_step(
            jax.random.key(0),
            damaged,
            jnp.int32(int(Action.REPAIR)),
            params,
        )
        assert int(new_state.ent_health[eidx]) == MACHINE_HEALTH


class TestPickupHealthGate:
    """Pickup is blocked unless the target is at full health."""

    def _placed_state(self, state_factory):
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
        params = EnvParams()
        placed = place_machine(state, params, 0, int(ItemType.MINER))
        eidx = int(placed.tile_entity[0, 1])
        return placed, params, eidx

    def test_pickup_blocked_when_damaged(self, state_factory) -> None:
        """A damaged target stays in place. The inventory is unchanged."""
        state, _, eidx = self._placed_state(state_factory)
        damaged = state.replace(ent_health=state.ent_health.at[eidx].set(50))
        result = pickup_machine(damaged, EnvParams(), 0)
        # Machine still on grid.
        assert int(result.machine_types[0, 1]) == int(Machine.MINER)
        # Entity slot still active.
        assert int(result.ent_y[eidx]) == 0
        # Inventory unchanged.
        assert int(result.player_inventory[0, ItemType.MINER]) == 0
        # Health unchanged.
        assert int(result.ent_health[eidx]) == 50

    def test_pickup_succeeds_at_full_health(self, state_factory) -> None:
        """A full-HP target is picked up as usual. Pickup clears the HP."""
        state, _, eidx = self._placed_state(state_factory)
        result = pickup_machine(state, EnvParams(), 0)
        # Tile cleared.
        assert int(result.machine_types[0, 1]) == int(Machine.NONE)
        # Entity slot deactivated.
        assert int(result.ent_y[eidx]) == -1
        # Inventory got the item back.
        assert int(result.player_inventory[0, ItemType.MINER]) == 1
        # ent_health for the slot cleared.
        assert int(result.ent_health[eidx]) == 0

    def test_pickup_then_replace_resets_to_full_health(self, state_factory) -> None:
        """Pickup-then-replace restores a fresh machine at full HP."""
        state, params, eidx = self._placed_state(state_factory)
        # Damage but not enough to block. Full HP, so pickup works.
        picked_up = pickup_machine(state, params, 0)
        # Now place again. It must land at full HP in some slot.
        replaced = place_machine(picked_up, params, 0, int(ItemType.MINER))
        new_eidx = int(replaced.tile_entity[0, 1])
        assert new_eidx >= 0
        assert int(replaced.ent_health[new_eidx]) == MACHINE_HEALTH
