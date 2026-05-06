"""Tests for the machine placement system."""

import jax
import jax.numpy as jnp
import pytest

from factoriax import BlockType, Direction, ItemType
from factoriax.constants import (
    MAX_HEALTH,
    NUM_ACTIONS,
    NUM_ITEM_TYPES,
    Action,
    MachineType,
)
from factoriax.envs.factoriax_env import FactoriaXEnv
from factoriax.game_logic import factoriax_step
from factoriax.machine_config import DEFAULT_MACHINE_CONFIG, MachineConfigOverride
from factoriax.placement import (
    apply_repair,
    get_tile_in_front,
    is_placeable_item,
    is_valid_placement_tile,
    pickup_machine,
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
        new_state = place_machine(state, EnvParams(), 0, int(ItemType.MINER))

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
        new_state = place_machine(state, EnvParams(), 0, int(ItemType.MINER))

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
        new_state = place_machine(state, EnvParams(), 0, int(ItemType.MINER))

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
        new_state = place_machine(state, EnvParams(), 0, int(ItemType.MINER))

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
        new_state = place_machine(state, EnvParams(), 0, int(ItemType.COAL))

        assert new_state.machine_types[0, 1] == MachineType.NONE
        assert new_state.player_inventory[0, ItemType.COAL] == 5


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
        """A freshly placed machine has ``MAX_HEALTH`` HP under defaults."""
        state = self._miner_state(state_factory)
        params = EnvParams()
        new_state = place_machine(state, params, 0, int(ItemType.MINER))
        eidx = int(new_state.tile_entity[0, 1])
        assert eidx >= 0
        assert int(new_state.ent_health[eidx]) == MAX_HEALTH

    def test_place_initializes_per_type_override(self, state_factory) -> None:
        """An override on MINER.max_health is honored by placement."""
        state = self._miner_state(state_factory)
        params = EnvParams(
            machine_config=DEFAULT_MACHINE_CONFIG.with_overrides(
                {int(MachineType.MINER): MachineConfigOverride(max_health=42)}
            )
        )
        new_state = place_machine(state, params, 0, int(ItemType.MINER))
        eidx = int(new_state.tile_entity[0, 1])
        assert eidx >= 0
        assert int(new_state.ent_health[eidx]) == 42

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
        from factoriax.levels import LevelBuilder, build_state

        level = (
            LevelBuilder(4, 4)
            .place_machine(2, 2, int(MachineType.FURNACE), int(Direction.UP))
            .build("hp_init_test")
        )
        params = EnvParams(map_width=4, map_height=4, num_players=1)
        state = build_state(level, params)
        eidx = int(state.tile_entity[2, 2])
        assert eidx >= 0
        assert int(state.ent_health[eidx]) == MAX_HEALTH


class TestActionRepair:
    """Tests for Action.REPAIR and apply_repair."""

    def test_action_repair_value(self) -> None:
        """REPAIR is the last action; NUM_ACTIONS reflects it."""
        assert int(Action.REPAIR) == 78
        assert NUM_ACTIONS == 79

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
        assert int(repaired.ent_health[eidx]) == MAX_HEALTH

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

    def test_apply_repair_honors_per_type_override(self, state_factory) -> None:
        """If max_health is overridden for a type, apply_repair restores
        to the override value (not MAX_HEALTH)."""
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
        params = EnvParams(
            machine_config=DEFAULT_MACHINE_CONFIG.with_overrides(
                {int(MachineType.MINER): MachineConfigOverride(max_health=42)}
            )
        )
        placed = place_machine(state, params, 0, int(ItemType.MINER))
        eidx = int(placed.tile_entity[0, 1])
        damaged = placed.replace(ent_health=placed.ent_health.at[eidx].set(5))
        repaired = apply_repair(damaged, params, 0)
        assert int(repaired.ent_health[eidx]) == 42

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
        assert int(new_state.ent_health[eidx]) == MAX_HEALTH


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
        """A damaged target stays in place; inventory unchanged."""
        state, _, eidx = self._placed_state(state_factory)
        damaged = state.replace(ent_health=state.ent_health.at[eidx].set(50))
        result = pickup_machine(damaged, EnvParams(), 0)
        # Machine still on grid.
        assert int(result.machine_types[0, 1]) == int(MachineType.MINER)
        # Entity slot still active.
        assert int(result.ent_y[eidx]) == 0
        # Inventory unchanged.
        assert int(result.player_inventory[0, ItemType.MINER]) == 0
        # Health unchanged.
        assert int(result.ent_health[eidx]) == 50

    def test_pickup_succeeds_at_full_health(self, state_factory) -> None:
        """A full-HP target is picked up normally; HP cleared on pickup."""
        state, _, eidx = self._placed_state(state_factory)
        result = pickup_machine(state, EnvParams(), 0)
        # Tile cleared.
        assert int(result.machine_types[0, 1]) == int(MachineType.NONE)
        # Entity slot deactivated.
        assert int(result.ent_y[eidx]) == -1
        # Inventory got the item back.
        assert int(result.player_inventory[0, ItemType.MINER]) == 1
        # ent_health for the slot cleared.
        assert int(result.ent_health[eidx]) == 0

    def test_pickup_then_replace_resets_to_full_health(self, state_factory) -> None:
        """Pickup-then-replace restores a fresh machine at full HP."""
        state, params, eidx = self._placed_state(state_factory)
        # Damage but not enough to block — actually, full HP so pickup works.
        picked_up = pickup_machine(state, params, 0)
        # Now place again — should land at full HP in some slot.
        replaced = place_machine(picked_up, params, 0, int(ItemType.MINER))
        new_eidx = int(replaced.tile_entity[0, 1])
        assert new_eidx >= 0
        assert int(replaced.ent_health[new_eidx]) == MAX_HEALTH


class TestTrajectoryRecordsEntHealth:
    """ent_health survives the state->trajectory->state roundtrip."""

    def test_ent_health_roundtrip_preserves_values(self, state_factory) -> None:
        """A non-trivial ent_health pattern is preserved end-to-end."""
        from factoriax.analysis.trajectory import (
            states_to_trajectory,
            trajectory_to_states,
        )

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
        damaged = placed.replace(ent_health=placed.ent_health.at[eidx].set(33))

        traj = states_to_trajectory([damaged])
        # Field should be present and shaped (B, T, MAX_M).
        assert traj.ent_health is not None
        assert traj.ent_health.shape == (1, 1, placed.ent_health.shape[0])
        # Roundtrip back into a state list.
        restored = trajectory_to_states(traj, episode=0)
        assert len(restored) == 1
        assert int(restored[0].ent_health[eidx]) == 33
