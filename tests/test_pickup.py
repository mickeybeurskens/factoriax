"""Tests for machine pickup (pouch inventory model)."""

import jax.numpy as jnp

from factoriax import BlockType, Direction, ItemType
from factoriax.constants import NUM_ITEM_TYPES, MachineType
from factoriax.placement import can_fit_in_player, pickup_machine


class TestCanFitInPlayer:
    """Tests for pouch-based fit checking."""

    def test_empty_inventory_fits_single_item(self) -> None:
        """Empty player inventory should accept any single item."""
        player = jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.int32)
        to_add = jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.int32)
        to_add = to_add.at[ItemType.IRON].set(5)
        assert bool(can_fit_in_player(player, to_add))

    def test_full_inventory_rejects(self) -> None:
        """Full inventory should reject additional items."""
        from factoriax.constants import PLAYER_MAX_STACK

        player = PLAYER_MAX_STACK.astype(jnp.int32)
        to_add = jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.int32)
        to_add = to_add.at[ItemType.IRON].set(1)
        assert not bool(can_fit_in_player(player, to_add))


class TestPickupMachine:
    """Tests for pickup_machine with pouch inventory."""

    def test_basic_pickup(self, state_factory) -> None:
        """Picking up a machine adds the machine item to inventory."""
        state = state_factory(
            world_map=jnp.full(
                (3, 3), BlockType.DIRT, dtype=jnp.int32,
            ),
            player_position=(1, 0),
            player_direction=int(Direction.DOWN),
            machine_types=jnp.full(
                (3, 3), MachineType.NONE, dtype=jnp.int32,
            ).at[1, 1].set(MachineType.CHEST),
        )
        new = pickup_machine(state, 0)
        assert int(new.player_inventory[0, ItemType.CHEST]) == 1
        assert int(new.machine_types[1, 1]) == MachineType.NONE

    def test_pickup_transfers_contents(self, state_factory) -> None:
        """Machine inventory contents should transfer to player."""
        m_inv = jnp.zeros(
            (3, 3, NUM_ITEM_TYPES), dtype=jnp.int16,
        )
        m_inv = m_inv.at[1, 1, ItemType.IRON].set(10)
        state = state_factory(
            world_map=jnp.full(
                (3, 3), BlockType.DIRT, dtype=jnp.int32,
            ),
            player_position=(1, 0),
            player_direction=int(Direction.DOWN),
            machine_types=jnp.full(
                (3, 3), MachineType.NONE, dtype=jnp.int32,
            ).at[1, 1].set(MachineType.CHEST),
            machine_inventory=m_inv,
        )
        new = pickup_machine(state, 0)
        assert int(new.player_inventory[0, ItemType.IRON]) == 10
        assert int(new.player_inventory[0, ItemType.CHEST]) == 1

    def test_pickup_empty_tile_noop(self, state_factory) -> None:
        """Picking up from an empty tile should be a no-op."""
        state = state_factory(
            world_map=jnp.full(
                (3, 3), BlockType.DIRT, dtype=jnp.int32,
            ),
            player_position=(1, 0),
            player_direction=int(Direction.DOWN),
        )
        new = pickup_machine(state, 0)
        assert jnp.array_equal(
            new.player_inventory, state.player_inventory,
        )

    def test_pickup_clears_machine_state(self, state_factory) -> None:
        """Pickup should zero out all machine state for the tile."""
        m_inv = jnp.zeros(
            (3, 3, NUM_ITEM_TYPES), dtype=jnp.int16,
        )
        m_inv = m_inv.at[1, 1, ItemType.COAL].set(5)
        state = state_factory(
            world_map=jnp.full(
                (3, 3), BlockType.DIRT, dtype=jnp.int32,
            ),
            player_position=(1, 0),
            player_direction=int(Direction.DOWN),
            machine_types=jnp.full(
                (3, 3), MachineType.NONE, dtype=jnp.int32,
            ).at[1, 1].set(MachineType.MINER),
            machine_inventory=m_inv,
            machine_power=jnp.zeros(
                (3, 3), dtype=jnp.int32,
            ).at[1, 1].set(5),
        )
        new = pickup_machine(state, 0)
        assert int(new.machine_types[1, 1]) == MachineType.NONE
        assert int(new.machine_health[1, 1]) == 0
        assert int(new.machine_power[1, 1]) == 0
        assert int(new.machine_inventory[1, 1].sum()) == 0
