"""Tests for the pouch-based machine inventory system.

Covers machine inventory state fields, constraints per machine type,
and basic machine operations (refueling, mining) using pouch indexing.
"""

from __future__ import annotations

import jax.numpy as jnp

from factoriax import BlockType, EnvParams, ItemType
from factoriax.constants import (
    MACHINE_MAX_TYPES,
    NUM_ITEM_TYPES,
    MachineType,
)
from factoriax.levels import generate_state
from factoriax.machines import refuel_machines, run_miners, update_all_machines


def _machine_inv(
    h: int, w: int, y: int, x: int, **items: int,
) -> jnp.ndarray:
    """Build a machine inventory with items at one tile."""
    inv = jnp.zeros((h, w, NUM_ITEM_TYPES), dtype=jnp.int16)
    name_to_type = {m.name: int(m) for m in ItemType}
    for name, count in items.items():
        inv = inv.at[y, x, name_to_type[name]].set(count)
    return inv


class TestMachineInventoryStateFields:
    """Verify machine inventory state fields have correct properties."""

    def test_inventory_dtype(self) -> None:
        """machine_inventory should be int16."""
        import jax

        params = EnvParams(map_width=4, map_height=4, num_players=1)
        state = generate_state(jax.random.PRNGKey(0), params)
        assert state.machine_inventory.dtype == jnp.int16

    def test_inventory_shape(self) -> None:
        """machine_inventory shape should be (H, W, NUM_ITEM_TYPES)."""
        import jax

        params = EnvParams(map_width=4, map_height=4, num_players=1)
        state = generate_state(jax.random.PRNGKey(0), params)
        assert state.machine_inventory.shape == (4, 4, NUM_ITEM_TYPES)


class TestGenerateStateInventoryFields:
    """Verify generated state has correct inventory initialization."""

    def test_shape(self) -> None:
        """Generated state should have correct inventory shape."""
        import jax

        params = EnvParams(map_width=8, map_height=6, num_players=1)
        state = generate_state(jax.random.PRNGKey(0), params)
        assert state.machine_inventory.shape == (
            6, 8, NUM_ITEM_TYPES,
        )

    def test_zero_initialized(self) -> None:
        """Generated state should have zero-initialized inventories."""
        import jax

        params = EnvParams(map_width=4, map_height=4, num_players=1)
        state = generate_state(jax.random.PRNGKey(0), params)
        assert jnp.all(state.machine_inventory == 0)


class TestMinerInventory:
    """Tests for miner inventory operations using pouches."""

    def test_refuel_reads_coal(self, state_factory) -> None:
        """Refueling should consume coal from the miner's pouch."""
        state = state_factory(
            world_map=jnp.array(
                [[BlockType.IRON]], dtype=jnp.int32,
            ),
            machine_types=jnp.array(
                [[MachineType.MINER]], dtype=jnp.int32,
            ),
            machine_inventory=_machine_inv(1, 1, 0, 0, COAL=5),
            machine_power=jnp.array([[0]], dtype=jnp.int32),
        )
        new = refuel_machines(state)
        assert int(new.machine_inventory[0, 0, ItemType.COAL]) == 4
        assert int(new.machine_power[0, 0]) == 10

    def test_run_miners_deposits_ore(self, state_factory) -> None:
        """Miners should deposit ore into their pouch."""
        state = state_factory(
            world_map=jnp.array(
                [[BlockType.IRON]], dtype=jnp.int32,
            ),
            block_resources=jnp.array([[50]], dtype=jnp.int16),
            machine_types=jnp.array(
                [[MachineType.MINER]], dtype=jnp.int32,
            ),
            machine_power=jnp.array([[5]], dtype=jnp.int32),
        )
        new = run_miners(state)
        assert int(new.machine_inventory[0, 0, ItemType.IRON_ORE]) > 0

    def test_output_full_blocks_mining(self, state_factory) -> None:
        """Miner should not mine when ore pouch is at max stack."""
        from factoriax.constants import MAX_MACHINE_STACK_SIZE

        state = state_factory(
            world_map=jnp.array(
                [[BlockType.IRON]], dtype=jnp.int32,
            ),
            block_resources=jnp.array([[50]], dtype=jnp.int16),
            machine_types=jnp.array(
                [[MachineType.MINER]], dtype=jnp.int32,
            ),
            machine_power=jnp.array([[5]], dtype=jnp.int32),
            machine_inventory=_machine_inv(
                1, 1, 0, 0, IRON=MAX_MACHINE_STACK_SIZE,
            ),
        )
        new = run_miners(state)
        # No change — already at capacity.
        assert int(new.machine_inventory[0, 0, ItemType.IRON_ORE]) == (
            MAX_MACHINE_STACK_SIZE
        )
        assert int(new.block_resources[0, 0]) == 50


class TestPalletInventory:
    """Tests for pallet inventory."""

    def test_pallet_initialized_empty(self, state_factory) -> None:
        """Pallet should start with empty inventory."""
        state = state_factory(
            world_map=jnp.array(
                [[BlockType.DIRT]], dtype=jnp.int32,
            ),
            machine_types=jnp.array(
                [[MachineType.PALLET]], dtype=jnp.int32,
            ),
        )
        assert jnp.all(state.machine_inventory[0, 0] == 0)

    def test_pallet_unaffected_by_update(self, state_factory) -> None:
        """update_all_machines should not modify pallet contents."""
        state = state_factory(
            world_map=jnp.array(
                [[BlockType.DIRT]], dtype=jnp.int32,
            ),
            machine_types=jnp.array(
                [[MachineType.PALLET]], dtype=jnp.int32,
            ),
            machine_inventory=_machine_inv(1, 1, 0, 0, IRON=10),
        )
        params = EnvParams(map_width=1, map_height=1, num_players=1)
        new = update_all_machines(state, params)
        assert int(new.machine_inventory[0, 0, ItemType.IRON_ORE]) == 10


class TestAssemblerInventory:
    """Tests for assembler inventory."""

    def test_assembler_initialized_empty(
        self, state_factory,
    ) -> None:
        """Assembler should start with empty inventory."""
        state = state_factory(
            world_map=jnp.array(
                [[BlockType.DIRT]], dtype=jnp.int32,
            ),
            machine_types=jnp.array(
                [[MachineType.ASSEMBLER]], dtype=jnp.int32,
            ),
        )
        assert jnp.all(state.machine_inventory[0, 0] == 0)

    def test_assembler_idle_without_inputs(
        self, state_factory,
    ) -> None:
        """Assembler without inputs should remain idle."""
        state = state_factory(
            world_map=jnp.array(
                [[BlockType.DIRT]], dtype=jnp.int32,
            ),
            machine_types=jnp.array(
                [[MachineType.ASSEMBLER]], dtype=jnp.int32,
            ),
        )
        params = EnvParams(map_width=1, map_height=1, num_players=1)
        new = update_all_machines(state, params)
        assert jnp.all(new.machine_inventory[0, 0] == 0)


class TestMaxTypesConstraint:
    """Tests for the MACHINE_MAX_TYPES constraint."""

    def test_belt_max_types_is_one(self) -> None:
        """Belt should hold at most 1 distinct item type."""
        assert int(MACHINE_MAX_TYPES[MachineType.CONVEYOR_BELT]) == 1

    def test_pallet_max_types_is_one(self) -> None:
        """Pallet should hold at most 1 distinct item type."""
        assert int(MACHINE_MAX_TYPES[MachineType.PALLET]) == 1

    def test_assembler_max_types_is_four(self) -> None:
        """Assembler should hold at most 4 distinct item types."""
        assert int(MACHINE_MAX_TYPES[MachineType.ASSEMBLER]) == 4
