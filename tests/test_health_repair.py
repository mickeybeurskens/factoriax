"""Tests for machine health, disabled machines, and the REPAIR action."""

import jax.numpy as jnp
from jax import random

from factoriax.constants import (
    DEFAULT_MACHINE_MAX_HEALTH,
    Action,
    BlockType,
    Direction,
    ItemType,
    MachineType,
)
from factoriax.game_logic import factoriax_step, repair_machine
from factoriax.machines import run_assemblers, run_conveyor_belts, run_miners
from factoriax.state import EnvParams


class TestDisabledMachines:
    """Machines with health == 0 should not operate."""

    def test_miner_disabled_at_zero_health(self, state_factory) -> None:
        """A miner at 0 HP should not extract ore even with power."""
        state = state_factory(
            world_map=jnp.array([[BlockType.IRON]], dtype=jnp.int32),
            machine_types=jnp.array([[MachineType.MINER]], dtype=jnp.int32),
            machine_health=jnp.array([[0]], dtype=jnp.int32),
            machine_power=jnp.array([[5]], dtype=jnp.int32),
            block_resources=jnp.array([[100]], dtype=jnp.int16),
        )
        new = run_miners(state)
        assert int(new.block_resources[0, 0]) == 100

    def test_miner_works_with_health(self, state_factory) -> None:
        """A miner with health > 0 should extract ore normally."""
        state = state_factory(
            world_map=jnp.array([[BlockType.IRON]], dtype=jnp.int32),
            machine_types=jnp.array([[MachineType.MINER]], dtype=jnp.int32),
            machine_health=jnp.array(
                [[DEFAULT_MACHINE_MAX_HEALTH]], dtype=jnp.int32
            ),
            machine_power=jnp.array([[5]], dtype=jnp.int32),
            block_resources=jnp.array([[100]], dtype=jnp.int16),
        )
        new = run_miners(state)
        assert int(new.block_resources[0, 0]) < 100

    def test_belt_disabled_at_zero_health(self, state_factory) -> None:
        """A belt at 0 HP should not push items."""
        mt = jnp.array(
            [[MachineType.CONVEYOR_BELT, MachineType.CONVEYOR_BELT]],
            dtype=jnp.int32,
        )
        md = jnp.array([[Direction.RIGHT, Direction.RIGHT]], dtype=jnp.int32)
        inv = jnp.zeros((1, 2, 8), dtype=jnp.int32)
        inv = inv.at[0, 0, 0].set(int(ItemType.IRON))
        cnt = jnp.zeros((1, 2, 8), dtype=jnp.int16)
        cnt = cnt.at[0, 0, 0].set(jnp.int16(5))
        state = state_factory(
            world_map=jnp.zeros((1, 2), dtype=jnp.int32),
            machine_types=mt,
            machine_direction=md,
            machine_inventory_items=inv,
            machine_inventory_counts=cnt,
            machine_health=jnp.array([[0, DEFAULT_MACHINE_MAX_HEALTH]], dtype=jnp.int32),
        )
        new = run_conveyor_belts(state)
        # Source belt disabled, items should not move.
        assert int(new.machine_inventory_counts[0, 0, 0]) == 5

    def test_assembler_disabled_at_zero_health(self, state_factory) -> None:
        """An assembler at 0 HP should not start crafting."""
        from factoriax.constants import NUM_TECHNOLOGIES

        state = state_factory(
            world_map=jnp.zeros((1, 1), dtype=jnp.int32),
            machine_types=jnp.array([[MachineType.ASSEMBLER]], dtype=jnp.int32),
            machine_health=jnp.array([[0]], dtype=jnp.int32),
            research_unlocked=jnp.ones(NUM_TECHNOLOGIES, dtype=jnp.bool_),
        )
        inv = state.machine_inventory_items.at[0, 0, 0].set(int(ItemType.IRON))
        cnt = state.machine_inventory_counts.at[0, 0, 0].set(jnp.int16(10))
        state = state.replace(
            machine_inventory_items=inv,
            machine_inventory_counts=cnt,
        )
        new = run_assemblers(state)
        assert int(new.machine_power[0, 0]) == 0


class TestRepairAction:
    """The REPAIR action should consume materials and restore health."""

    def test_repair_restores_full_health(self, state_factory) -> None:
        """Repairing a damaged miner consumes 5 copper + 5 iron, restores HP."""
        # Miner recipe: 5 copper, 5 iron. Place miner at (1, 0), player at (0, 0) facing right.
        inv_items = jnp.zeros((1, 10), dtype=jnp.int32)
        inv_items = inv_items.at[0, 0].set(int(ItemType.COPPER))
        inv_items = inv_items.at[0, 1].set(int(ItemType.IRON))
        inv_counts = jnp.zeros((1, 10), dtype=jnp.int32)
        inv_counts = inv_counts.at[0, 0].set(10)
        inv_counts = inv_counts.at[0, 1].set(10)

        state = state_factory(
            world_map=jnp.zeros((1, 3), dtype=jnp.int32),
            player_position=(0, 0),
            player_direction=int(Direction.RIGHT),
            inventory_items=inv_items,
            inventory_counts=inv_counts,
            machine_types=jnp.array(
                [[MachineType.NONE, MachineType.MINER, MachineType.NONE]],
                dtype=jnp.int32,
            ),
            machine_health=jnp.array([[0, 10, 0]], dtype=jnp.int32),
        )
        new = repair_machine(state, 0)
        assert int(new.machine_health[0, 1]) == DEFAULT_MACHINE_MAX_HEALTH
        assert int(new.inventory_counts[0, 0]) == 5  # 10 - 5 copper
        assert int(new.inventory_counts[0, 1]) == 5  # 10 - 5 iron

    def test_repair_noop_without_materials(self, state_factory) -> None:
        """Repair should fail if player lacks required materials."""
        inv_items = jnp.zeros((1, 10), dtype=jnp.int32)
        inv_items = inv_items.at[0, 0].set(int(ItemType.IRON))
        inv_counts = jnp.zeros((1, 10), dtype=jnp.int32)
        inv_counts = inv_counts.at[0, 0].set(2)  # Not enough iron (need 5)

        state = state_factory(
            world_map=jnp.zeros((1, 3), dtype=jnp.int32),
            player_position=(0, 0),
            player_direction=int(Direction.RIGHT),
            inventory_items=inv_items,
            inventory_counts=inv_counts,
            machine_types=jnp.array(
                [[MachineType.NONE, MachineType.MINER, MachineType.NONE]],
                dtype=jnp.int32,
            ),
            machine_health=jnp.array([[0, 10, 0]], dtype=jnp.int32),
        )
        new = repair_machine(state, 0)
        assert int(new.machine_health[0, 1]) == 10  # Unchanged
        assert int(new.inventory_counts[0, 0]) == 2  # Unchanged

    def test_repair_noop_at_full_health(self, state_factory) -> None:
        """Repair should be a no-op if machine is already at full health."""
        inv_items = jnp.zeros((1, 10), dtype=jnp.int32)
        inv_items = inv_items.at[0, 0].set(int(ItemType.COPPER))
        inv_items = inv_items.at[0, 1].set(int(ItemType.IRON))
        inv_counts = jnp.zeros((1, 10), dtype=jnp.int32)
        inv_counts = inv_counts.at[0, 0].set(10)
        inv_counts = inv_counts.at[0, 1].set(10)

        state = state_factory(
            world_map=jnp.zeros((1, 3), dtype=jnp.int32),
            player_position=(0, 0),
            player_direction=int(Direction.RIGHT),
            inventory_items=inv_items,
            inventory_counts=inv_counts,
            machine_types=jnp.array(
                [[MachineType.NONE, MachineType.MINER, MachineType.NONE]],
                dtype=jnp.int32,
            ),
            machine_health=jnp.array(
                [[0, DEFAULT_MACHINE_MAX_HEALTH, 0]], dtype=jnp.int32
            ),
        )
        new = repair_machine(state, 0)
        assert int(new.inventory_counts[0, 0]) == 10  # Not consumed
        assert int(new.inventory_counts[0, 1]) == 10

    def test_repair_noop_no_machine(self, state_factory) -> None:
        """Repair should be a no-op when facing an empty tile."""
        inv_items = jnp.zeros((1, 10), dtype=jnp.int32)
        inv_items = inv_items.at[0, 0].set(int(ItemType.IRON))
        inv_counts = jnp.zeros((1, 10), dtype=jnp.int32)
        inv_counts = inv_counts.at[0, 0].set(10)

        state = state_factory(
            world_map=jnp.zeros((1, 3), dtype=jnp.int32),
            player_position=(0, 0),
            player_direction=int(Direction.RIGHT),
            inventory_items=inv_items,
            inventory_counts=inv_counts,
        )
        new = repair_machine(state, 0)
        assert int(new.inventory_counts[0, 0]) == 10

    def test_repair_via_step(self, state_factory) -> None:
        """REPAIR action through factoriax_step should work."""
        inv_items = jnp.zeros((1, 10), dtype=jnp.int32)
        inv_items = inv_items.at[0, 0].set(int(ItemType.COPPER))
        inv_items = inv_items.at[0, 1].set(int(ItemType.IRON))
        inv_counts = jnp.zeros((1, 10), dtype=jnp.int32)
        inv_counts = inv_counts.at[0, 0].set(10)
        inv_counts = inv_counts.at[0, 1].set(10)

        state = state_factory(
            world_map=jnp.zeros((1, 3), dtype=jnp.int32),
            player_position=(0, 0),
            player_direction=int(Direction.RIGHT),
            inventory_items=inv_items,
            inventory_counts=inv_counts,
            machine_types=jnp.array(
                [[MachineType.NONE, MachineType.MINER, MachineType.NONE]],
                dtype=jnp.int32,
            ),
            machine_health=jnp.array([[0, 1, 0]], dtype=jnp.int32),
        )
        params = EnvParams(map_width=3, map_height=1, num_players=1)
        rng = random.PRNGKey(0)
        new = factoriax_step(rng, state, int(Action.REPAIR), params)
        assert int(new.machine_health[0, 1]) == DEFAULT_MACHINE_MAX_HEALTH

    def test_repair_chest_costs_iron(self, state_factory) -> None:
        """Repairing a chest should consume 5 iron (chest recipe cost)."""
        inv_items = jnp.zeros((1, 10), dtype=jnp.int32)
        inv_items = inv_items.at[0, 0].set(int(ItemType.IRON))
        inv_counts = jnp.zeros((1, 10), dtype=jnp.int32)
        inv_counts = inv_counts.at[0, 0].set(10)

        state = state_factory(
            world_map=jnp.zeros((1, 3), dtype=jnp.int32),
            player_position=(0, 0),
            player_direction=int(Direction.RIGHT),
            inventory_items=inv_items,
            inventory_counts=inv_counts,
            machine_types=jnp.array(
                [[MachineType.NONE, MachineType.CHEST, MachineType.NONE]],
                dtype=jnp.int32,
            ),
            machine_health=jnp.array([[0, 50, 0]], dtype=jnp.int32),
        )
        new = repair_machine(state, 0)
        assert int(new.machine_health[0, 1]) == DEFAULT_MACHINE_MAX_HEALTH
        assert int(new.inventory_counts[0, 0]) == 5  # 10 - 5
