"""Tests for machine health, disabled machines, and the REPAIR action."""

import jax.numpy as jnp
from jax import random

from factoriax.constants import (
    DEFAULT_MACHINE_MAX_HEALTH,
    MACHINE_INVENTORY_COUNT_DTYPE,
    NUM_ITEM_TYPES,
    Action,
    BlockType,
    Direction,
    ItemType,
    MachineType,
)
from factoriax.game_logic import factoriax_step, repair_machine
from factoriax.machines import run_assemblers, run_conveyor_belts, run_miners
from factoriax.state import EnvParams


def _machine_inv(h: int, w: int, **tile_items: dict) -> jnp.ndarray:
    """Build a machine_inventory pouch array.

    Args:
        h: Map height.
        w: Map width.
        **tile_items: Mapping of ``"y_x_ItemName"`` to count.

    Returns:
        Machine inventory array of shape ``(h, w, NUM_ITEM_TYPES)``.
    """
    inv = jnp.zeros(
        (h, w, NUM_ITEM_TYPES), dtype=MACHINE_INVENTORY_COUNT_DTYPE,
    )
    name_to_type = {m.name: int(m) for m in ItemType}
    for key, count in tile_items.items():
        parts = key.split("_")
        y = int(parts[0].lstrip("y"))
        x = int(parts[1].lstrip("x"))
        item_name = "_".join(parts[2:])
        inv = inv.at[y, x, name_to_type[item_name]].set(count)
    return inv


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
        md = jnp.array(
            [[Direction.RIGHT, Direction.RIGHT]], dtype=jnp.int32,
        )
        inv = _machine_inv(1, 2, y0_x0_IRON=5)
        state = state_factory(
            world_map=jnp.zeros((1, 2), dtype=jnp.int32),
            machine_types=mt,
            machine_direction=md,
            machine_inventory=inv,
            machine_health=jnp.array(
                [[0, DEFAULT_MACHINE_MAX_HEALTH]], dtype=jnp.int32,
            ),
        )
        new = run_conveyor_belts(state)
        # Source belt disabled, items should not move.
        assert int(new.machine_inventory[0, 0, ItemType.IRON]) == 5

    def test_assembler_disabled_at_zero_health(self, state_factory) -> None:
        """An assembler at 0 HP should not start crafting."""
        from factoriax.constants import NUM_TECHNOLOGIES

        state = state_factory(
            world_map=jnp.zeros((1, 1), dtype=jnp.int32),
            machine_types=jnp.array(
                [[MachineType.ASSEMBLER]], dtype=jnp.int32,
            ),
            machine_health=jnp.array([[0]], dtype=jnp.int32),
            research_unlocked=jnp.ones(
                NUM_TECHNOLOGIES, dtype=jnp.bool_,
            ),
            machine_inventory=_machine_inv(1, 1, y0_x0_IRON=10),
        )
        new = run_assemblers(state)
        assert int(new.machine_power[0, 0]) == 0


class TestRepairAction:
    """The REPAIR action should consume materials and restore health."""

    def test_repair_restores_full_health(self, state_factory) -> None:
        """Repairing a damaged miner consumes 5 copper + 5 iron, restores HP."""
        inv = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int32)
        inv = inv.at[0, ItemType.COPPER].set(10)
        inv = inv.at[0, ItemType.IRON].set(10)

        state = state_factory(
            world_map=jnp.zeros((1, 3), dtype=jnp.int32),
            player_position=(0, 0),
            player_direction=int(Direction.RIGHT),
            player_inventory=inv,
            machine_types=jnp.array(
                [[MachineType.NONE, MachineType.MINER, MachineType.NONE]],
                dtype=jnp.int32,
            ),
            machine_health=jnp.array([[0, 10, 0]], dtype=jnp.int32),
        )
        new = repair_machine(state, 0)
        assert int(new.machine_health[0, 1]) == DEFAULT_MACHINE_MAX_HEALTH
        assert int(new.player_inventory[0, ItemType.COPPER]) == 5
        assert int(new.player_inventory[0, ItemType.IRON]) == 5

    def test_repair_noop_without_materials(self, state_factory) -> None:
        """Repair should fail if player lacks required materials."""
        inv = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int32)
        inv = inv.at[0, ItemType.IRON].set(2)  # Not enough (need 5)

        state = state_factory(
            world_map=jnp.zeros((1, 3), dtype=jnp.int32),
            player_position=(0, 0),
            player_direction=int(Direction.RIGHT),
            player_inventory=inv,
            machine_types=jnp.array(
                [[MachineType.NONE, MachineType.MINER, MachineType.NONE]],
                dtype=jnp.int32,
            ),
            machine_health=jnp.array([[0, 10, 0]], dtype=jnp.int32),
        )
        new = repair_machine(state, 0)
        assert int(new.machine_health[0, 1]) == 10  # Unchanged
        assert int(new.player_inventory[0, ItemType.IRON]) == 2

    def test_repair_noop_at_full_health(self, state_factory) -> None:
        """Repair should be a no-op if machine is already at full health."""
        inv = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int32)
        inv = inv.at[0, ItemType.COPPER].set(10)
        inv = inv.at[0, ItemType.IRON].set(10)

        state = state_factory(
            world_map=jnp.zeros((1, 3), dtype=jnp.int32),
            player_position=(0, 0),
            player_direction=int(Direction.RIGHT),
            player_inventory=inv,
            machine_types=jnp.array(
                [[MachineType.NONE, MachineType.MINER, MachineType.NONE]],
                dtype=jnp.int32,
            ),
            machine_health=jnp.array(
                [[0, DEFAULT_MACHINE_MAX_HEALTH, 0]], dtype=jnp.int32
            ),
        )
        new = repair_machine(state, 0)
        assert int(new.player_inventory[0, ItemType.COPPER]) == 10
        assert int(new.player_inventory[0, ItemType.IRON]) == 10

    def test_repair_noop_no_machine(self, state_factory) -> None:
        """Repair should be a no-op when facing an empty tile."""
        inv = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int32)
        inv = inv.at[0, ItemType.IRON].set(10)

        state = state_factory(
            world_map=jnp.zeros((1, 3), dtype=jnp.int32),
            player_position=(0, 0),
            player_direction=int(Direction.RIGHT),
            player_inventory=inv,
        )
        new = repair_machine(state, 0)
        assert int(new.player_inventory[0, ItemType.IRON]) == 10

    def test_repair_via_step(self, state_factory) -> None:
        """REPAIR action through factoriax_step should work."""
        inv = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int32)
        inv = inv.at[0, ItemType.COPPER].set(10)
        inv = inv.at[0, ItemType.IRON].set(10)

        state = state_factory(
            world_map=jnp.zeros((1, 3), dtype=jnp.int32),
            player_position=(0, 0),
            player_direction=int(Direction.RIGHT),
            player_inventory=inv,
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

    def test_repair_pallet_costs_iron(self, state_factory) -> None:
        """Repairing a pallet should consume 5 iron (pallet recipe cost)."""
        inv = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int32)
        inv = inv.at[0, ItemType.IRON].set(10)

        state = state_factory(
            world_map=jnp.zeros((1, 3), dtype=jnp.int32),
            player_position=(0, 0),
            player_direction=int(Direction.RIGHT),
            player_inventory=inv,
            machine_types=jnp.array(
                [[MachineType.NONE, MachineType.PALLET, MachineType.NONE]],
                dtype=jnp.int32,
            ),
            machine_health=jnp.array([[0, 50, 0]], dtype=jnp.int32),
        )
        new = repair_machine(state, 0)
        assert int(new.machine_health[0, 1]) == DEFAULT_MACHINE_MAX_HEALTH
        assert int(new.player_inventory[0, ItemType.IRON]) == 5  # 10 - 5
