"""Tests for the machine system."""

import jax.numpy as jnp
from jax import random

from factoriax import BlockType, EnvParams, ItemType
from factoriax.constants import (
    MACHINE_INVENTORY_COUNT_DTYPE,
    MAX_MACHINE_STACK_SIZE,
    NUM_ITEM_TYPES,
    POWER_PER_COAL,
    MachineType,
)
from factoriax.machines import refuel_machines, run_miners, update_all_machines
from factoriax.world_gen import generate_world


def _machine_inv(h: int, w: int, **tile_items: dict) -> jnp.ndarray:
    """Build a machine_inventory pouch array.

    Args:
        h: Map height.
        w: Map width.
        **tile_items: Mapping of ``"y_x_ItemName"`` to count, e.g.
            ``y0_x0_COAL=5`` sets ``inv[0, 0, ItemType.COAL] = 5``.

    Returns:
        Machine inventory array of shape ``(h, w, NUM_ITEM_TYPES)``.
    """
    inv = jnp.zeros((h, w, NUM_ITEM_TYPES), dtype=MACHINE_INVENTORY_COUNT_DTYPE)
    name_to_type = {m.name: int(m) for m in ItemType}
    for key, count in tile_items.items():
        parts = key.split("_")
        y = int(parts[0].lstrip("y"))
        x = int(parts[1].lstrip("x"))
        item_name = "_".join(parts[2:])
        inv = inv.at[y, x, name_to_type[item_name]].set(count)
    return inv


class TestMachineInitialization:
    """Tests for machine state initialization."""

    def test_world_gen_initializes_no_machines(self) -> None:
        """Generated world should have no machines by default."""
        rng = random.PRNGKey(0)
        params = EnvParams()
        state = generate_world(rng, params)

        assert jnp.all(state.machine_types == MachineType.NONE)
        assert jnp.all(state.machine_power == 0)
        assert jnp.all(state.machine_inventory == 0)

    def test_machine_arrays_match_map_shape(self) -> None:
        """Machine state arrays should have the expected shapes."""
        rng = random.PRNGKey(0)
        params = EnvParams(map_width=16, map_height=24)
        state = generate_world(rng, params)

        assert state.machine_types.shape == state.map.shape
        assert state.machine_power.shape == state.map.shape
        assert state.machine_inventory.shape == (24, 16, NUM_ITEM_TYPES)


class TestMachineRefueling:
    """Tests for machine refueling logic."""

    def test_machine_consumes_coal_when_no_power(self, state_factory) -> None:
        """Machine with no power and coal should consume coal and gain power."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[10]], dtype=jnp.int16),
            machine_types=jnp.array([[MachineType.MINER]], dtype=jnp.int32),
            machine_power=jnp.array([[0]], dtype=jnp.int32),
            machine_inventory=_machine_inv(1, 1, y0_x0_COAL=5),
        )

        new_state = refuel_machines(state)

        assert new_state.machine_inventory[0, 0, ItemType.COAL] == 4
        assert new_state.machine_power[0, 0] == POWER_PER_COAL

    def test_machine_does_not_refuel_with_power(self, state_factory) -> None:
        """Machine with power should not consume coal."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[10]], dtype=jnp.int16),
            machine_types=jnp.array([[MachineType.MINER]], dtype=jnp.int32),
            machine_power=jnp.array([[5]], dtype=jnp.int32),
            machine_inventory=_machine_inv(1, 1, y0_x0_COAL=5),
        )

        new_state = refuel_machines(state)

        assert new_state.machine_inventory[0, 0, ItemType.COAL] == 5
        assert new_state.machine_power[0, 0] == 5

    def test_machine_does_not_refuel_without_coal(self, state_factory) -> None:
        """Machine without coal cannot refuel."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[10]], dtype=jnp.int16),
            machine_types=jnp.array([[MachineType.MINER]], dtype=jnp.int32),
            machine_power=jnp.array([[0]], dtype=jnp.int32),
        )

        new_state = refuel_machines(state)

        assert new_state.machine_inventory[0, 0, ItemType.COAL] == 0
        assert new_state.machine_power[0, 0] == 0

    def test_one_coal_gives_10_power(self) -> None:
        """One coal should provide exactly POWER_PER_COAL (10) power."""
        assert POWER_PER_COAL == 10


class TestMinerOperation:
    """Tests for miner machine behavior."""

    def test_miner_extracts_resources(self, state_factory) -> None:
        """Miner with power should extract resources from block below."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[50]], dtype=jnp.int16),
            machine_types=jnp.array([[MachineType.MINER]], dtype=jnp.int32),
            machine_power=jnp.array([[10]], dtype=jnp.int32),
        )

        new_state = run_miners(state)

        assert new_state.block_resources[0, 0] == 47  # 50 - 3 (mining rate)
        assert new_state.machine_inventory[0, 0, ItemType.COAL] == 3

    def test_miner_consumes_power_when_mining(self, state_factory) -> None:
        """Miner should consume 1 power per step when mining."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[50]], dtype=jnp.int16),
            machine_types=jnp.array([[MachineType.MINER]], dtype=jnp.int32),
            machine_power=jnp.array([[10]], dtype=jnp.int32),
        )

        new_state = run_miners(state)

        assert new_state.machine_power[0, 0] == 9

    def test_miner_does_not_consume_power_when_idle(self, state_factory) -> None:
        """Miner should not consume power when it cannot mine."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            machine_types=jnp.array([[MachineType.MINER]], dtype=jnp.int32),
            machine_power=jnp.array([[10]], dtype=jnp.int32),
        )

        new_state = run_miners(state)

        assert new_state.machine_power[0, 0] == 10

    def test_miner_stops_when_no_power(self, state_factory) -> None:
        """Miner without power should not extract resources."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[50]], dtype=jnp.int16),
            machine_types=jnp.array([[MachineType.MINER]], dtype=jnp.int32),
            machine_power=jnp.array([[0]], dtype=jnp.int32),
        )

        new_state = run_miners(state)

        assert new_state.block_resources[0, 0] == 50
        assert new_state.machine_inventory[0, 0, ItemType.COAL] == 0

    def test_miner_stops_when_output_full(self, state_factory) -> None:
        """Miner should stop when output type is at max stack."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[50]], dtype=jnp.int16),
            machine_types=jnp.array([[MachineType.MINER]], dtype=jnp.int32),
            machine_power=jnp.array([[10]], dtype=jnp.int32),
            machine_inventory=_machine_inv(
                1, 1, y0_x0_COAL=MAX_MACHINE_STACK_SIZE,
            ),
        )

        new_state = run_miners(state)

        assert new_state.block_resources[0, 0] == 50
        assert (
            new_state.machine_inventory[0, 0, ItemType.COAL]
            == MAX_MACHINE_STACK_SIZE
        )
        assert new_state.machine_power[0, 0] == 10  # No power consumed

    def test_miner_stops_when_no_resources(self, state_factory) -> None:
        """Miner should stop when block has no resources."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[0]], dtype=jnp.int16),
            machine_types=jnp.array([[MachineType.MINER]], dtype=jnp.int32),
            machine_power=jnp.array([[10]], dtype=jnp.int32),
        )

        new_state = run_miners(state)

        assert new_state.machine_inventory[0, 0, ItemType.COAL] == 0
        assert new_state.machine_power[0, 0] == 10

    def test_miner_depletes_block_to_dirt(self, state_factory) -> None:
        """Block should become dirt when fully depleted by miner."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[2]], dtype=jnp.int16),
            machine_types=jnp.array([[MachineType.MINER]], dtype=jnp.int32),
            machine_power=jnp.array([[10]], dtype=jnp.int32),
        )

        new_state = run_miners(state)

        assert new_state.map[0, 0] == BlockType.DIRT
        assert new_state.block_resources[0, 0] == 0
        assert new_state.machine_inventory[0, 0, ItemType.COAL] == 2

    def test_miner_mines_partial_when_limited_by_resources(
        self, state_factory,
    ) -> None:
        """Miner should extract only available resources when less than rate."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[1]], dtype=jnp.int16),
            machine_types=jnp.array([[MachineType.MINER]], dtype=jnp.int32),
            machine_power=jnp.array([[10]], dtype=jnp.int32),
        )

        new_state = run_miners(state)

        assert new_state.block_resources[0, 0] == 0
        assert new_state.machine_inventory[0, 0, ItemType.COAL] == 1

    def test_miner_mines_partial_when_limited_by_space(
        self, state_factory,
    ) -> None:
        """Miner should extract only what fits in output stack."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[50]], dtype=jnp.int16),
            machine_types=jnp.array([[MachineType.MINER]], dtype=jnp.int32),
            machine_power=jnp.array([[10]], dtype=jnp.int32),
            machine_inventory=_machine_inv(1, 1, y0_x0_COAL=62),
        )

        new_state = run_miners(state)

        assert new_state.machine_inventory[0, 0, ItemType.COAL] == 64
        assert new_state.block_resources[0, 0] == 48  # 50 - 2


class TestMinerDifferentOres:
    """Tests for miners on different ore types."""

    import pytest

    @pytest.mark.parametrize(
        "block_type, item_type",
        [
            (BlockType.IRON, ItemType.IRON),
            (BlockType.COPPER, ItemType.COPPER),
        ],
        ids=["iron", "copper"],
    )
    def test_miner_on_ore(self, state_factory, block_type, item_type) -> None:
        """Miner should correctly mine the given ore type."""
        state = state_factory(
            world_map=jnp.array([[block_type]], dtype=jnp.int32),
            block_resources=jnp.array([[50]], dtype=jnp.int16),
            machine_types=jnp.array([[MachineType.MINER]], dtype=jnp.int32),
            machine_power=jnp.array([[10]], dtype=jnp.int32),
        )

        new_state = run_miners(state)

        assert new_state.machine_inventory[0, 0, item_type] == 3


class TestMultipleMiners:
    """Tests for multiple miners operating in parallel."""

    def test_multiple_miners_update_in_parallel(self, state_factory) -> None:
        """Multiple miners should all update in a single step."""
        state = state_factory(
            world_map=jnp.array(
                [
                    [BlockType.COAL, BlockType.IRON],
                    [BlockType.COPPER, BlockType.DIRT],
                ],
                dtype=jnp.int32,
            ),
            block_resources=jnp.array([[50, 50], [50, 0]], dtype=jnp.int16),
            machine_types=jnp.array(
                [
                    [MachineType.MINER, MachineType.MINER],
                    [MachineType.MINER, MachineType.NONE],
                ],
                dtype=jnp.int32,
            ),
            machine_power=jnp.array([[10, 10], [10, 0]], dtype=jnp.int32),
        )

        new_state = run_miners(state)

        assert new_state.block_resources[0, 0] == 47
        assert new_state.block_resources[0, 1] == 47
        assert new_state.block_resources[1, 0] == 47
        assert new_state.block_resources[1, 1] == 0

        assert new_state.machine_inventory[0, 0, ItemType.COAL] == 3
        assert new_state.machine_inventory[0, 1, ItemType.IRON] == 3
        assert new_state.machine_inventory[1, 0, ItemType.COPPER] == 3


class TestUpdateAllMachines:
    """Tests for the combined machine update function."""

    def test_refuel_then_mine_in_same_step(self, state_factory) -> None:
        """Machine should refuel and then mine in the same step."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[50]], dtype=jnp.int16),
            machine_types=jnp.array([[MachineType.MINER]], dtype=jnp.int32),
            machine_power=jnp.array([[0]], dtype=jnp.int32),
            machine_inventory=_machine_inv(1, 1, y0_x0_COAL=5),
        )

        new_state = update_all_machines(state)

        # Refuel: 5 coal - 1 = 4 coal, but miner also mines 3 coal back
        assert new_state.machine_inventory[0, 0, ItemType.COAL] == 4 + 3
        assert new_state.machine_power[0, 0] == POWER_PER_COAL - 1
        assert new_state.block_resources[0, 0] == 47
