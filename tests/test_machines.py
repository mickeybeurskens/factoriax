"""Tests for the machine system."""

import jax.numpy as jnp
from jax import random

from factoriax import BlockType, EnvParams, ItemType
from factoriax.constants import (
    MAX_MACHINE_STACK_SIZE,
    POWER_PER_COAL,
    MachineType,
)
from factoriax.machines import refuel_machines, run_miners, update_all_machines
from factoriax.world_gen import generate_world


class TestMachineInitialization:
    """Tests for machine state initialization."""

    def test_world_gen_initializes_no_machines(self) -> None:
        """Generated world should have no machines by default."""
        rng = random.PRNGKey(0)
        params = EnvParams()
        state = generate_world(rng, params)

        assert jnp.all(state.machine_types == MachineType.NONE)
        assert jnp.all(state.machine_power == 0)
        assert jnp.all(state.machine_fuel_count == 0)
        assert jnp.all(state.machine_output_count == 0)

    def test_machine_arrays_match_map_shape(self) -> None:
        """Machine state arrays should have same shape as map."""
        rng = random.PRNGKey(0)
        params = EnvParams(map_width=16, map_height=24)
        state = generate_world(rng, params)

        assert state.machine_types.shape == state.map.shape
        assert state.machine_power.shape == state.map.shape
        assert state.machine_fuel_count.shape == state.map.shape
        assert state.machine_output_item.shape == state.map.shape
        assert state.machine_output_count.shape == state.map.shape


class TestMachineRefueling:
    """Tests for machine refueling logic."""

    def test_machine_consumes_coal_when_no_power(self, state_factory) -> None:
        """Machine with no power and coal should consume coal and gain power."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[10]], dtype=jnp.int16),
            machine_types=jnp.array([[MachineType.MINER]], dtype=jnp.int32),
            machine_power=jnp.array([[0]], dtype=jnp.int32),
            machine_fuel_count=jnp.array([[5]], dtype=jnp.int16),
        )

        new_state = refuel_machines(state)

        assert new_state.machine_fuel_count[0, 0] == 4
        assert new_state.machine_power[0, 0] == POWER_PER_COAL

    def test_machine_does_not_refuel_with_power(self, state_factory) -> None:
        """Machine with power should not consume coal."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[10]], dtype=jnp.int16),
            machine_types=jnp.array([[MachineType.MINER]], dtype=jnp.int32),
            machine_power=jnp.array([[5]], dtype=jnp.int32),
            machine_fuel_count=jnp.array([[5]], dtype=jnp.int16),
        )

        new_state = refuel_machines(state)

        assert new_state.machine_fuel_count[0, 0] == 5
        assert new_state.machine_power[0, 0] == 5

    def test_machine_does_not_refuel_without_coal(self, state_factory) -> None:
        """Machine without coal cannot refuel."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[10]], dtype=jnp.int16),
            machine_types=jnp.array([[MachineType.MINER]], dtype=jnp.int32),
            machine_power=jnp.array([[0]], dtype=jnp.int32),
            machine_fuel_count=jnp.array([[0]], dtype=jnp.int16),
        )

        new_state = refuel_machines(state)

        assert new_state.machine_fuel_count[0, 0] == 0
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
        assert new_state.machine_output_count[0, 0] == 3
        assert new_state.machine_output_item[0, 0] == ItemType.COAL

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
        assert new_state.machine_output_count[0, 0] == 0

    def test_miner_stops_when_output_full(self, state_factory) -> None:
        """Miner should stop when output slot is full."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[50]], dtype=jnp.int16),
            machine_types=jnp.array([[MachineType.MINER]], dtype=jnp.int32),
            machine_power=jnp.array([[10]], dtype=jnp.int32),
            machine_output_item=jnp.array([[ItemType.COAL]], dtype=jnp.int32),
            machine_output_count=jnp.array([[MAX_MACHINE_STACK_SIZE]], dtype=jnp.int16),
        )

        new_state = run_miners(state)

        assert new_state.block_resources[0, 0] == 50
        assert new_state.machine_output_count[0, 0] == MAX_MACHINE_STACK_SIZE
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

        assert new_state.machine_output_count[0, 0] == 0
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
        assert new_state.machine_output_count[0, 0] == 2

    def test_miner_mines_partial_when_limited_by_resources(self, state_factory) -> None:
        """Miner should extract only available resources when less than rate."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[1]], dtype=jnp.int16),
            machine_types=jnp.array([[MachineType.MINER]], dtype=jnp.int32),
            machine_power=jnp.array([[10]], dtype=jnp.int32),
        )

        new_state = run_miners(state)

        assert new_state.block_resources[0, 0] == 0
        assert new_state.machine_output_count[0, 0] == 1

    def test_miner_mines_partial_when_limited_by_space(self, state_factory) -> None:
        """Miner should extract only what fits in output slot."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[50]], dtype=jnp.int16),
            machine_types=jnp.array([[MachineType.MINER]], dtype=jnp.int32),
            machine_power=jnp.array([[10]], dtype=jnp.int32),
            machine_output_item=jnp.array([[ItemType.COAL]], dtype=jnp.int32),
            machine_output_count=jnp.array([[62]], dtype=jnp.int16),
        )

        new_state = run_miners(state)

        assert new_state.machine_output_count[0, 0] == 64  # 62 + 2 (capped)
        assert new_state.block_resources[0, 0] == 48  # 50 - 2


class TestMinerDifferentOres:
    """Tests for miners on different ore types."""

    def test_miner_on_iron(self, state_factory) -> None:
        """Miner should correctly mine iron ore."""
        state = state_factory(
            world_map=jnp.array([[BlockType.IRON]], dtype=jnp.int32),
            block_resources=jnp.array([[50]], dtype=jnp.int16),
            machine_types=jnp.array([[MachineType.MINER]], dtype=jnp.int32),
            machine_power=jnp.array([[10]], dtype=jnp.int32),
        )

        new_state = run_miners(state)

        assert new_state.machine_output_item[0, 0] == ItemType.IRON
        assert new_state.machine_output_count[0, 0] == 3

    def test_miner_on_copper(self, state_factory) -> None:
        """Miner should correctly mine copper ore."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COPPER]], dtype=jnp.int32),
            block_resources=jnp.array([[50]], dtype=jnp.int16),
            machine_types=jnp.array([[MachineType.MINER]], dtype=jnp.int32),
            machine_power=jnp.array([[10]], dtype=jnp.int32),
        )

        new_state = run_miners(state)

        assert new_state.machine_output_item[0, 0] == ItemType.COPPER
        assert new_state.machine_output_count[0, 0] == 3


class TestMultipleMiners:
    """Tests for multiple miners operating in parallel."""

    def test_multiple_miners_update_in_parallel(self, state_factory) -> None:
        """Multiple miners should all update in a single step."""
        state = state_factory(
            world_map=jnp.array(
                [[BlockType.COAL, BlockType.IRON], [BlockType.COPPER, BlockType.DIRT]],
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

        assert new_state.machine_output_item[0, 0] == ItemType.COAL
        assert new_state.machine_output_item[0, 1] == ItemType.IRON
        assert new_state.machine_output_item[1, 0] == ItemType.COPPER


class TestUpdateAllMachines:
    """Tests for the combined machine update function."""

    def test_refuel_then_mine_in_same_step(self, state_factory) -> None:
        """Machine should refuel and then mine in the same step."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[50]], dtype=jnp.int16),
            machine_types=jnp.array([[MachineType.MINER]], dtype=jnp.int32),
            machine_power=jnp.array([[0]], dtype=jnp.int32),
            machine_fuel_count=jnp.array([[5]], dtype=jnp.int16),
        )

        new_state = update_all_machines(state)

        assert new_state.machine_fuel_count[0, 0] == 4
        assert new_state.machine_power[0, 0] == POWER_PER_COAL - 1
        assert new_state.machine_output_count[0, 0] == 3
        assert new_state.block_resources[0, 0] == 47
