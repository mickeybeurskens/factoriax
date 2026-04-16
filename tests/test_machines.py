"""Tests for the machine system."""

import jax.numpy as jnp
from jax import random

from factoriax import BlockType, EnvParams, EnvState, ItemType
from factoriax.constants import (
    MAX_MACHINE_STACK_SIZE,
    POWER_PER_COAL,
    MachineType,
)
from factoriax.machines import refuel_machines, run_miners, update_all_machines
from factoriax.world_gen import generate_world


def _eid(state: EnvState, y: int, x: int) -> int:
    """Look up the entity index at grid position (y, x).

    Args:
        state: Environment state with tile_entity grid.
        y: Row index.
        x: Column index.

    Returns:
        Entity index at the given tile.
    """
    return int(state.tile_entity[y, x])


class TestMachineInitialization:
    """Tests for machine state initialization."""

    def test_world_gen_initializes_no_machines(self) -> None:
        """Generated world should have no machines by default."""
        rng = random.PRNGKey(0)
        params = EnvParams()
        state = generate_world(rng, params)

        assert jnp.all(state.machine_types == MachineType.NONE)
        assert jnp.all(state.ent_power == 0)
        assert jnp.all(state.ent_buf_count == 0)
        assert jnp.all(state.ent_fuel == 0)

    def test_machine_arrays_match_map_shape(self) -> None:
        """Machine state arrays should have the expected shapes."""
        rng = random.PRNGKey(0)
        params = EnvParams(map_width=16, map_height=24)
        state = generate_world(rng, params)

        assert state.machine_types.shape == state.map.shape
        mm = params.resolved_max_machines()
        assert state.ent_power.shape == (mm,)
        assert state.ent_buf_type.shape == (mm,)
        assert state.ent_buf_count.shape == (mm,)


class TestMachineRefueling:
    """Tests for machine refueling logic."""

    def test_machine_consumes_coal_when_no_power(self, state_factory) -> None:
        """Machine with no power and coal should consume coal and gain power."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[10]], dtype=jnp.int16),
            machine_types=jnp.array([[MachineType.MINER]], dtype=jnp.int32),
            machine_power=jnp.array([[0]], dtype=jnp.int32),
            machine_fuel=jnp.array([[5]], dtype=jnp.int16),
        )
        params = EnvParams(map_width=1, map_height=1)

        new_state = refuel_machines(state, params)

        eid = _eid(new_state, 0, 0)
        assert new_state.ent_fuel[eid] == 4
        assert new_state.ent_power[eid] == POWER_PER_COAL

    def test_machine_does_not_refuel_with_power(self, state_factory) -> None:
        """Machine with power should not consume coal."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[10]], dtype=jnp.int16),
            machine_types=jnp.array([[MachineType.MINER]], dtype=jnp.int32),
            machine_power=jnp.array([[5]], dtype=jnp.int32),
            machine_fuel=jnp.array([[5]], dtype=jnp.int16),
        )
        params = EnvParams(map_width=1, map_height=1)

        new_state = refuel_machines(state, params)

        eid = _eid(new_state, 0, 0)
        assert new_state.ent_fuel[eid] == 5
        assert new_state.ent_power[eid] == 5

    def test_machine_does_not_refuel_without_coal(self, state_factory) -> None:
        """Machine without coal cannot refuel."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[10]], dtype=jnp.int16),
            machine_types=jnp.array([[MachineType.MINER]], dtype=jnp.int32),
            machine_power=jnp.array([[0]], dtype=jnp.int32),
        )
        params = EnvParams(map_width=1, map_height=1)

        new_state = refuel_machines(state, params)

        eid = _eid(new_state, 0, 0)
        assert new_state.ent_fuel[eid] == 0
        assert new_state.ent_power[eid] == 0

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
        params = EnvParams(map_width=1, map_height=1)

        new_state = run_miners(state, params)

        eid = _eid(new_state, 0, 0)
        assert new_state.block_resources[0, 0] == 47  # 50 - 3 (mining rate)
        assert new_state.ent_buf_count[eid] == 3
        assert new_state.ent_buf_type[eid] == ItemType.COAL

    def test_miner_consumes_power_when_mining(self, state_factory) -> None:
        """Miner should consume 1 power per step when mining."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[50]], dtype=jnp.int16),
            machine_types=jnp.array([[MachineType.MINER]], dtype=jnp.int32),
            machine_power=jnp.array([[10]], dtype=jnp.int32),
        )
        params = EnvParams(map_width=1, map_height=1)

        new_state = run_miners(state, params)

        eid = _eid(new_state, 0, 0)
        assert new_state.ent_power[eid] == 9

    def test_miner_does_not_consume_power_when_idle(self, state_factory) -> None:
        """Miner should not consume power when it cannot mine."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            machine_types=jnp.array([[MachineType.MINER]], dtype=jnp.int32),
            machine_power=jnp.array([[10]], dtype=jnp.int32),
        )
        params = EnvParams(map_width=1, map_height=1)

        new_state = run_miners(state, params)

        eid = _eid(new_state, 0, 0)
        assert new_state.ent_power[eid] == 10

    def test_miner_stops_when_no_power(self, state_factory) -> None:
        """Miner without power should not extract resources."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[50]], dtype=jnp.int16),
            machine_types=jnp.array([[MachineType.MINER]], dtype=jnp.int32),
            machine_power=jnp.array([[0]], dtype=jnp.int32),
        )
        params = EnvParams(map_width=1, map_height=1)

        new_state = run_miners(state, params)

        eid = _eid(new_state, 0, 0)
        assert new_state.block_resources[0, 0] == 50
        assert new_state.ent_buf_count[eid] == 0

    def test_miner_stops_when_output_full(self, state_factory) -> None:
        """Miner should stop when output type is at max stack."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[50]], dtype=jnp.int16),
            machine_types=jnp.array([[MachineType.MINER]], dtype=jnp.int32),
            machine_power=jnp.array([[10]], dtype=jnp.int32),
            buffer_type=jnp.array([[int(ItemType.COAL)]], dtype=jnp.int8),
            buffer_count=jnp.array(
                [[MAX_MACHINE_STACK_SIZE]],
                dtype=jnp.int16,
            ),
        )
        params = EnvParams(map_width=1, map_height=1)

        new_state = run_miners(state, params)

        eid = _eid(new_state, 0, 0)
        assert new_state.block_resources[0, 0] == 50
        assert new_state.ent_buf_count[eid] == MAX_MACHINE_STACK_SIZE
        assert new_state.ent_power[eid] == 10  # No power consumed

    def test_miner_stops_when_no_resources(self, state_factory) -> None:
        """Miner should stop when block has no resources."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[0]], dtype=jnp.int16),
            machine_types=jnp.array([[MachineType.MINER]], dtype=jnp.int32),
            machine_power=jnp.array([[10]], dtype=jnp.int32),
        )
        params = EnvParams(map_width=1, map_height=1)

        new_state = run_miners(state, params)

        eid = _eid(new_state, 0, 0)
        assert new_state.ent_buf_count[eid] == 0
        assert new_state.ent_power[eid] == 10

    def test_miner_depletes_block_to_dirt(self, state_factory) -> None:
        """Block should become dirt when fully depleted by miner."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[2]], dtype=jnp.int16),
            machine_types=jnp.array([[MachineType.MINER]], dtype=jnp.int32),
            machine_power=jnp.array([[10]], dtype=jnp.int32),
            max_machines=1,
        )
        params = EnvParams(map_width=1, map_height=1)

        new_state = run_miners(state, params)

        eid = _eid(new_state, 0, 0)
        assert new_state.map[0, 0] == BlockType.DIRT
        assert new_state.block_resources[0, 0] == 0
        assert new_state.ent_buf_count[eid] == 2

    def test_miner_mines_partial_when_limited_by_resources(
        self,
        state_factory,
    ) -> None:
        """Miner should extract only available resources when less than rate."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[1]], dtype=jnp.int16),
            machine_types=jnp.array([[MachineType.MINER]], dtype=jnp.int32),
            machine_power=jnp.array([[10]], dtype=jnp.int32),
        )
        params = EnvParams(map_width=1, map_height=1)

        new_state = run_miners(state, params)

        eid = _eid(new_state, 0, 0)
        assert new_state.block_resources[0, 0] == 0
        assert new_state.ent_buf_count[eid] == 1

    def test_miner_mines_partial_when_limited_by_space(
        self,
        state_factory,
    ) -> None:
        """Miner should extract only what fits in output stack."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[50]], dtype=jnp.int16),
            machine_types=jnp.array([[MachineType.MINER]], dtype=jnp.int32),
            machine_power=jnp.array([[10]], dtype=jnp.int32),
            buffer_type=jnp.array([[int(ItemType.COAL)]], dtype=jnp.int8),
            buffer_count=jnp.array([[62]], dtype=jnp.int16),
        )
        params = EnvParams(map_width=1, map_height=1)

        new_state = run_miners(state, params)

        eid = _eid(new_state, 0, 0)
        assert new_state.ent_buf_count[eid] == 64
        assert new_state.block_resources[0, 0] == 48  # 50 - 2


class TestMinerDifferentOres:
    """Tests for miners on different ore types."""

    import pytest

    @pytest.mark.parametrize(
        "block_type, item_type",
        [
            (BlockType.IRON, ItemType.IRON_ORE),
            (BlockType.COPPER, ItemType.COPPER_ORE),
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
        params = EnvParams(map_width=1, map_height=1)

        new_state = run_miners(state, params)

        eid = _eid(new_state, 0, 0)
        assert new_state.ent_buf_count[eid] == 3
        assert new_state.ent_buf_type[eid] == item_type


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
        params = EnvParams(map_width=2, map_height=2)

        new_state = run_miners(state, params)

        assert new_state.block_resources[0, 0] == 47
        assert new_state.block_resources[0, 1] == 47
        assert new_state.block_resources[1, 0] == 47
        assert new_state.block_resources[1, 1] == 0

        eid_coal = _eid(new_state, 0, 0)
        eid_iron = _eid(new_state, 0, 1)
        eid_copper = _eid(new_state, 1, 0)
        assert new_state.ent_buf_count[eid_coal] == 3
        assert new_state.ent_buf_type[eid_coal] == ItemType.COAL
        assert new_state.ent_buf_count[eid_iron] == 3
        assert new_state.ent_buf_type[eid_iron] == ItemType.IRON_ORE
        assert new_state.ent_buf_count[eid_copper] == 3
        assert new_state.ent_buf_type[eid_copper] == ItemType.COPPER_ORE


class TestUpdateAllMachines:
    """Tests for the combined machine update function."""

    def test_refuel_then_mine_in_same_step(self, state_factory) -> None:
        """Machine should refuel and then mine in the same step."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[50]], dtype=jnp.int16),
            machine_types=jnp.array([[MachineType.MINER]], dtype=jnp.int32),
            machine_power=jnp.array([[0]], dtype=jnp.int32),
            machine_fuel=jnp.array([[5]], dtype=jnp.int16),
        )
        params = EnvParams(map_width=1, map_height=1)

        new_state = update_all_machines(state, params)

        eid = _eid(new_state, 0, 0)
        # Refuel: 5 fuel - 1 = 4, power = 10. Mine: 3 coal into buffer.
        assert new_state.ent_buf_count[eid] == 3
        assert new_state.ent_fuel[eid] == 4
        assert new_state.ent_power[eid] == POWER_PER_COAL - 1
        assert new_state.block_resources[0, 0] == 47
