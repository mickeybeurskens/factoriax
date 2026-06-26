"""Tests for the entity-based machine inventory system.

Covers machine entity state fields, constraints per machine type,
and basic machine operations using entity arrays.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from factoriax.engine.constants import BlockType, ItemType
from factoriax.engine.state import EnvParams, EnvState
from factoriax.engine.constants import Machine
from factoriax.engine.levels import generate_state
from factoriax.engine.machine_spec import MACHINE_MAX_STACK, MACHINE_MAX_TYPES
from factoriax.engine.machines import run_miners, update_all_machines

# The miner's buffer capacity — a "full" buffer value for the stop test.
_MINER_BUF_CAP = int(MACHINE_MAX_STACK[int(Machine.MINER)])


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


class TestEntityStateFields:
    """Verify entity state fields have correct properties."""

    def test_entity_dtypes(self) -> None:
        """Entity buffer arrays should use expected dtypes."""
        params = EnvParams(num_players=1)
        state = generate_state(jax.random.PRNGKey(0), params, 4, 4)
        assert state.ent_buf_type.dtype == jnp.int8
        assert state.ent_buf_count.dtype == jnp.int16

    def test_entity_shapes(self) -> None:
        """Entity arrays should be 1-D with max_machines length."""
        params = EnvParams(num_players=1)
        state = generate_state(jax.random.PRNGKey(0), params, 4, 4)
        mm = 64
        assert state.ent_buf_type.shape == (mm,)
        assert state.ent_buf_count.shape == (mm,)


class TestGenerateStateEntityFields:
    """Verify generated state has correct entity initialization."""

    def test_shapes(self) -> None:
        """Generated state entity arrays should match max_machines."""
        params = EnvParams(num_players=1)
        state = generate_state(jax.random.PRNGKey(0), params, 8, 6)
        mm = max(64, 8 * 6 // 4)
        assert state.ent_buf_type.shape == (mm,)
        assert state.ent_buf_count.shape == (mm,)

    def test_zero_initialized(self) -> None:
        """Generated state should have zero-initialized entity buffers."""
        params = EnvParams(num_players=1)
        state = generate_state(jax.random.PRNGKey(0), params)
        assert jnp.all(state.ent_buf_count == 0)


class TestMinerInventory:
    """Tests for miner inventory operations using entity buffers."""

    def test_run_miners_deposits_ore(self, state_factory) -> None:
        """Miners should deposit ore into their entity buffer."""
        state = state_factory(
            world_map=jnp.array(
                [[BlockType.IRON]],
                dtype=jnp.int32,
            ),
            block_resources=jnp.array([[50]], dtype=jnp.int16),
            machine_types=jnp.array(
                [[Machine.MINER]],
                dtype=jnp.int32,
            ),
        )
        params = EnvParams(num_players=1)
        new = run_miners(state, params)
        eid = _eid(new, 0, 0)
        assert int(new.ent_buf_count[eid]) > 0
        assert int(new.ent_buf_type[eid]) == int(ItemType.IRON_ORE)

    def test_output_full_blocks_mining(self, state_factory) -> None:
        """Miner should not mine when buffer is at max stack."""
        state = state_factory(
            world_map=jnp.array(
                [[BlockType.IRON]],
                dtype=jnp.int32,
            ),
            block_resources=jnp.array([[50]], dtype=jnp.int16),
            machine_types=jnp.array(
                [[Machine.MINER]],
                dtype=jnp.int32,
            ),
            buffer_type=jnp.array(
                [[int(ItemType.IRON_ORE)]],
                dtype=jnp.int8,
            ),
            buffer_count=jnp.array(
                [[_MINER_BUF_CAP]],
                dtype=jnp.int16,
            ),
        )
        params = EnvParams(num_players=1)
        new = run_miners(state, params)
        eid = _eid(new, 0, 0)
        assert int(new.ent_buf_count[eid]) == _MINER_BUF_CAP
        assert int(new.block_resources[0, 0]) == 50


class TestPalletInventory:
    """Tests for pallet inventory."""

    def test_pallet_initialized_empty(self, state_factory) -> None:
        """Pallet should start with empty buffer."""
        state = state_factory(
            world_map=jnp.array(
                [[BlockType.DIRT]],
                dtype=jnp.int32,
            ),
            machine_types=jnp.array(
                [[Machine.PALLET]],
                dtype=jnp.int32,
            ),
        )
        eid = _eid(state, 0, 0)
        assert int(state.ent_buf_count[eid]) == 0

    def test_pallet_unaffected_by_update(self, state_factory) -> None:
        """update_all_machines should not modify pallet contents."""
        state = state_factory(
            world_map=jnp.array(
                [[BlockType.DIRT]],
                dtype=jnp.int32,
            ),
            machine_types=jnp.array(
                [[Machine.PALLET]],
                dtype=jnp.int32,
            ),
            buffer_type=jnp.array(
                [[int(ItemType.IRON_ORE)]],
                dtype=jnp.int8,
            ),
            buffer_count=jnp.array([[10]], dtype=jnp.int16),
        )
        params = EnvParams(num_players=1)
        new = update_all_machines(state, params)
        eid = _eid(new, 0, 0)
        assert int(new.ent_buf_count[eid]) == 10
        assert int(new.ent_buf_type[eid]) == int(ItemType.IRON_ORE)


class TestAssemblerInventory:
    """Tests for assembler inventory."""

    def test_assembler_initialized_empty(
        self,
        state_factory,
    ) -> None:
        """Assembler should start with empty buffers."""
        state = state_factory(
            world_map=jnp.array(
                [[BlockType.DIRT]],
                dtype=jnp.int32,
            ),
            machine_types=jnp.array(
                [[Machine.ASSEMBLER]],
                dtype=jnp.int32,
            ),
        )
        eid = _eid(state, 0, 0)
        assert int(state.ent_buf_count[eid]) == 0
        assert jnp.all(state.ent_asm_in_count[eid] == 0)
        assert int(state.ent_asm_out_count[eid]) == 0

    def test_assembler_idle_without_inputs(
        self,
        state_factory,
    ) -> None:
        """Assembler without inputs should remain idle."""
        state = state_factory(
            world_map=jnp.array(
                [[BlockType.DIRT]],
                dtype=jnp.int32,
            ),
            machine_types=jnp.array(
                [[Machine.ASSEMBLER]],
                dtype=jnp.int32,
            ),
        )
        params = EnvParams(num_players=1)
        new = update_all_machines(state, params)
        eid = _eid(new, 0, 0)
        assert int(new.ent_buf_count[eid]) == 0
        assert jnp.all(new.ent_asm_in_count[eid] == 0)
        assert int(new.ent_asm_out_count[eid]) == 0


class TestMaxTypesConstraint:
    """Tests for the MACHINE_MAX_TYPES constraint."""

    def test_belt_max_types_is_one(self) -> None:
        """Belt should hold at most 1 distinct item type."""
        assert int(MACHINE_MAX_TYPES[Machine.CONVEYOR_BELT]) == 1

    def test_pallet_max_types_is_one(self) -> None:
        """Pallet should hold at most 1 distinct item type."""
        assert int(MACHINE_MAX_TYPES[Machine.PALLET]) == 1

    def test_assembler_max_types_is_four(self) -> None:
        """Assembler should hold at most 4 distinct item types."""
        assert int(MACHINE_MAX_TYPES[Machine.ASSEMBLER]) == 4
