"""Tests for the entity-based machine inventory system.

Covers machine entity state fields, constraints per machine type,
and basic machine operations using entity arrays.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from factoriax.engine.constants import BlockType, ItemType, Machine
from factoriax.engine.levels import generate_state
from factoriax.engine.machines import run_miners, update_all_machines
from factoriax.engine.state import EnvParams, EnvState
from factoriax.engine.tables import MACHINE_MAX_STACK

# The miner's buffer capacity, used as the "full" value in the stop test.
_MINER_BUF_CAP = int(MACHINE_MAX_STACK[int(Machine.MINER)])


def _eid(state: EnvState, y: int, x: int) -> int:
    """Look up the entity index at grid position (y, x).

    Parameters
    ----------
    state
        Environment state with a ``tile_entity`` grid.
    y
        Row index.
    x
        Column index.

    Returns
    -------
    int
        Entity index at the given tile.
    """
    return int(state.tile_entity[y, x])


class TestEntityStateFields:
    """Verify entity state fields have correct properties."""

    def test_entity_dtypes(self) -> None:
        """Entity buffer arrays use the expected dtypes."""
        params = EnvParams()
        state = generate_state(jax.random.PRNGKey(0), params, 4, 4)
        assert state.ent_buf_type.dtype == jnp.int8
        assert state.ent_buf_count.dtype == jnp.int16

    def test_entity_shapes(self) -> None:
        """Entity arrays are 1-D, with a length of max_machines."""
        params = EnvParams()
        state = generate_state(jax.random.PRNGKey(0), params, 4, 4)
        mm = 64
        assert state.ent_buf_type.shape == (mm,)
        assert state.ent_buf_count.shape == (mm,)


class TestGenerateStateEntityFields:
    """Verify generated state has correct entity initialization."""

    def test_shapes(self) -> None:
        """The entity arrays of a generated state match max_machines."""
        params = EnvParams()
        state = generate_state(jax.random.PRNGKey(0), params, 8, 6)
        mm = max(64, 8 * 6 // 4)
        assert state.ent_buf_type.shape == (mm,)
        assert state.ent_buf_count.shape == (mm,)

    def test_zero_initialized(self) -> None:
        """A generated state has zero-initialized entity buffers."""
        params = EnvParams()
        state = generate_state(jax.random.PRNGKey(0), params)
        assert jnp.all(state.ent_buf_count == 0)


class TestMinerInventory:
    """Tests for miner inventory operations using entity buffers."""

    def test_run_miners_deposits_ore(self, state_factory) -> None:
        """A miner deposits ore into its entity buffer."""
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
        params = EnvParams()
        new = run_miners(state, params)
        eid = _eid(new, 0, 0)
        assert int(new.ent_buf_count[eid]) > 0
        assert int(new.ent_buf_type[eid]) == int(ItemType.IRON_ORE)

    def test_output_full_blocks_mining(self, state_factory) -> None:
        """A miner does not mine when its buffer is at max stack."""
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
        params = EnvParams()
        new = run_miners(state, params)
        eid = _eid(new, 0, 0)
        assert int(new.ent_buf_count[eid]) == _MINER_BUF_CAP
        assert int(new.block_resources[0, 0]) == 50


class TestPalletInventory:
    """Tests for pallet inventory."""

    def test_pallet_initialized_empty(self, state_factory) -> None:
        """A pallet starts with an empty buffer."""
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
        """update_all_machines does not modify the pallet contents."""
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
        params = EnvParams()
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
        """An assembler starts with empty buffers."""
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
        """An assembler without inputs stays idle."""
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
        params = EnvParams()
        new = update_all_machines(state, params)
        eid = _eid(new, 0, 0)
        assert int(new.ent_buf_count[eid]) == 0
        assert jnp.all(new.ent_asm_in_count[eid] == 0)
        assert int(new.ent_asm_out_count[eid]) == 0
