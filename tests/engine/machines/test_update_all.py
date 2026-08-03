"""Tests for ``update_all_machines`` in :mod:`factoriax.engine.machines`.

``update_all_machines`` runs every machine pass in order for one tick. This
file covers the ordering, the entity arrays the passes read, and the private
helpers they share.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random

from factoriax.engine.constants import BlockType, Machine
from factoriax.engine.levels import generate_state
from factoriax.engine.machines import (
    _lookup_neighbor,
    _subtract_buffer,
    update_all_machines,
)
from factoriax.engine.state import EnvParams
from tests.helpers.states import entity_at as _eid


class TestMachineInitialization:
    """Tests for machine state initialization."""

    def test_generate_state_initializes_no_machines(self) -> None:
        """A generated world has no machines by default."""
        rng = random.PRNGKey(0)
        params = EnvParams()
        state = generate_state(rng, params)

        assert jnp.all(state.machine_types == Machine.NONE)
        assert jnp.all(state.ent_power == 0)
        assert jnp.all(state.ent_buf_count == 0)

    def test_machine_arrays_match_map_shape(self) -> None:
        """The machine state arrays have the expected shapes."""
        rng = random.PRNGKey(0)
        params = EnvParams()
        state = generate_state(rng, params, 16, 24)

        assert state.machine_types.shape == state.map.shape
        mm = max(64, 16 * 24 // 4)
        assert state.ent_power.shape == (mm,)
        assert state.ent_buf_type.shape == (mm,)
        assert state.ent_buf_count.shape == (mm,)


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


class TestUpdateAllMachines:
    """Tests for the combined machine update function."""

    def test_miner_runs_via_update_all(self, state_factory) -> None:
        """update_all_machines runs miners end-to-end."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[50]], dtype=jnp.int16),
            machine_types=jnp.array([[Machine.MINER]], dtype=jnp.int32),
        )
        params = EnvParams()

        new_state = update_all_machines(state, params)

        eid = _eid(new_state, 0, 0)
        assert new_state.ent_buf_count[eid] == 3
        assert new_state.block_resources[0, 0] == 47


class TestMachineHelpers:
    """Unit tests for private machine helpers: JIT-free, no state_factory."""

    def test_subtract_buffer_decrements_by_amount(self) -> None:
        cond = jnp.array([True, True, False])
        bt = jnp.array([5, 5, 5], dtype=jnp.int8)
        bc = jnp.array([3, 2, 10], dtype=jnp.int16)
        new_bt, new_bc = _subtract_buffer(cond, bt, bc, jnp.int16(2))
        assert new_bc.tolist() == [1, 0, 10]
        assert int(new_bt[0]) == 5  # still positive
        assert int(new_bt[1]) == 0  # count hit zero, so cleared
        assert int(new_bt[2]) == 5  # unconditioned

    def test_subtract_buffer_clears_type_exactly_at_zero(self) -> None:
        cond = jnp.array([True, True])
        bt = jnp.array([7, 7], dtype=jnp.int8)
        bc = jnp.array([1, 2], dtype=jnp.int16)
        new_bt, new_bc = _subtract_buffer(cond, bt, bc, jnp.int16(1))
        assert int(new_bc[0]) == 0 and int(new_bt[0]) == 0  # zero, so cleared
        assert int(new_bc[1]) == 1 and int(new_bt[1]) == 7  # still positive

    def test_subtract_buffer_no_change_when_false(self) -> None:
        cond = jnp.array([False])
        bt = jnp.array([3], dtype=jnp.int8)
        bc = jnp.array([5], dtype=jnp.int16)
        new_bt, new_bc = _subtract_buffer(cond, bt, bc, jnp.int16(1))
        assert int(new_bc[0]) == 5
        assert int(new_bt[0]) == 3

    def test_lookup_neighbor_finds_entity_at_neighbor_tile(self) -> None:
        h, w = 3, 3
        ey = jnp.array([1], dtype=jnp.int16)
        ex = jnp.array([1], dtype=jnp.int16)
        tile_entity = jnp.full((h, w), -1, dtype=jnp.int16).at[1, 2].set(5)
        _, _, eidx, valid, diff, safe = _lookup_neighbor(
            ey, ex, 0, 1, h, w, tile_entity, 9
        )
        assert int(eidx[0]) == 5
        assert bool(valid[0])
        assert bool(diff[0])
        assert int(safe[0]) == 5

    def test_lookup_neighbor_invalid_on_empty_tile(self) -> None:
        h, w = 3, 3
        ey = jnp.array([1], dtype=jnp.int16)
        ex = jnp.array([1], dtype=jnp.int16)
        tile_entity = jnp.full((h, w), -1, dtype=jnp.int16)
        _, _, eidx, valid, _, safe = _lookup_neighbor(
            ey, ex, 0, 1, h, w, tile_entity, 9
        )
        assert int(eidx[0]) == -1
        assert not bool(valid[0])
        assert int(safe[0]) == 0  # -1 clipped to 0

    def test_lookup_neighbor_diff_false_at_grid_edge(self) -> None:
        h, w = 3, 3
        # Entity at row 0 moving UP (dy=-1): clamped ny==ey, so diff=False
        ey = jnp.array([0], dtype=jnp.int16)
        ex = jnp.array([1], dtype=jnp.int16)
        tile_entity = jnp.full((h, w), -1, dtype=jnp.int16)
        _, _, _, _, diff, _ = _lookup_neighbor(ey, ex, -1, 0, h, w, tile_entity, 9)
        assert not bool(diff[0])
