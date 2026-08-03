"""Tests for the conveyor belt pass in :mod:`factoriax.engine.machines`.

``run_conveyor_belts`` folds belts, splitters, and crossings into one
scatter-gather pass. This file covers the plain belt: the push, the merge, and
what a blocked destination leaves in place.
"""

from __future__ import annotations

import jax.numpy as jnp

from factoriax.engine.constants import (
    BlockType,
    Direction,
    ItemType,
    Machine,
)
from factoriax.engine.machines import run_conveyor_belts
from factoriax.engine.state import EnvParams, EnvState
from factoriax.engine.tables import MACHINE_MAX_STACK
from tests.helpers.states import entity_at as _eid

_MINER_BUF_CAP = int(MACHINE_MAX_STACK[int(Machine.MINER)])

_MINER_BUF_CAP = int(MACHINE_MAX_STACK[4])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Per MACHINE_MAX_STACK: NONE=0, MINER=64, PALLET=256, ASM=1000, BELT=3,
# ARM=1, ROCKET=0.
_BELT_MAX = 3
_NOOP = 0  # Direction that does not push anywhere.

# Default params for run_conveyor_belts / run_arms calls.
_PARAMS = EnvParams()


def _buf_grids(
    shape: tuple[int, int],
    entries: dict[tuple[int, int], tuple[int, int]] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build buffer_type and buffer_count grids.

    Parameters
    ----------
    shape
        Grid ``(H, W)``.
    entries
        Mapping of ``(y, x)`` to ``(item_type, count)``.

    Returns
    -------
    tuple of jnp.ndarray
        ``(buffer_type, buffer_count)``, each shaped ``(H, W)``.
    """
    bt = jnp.zeros(shape, dtype=jnp.int8)
    bc = jnp.zeros(shape, dtype=jnp.int16)
    for (y, x), (item_type, count) in (entries or {}).items():
        bt = bt.at[y, x].set(item_type)
        bc = bc.at[y, x].set(count)
    return bt, bc


def _make_state(
    state_factory,
    *,
    machine_types: jnp.ndarray,
    machine_direction: jnp.ndarray,
    buffer_type: jnp.ndarray | None = None,
    buffer_count: jnp.ndarray | None = None,
):
    """Build a state with the given machine grid and optional buffer.

    Uses max_machines equal to the actual entity count to avoid
    inactive-entity scatter collisions in JAX vectorised loops.

    Parameters
    ----------
    state_factory
        Pytest fixture that creates an ``EnvState``.
    machine_types
        Machine type per tile.
    machine_direction
        Direction per tile.
    buffer_type
        Buffer item type per tile. Optional.
    buffer_count
        Buffer item count per tile. Optional.

    Returns
    -------
    EnvState
        Configured state.
    """
    shape = machine_types.shape
    n_entities = int((machine_types != int(Machine.NONE)).sum())
    return state_factory(
        world_map=jnp.zeros(shape, dtype=jnp.int32),
        machine_types=machine_types,
        machine_direction=machine_direction,
        buffer_type=buffer_type,
        buffer_count=buffer_count,
        max_machines=n_entities,
    )


def _get_buf(state, y: int, x: int) -> tuple[int, int]:
    """Return (buf_type, buf_count) for the entity at tile (y, x).

    Parameters
    ----------
    state
        Environment state.
    y
        Tile row.
    x
        Tile column.

    Returns
    -------
    tuple of int
        ``(item_type, count)``, or ``(0, 0)`` when the tile is empty.
    """
    eid = int(state.tile_entity[y, x])
    if eid < 0:
        return (0, 0)
    return (int(state.ent_buf_type[eid]), int(state.ent_buf_count[eid]))


# ---------------------------------------------------------------------------
# Conveyor belt tests
# ---------------------------------------------------------------------------

B = Machine.CONVEYOR_BELT
R = int(Direction.RIGHT)
D = int(Direction.DOWN)


class TestConveyorBelt:
    """Conveyor belt item transport tests."""

    def test_noop_when_no_items(self, state_factory) -> None:
        """Belt with empty inventory does nothing."""
        types = jnp.array([[B, B]])
        dirs = jnp.array([[R, _NOOP]])
        state = _make_state(state_factory, machine_types=types, machine_direction=dirs)
        result = run_conveyor_belts(state, _PARAMS)
        assert _get_buf(result, 0, 0) == (0, 0)
        assert _get_buf(result, 0, 1) == (0, 0)

    def test_pushes_item_right(self, state_factory) -> None:
        """Belt facing right moves items from (0,0) to (0,1)."""
        types = jnp.array([[B, B]])
        dirs = jnp.array([[R, _NOOP]])
        bt, bc = _buf_grids((1, 2), {(0, 0): (ItemType.COAL, 2)})
        state = _make_state(
            state_factory,
            machine_types=types,
            machine_direction=dirs,
            buffer_type=bt,
            buffer_count=bc,
        )
        result = run_conveyor_belts(state, _PARAMS)
        _, src_count = _get_buf(result, 0, 0)
        dst_type, dst_count = _get_buf(result, 0, 1)
        assert src_count == 0
        assert dst_type == int(ItemType.COAL)
        assert dst_count == 2

    def test_pushes_item_down(self, state_factory) -> None:
        """Belt facing down moves items from row 0 to row 1."""
        types = jnp.array([[B], [B]])
        dirs = jnp.array([[D], [_NOOP]])
        bt, bc = _buf_grids((2, 1), {(0, 0): (ItemType.IRON_ORE, 2)})
        state = _make_state(
            state_factory,
            machine_types=types,
            machine_direction=dirs,
            buffer_type=bt,
            buffer_count=bc,
        )
        result = run_conveyor_belts(state, _PARAMS)
        _, src_count = _get_buf(result, 0, 0)
        dst_type, dst_count = _get_buf(result, 1, 0)
        assert src_count == 0
        assert dst_type == int(ItemType.IRON_ORE)
        assert dst_count == 2

    def test_does_not_push_to_empty_tile(self, state_factory) -> None:
        """Belt facing an empty tile does not transfer items."""
        types = jnp.array([[B, Machine.NONE]])
        dirs = jnp.array([[R, _NOOP]])
        bt, bc = _buf_grids((1, 2), {(0, 0): (ItemType.COAL, 2)})
        state = _make_state(
            state_factory,
            machine_types=types,
            machine_direction=dirs,
            buffer_type=bt,
            buffer_count=bc,
        )
        result = run_conveyor_belts(state, _PARAMS)
        src_type, src_count = _get_buf(result, 0, 0)
        assert src_type == int(ItemType.COAL)
        assert src_count == 2

    def test_does_not_push_to_blocked_target(self, state_factory) -> None:
        """Belt does not push when target holds a different item at max."""
        types = jnp.array([[B, B]])
        dirs = jnp.array([[R, _NOOP]])
        bt, bc = _buf_grids(
            (1, 2),
            {
                (0, 0): (ItemType.COAL, 2),
                (0, 1): (ItemType.IRON_ORE, _BELT_MAX),
            },
        )
        state = _make_state(
            state_factory,
            machine_types=types,
            machine_direction=dirs,
            buffer_type=bt,
            buffer_count=bc,
        )
        result = run_conveyor_belts(state, _PARAMS)
        src_type, src_count = _get_buf(result, 0, 0)
        assert src_type == int(ItemType.COAL)
        assert src_count == 2

    def test_merges_same_item_into_target(self, state_factory) -> None:
        """Items of the same type merge into the target's existing stack."""
        types = jnp.array([[B, B]])
        dirs = jnp.array([[R, _NOOP]])
        bt, bc = _buf_grids(
            (1, 2),
            {(0, 0): (ItemType.COAL, 1), (0, 1): (ItemType.COAL, 2)},
        )
        state = _make_state(
            state_factory,
            machine_types=types,
            machine_direction=dirs,
            buffer_type=bt,
            buffer_count=bc,
        )
        result = run_conveyor_belts(state, _PARAMS)
        dst_type, dst_count = _get_buf(result, 0, 1)
        assert dst_type == int(ItemType.COAL)
        assert dst_count == 3
        _, src_count = _get_buf(result, 0, 0)
        assert src_count == 0

    def test_noop_with_zero_direction(self, state_factory) -> None:
        """Belt with NOOP direction (0) does not push."""
        types = jnp.array([[B]])
        dirs = jnp.array([[_NOOP]])
        bt, bc = _buf_grids((1, 1), {(0, 0): (ItemType.COAL, 2)})
        state = _make_state(
            state_factory,
            machine_types=types,
            machine_direction=dirs,
            buffer_type=bt,
            buffer_count=bc,
        )
        result = run_conveyor_belts(state, _PARAMS)
        buf_type, buf_count = _get_buf(result, 0, 0)
        assert buf_type == int(ItemType.COAL)
        assert buf_count == 2


# ---------------------------------------------------------------------------
# Pick-and-place arm tests
# ---------------------------------------------------------------------------


# -------------------------------------------------------------------------
# A belt push writes no slot the belt does not own
# -------------------------------------------------------------------------


class TestBeltPushDoesNotLeakIntoInactiveSlots:
    """Regression: a belt push must reach only the placed receiver.

    ``run_conveyor_belts`` shares the gather pattern of
    :func:`run_miners`: every inactive slot clips onto tile (0, 0), so a
    belt next to (0, 0) and facing into it must not have its item
    credited to those phantom slots. The crossing-axis receiver track is
    already gated on ``is_crossing`` (active), but the buffer track is
    not, so belts and splitters leak without an explicit active gate.
    """

    def _belt_into_corner_state(self, state_factory) -> EnvState:
        """Build a belt at (0, 1) pushing COAL left into a pallet at (0, 0).

        Parameters
        ----------
        state_factory
            The shared ``state_factory`` fixture.

        Returns
        -------
        EnvState
            A state with two active entities (pallet, belt) and the
            remaining slots inactive at (0, 0). The belt holds one COAL.
        """
        return state_factory(
            world_map=jnp.array([[BlockType.DIRT, BlockType.DIRT]], dtype=jnp.int32),
            machine_types=jnp.array(
                [[Machine.PALLET, Machine.CONVEYOR_BELT]], dtype=jnp.int32
            ),
            machine_direction=jnp.array(
                [[Direction.DOWN, Direction.LEFT]], dtype=jnp.int8
            ),
            buffer_type=jnp.array([[0, int(ItemType.COAL)]], dtype=jnp.int8),
            buffer_count=jnp.array([[0, 1]], dtype=jnp.int16),
        )

    def test_inactive_slots_stay_empty_under_belt_push(self, state_factory) -> None:
        """Inactive slots keep an empty buffer while the pallet fills."""
        state = self._belt_into_corner_state(state_factory)
        params = EnvParams()

        state = run_conveyor_belts(state, params)

        pallet_eid = _eid(state, 0, 0)
        assert state.ent_buf_count[pallet_eid] == 1
        assert state.ent_buf_type[pallet_eid] == ItemType.COAL

        inactive = state.ent_y < 0
        assert bool(jnp.all(state.ent_buf_count[inactive] == 0))
        assert bool(jnp.all(state.ent_buf_type[inactive] == 0))

    def test_belt_push_conserves_items(self, state_factory) -> None:
        """The single COAL on the belt is moved, never duplicated."""
        state = self._belt_into_corner_state(state_factory)
        params = EnvParams()

        before = int(jnp.sum(state.ent_buf_count.astype(jnp.int32)))
        state = run_conveyor_belts(state, params)
        after = int(jnp.sum(state.ent_buf_count.astype(jnp.int32)))
        assert after == before
