"""Tests for conveyor belt and pick-and-place arm machine logic.

Uses the entity-based state model where each machine has a single buffer
slot (ent_buf_type, ent_buf_count) accessed via tile_entity mapping.
Belt max stack is 3, arm has no internal buffer and transfers 1 item per
tick between the entity behind it and the entity in front.
"""

from __future__ import annotations

import jax.numpy as jnp

from factoriax.engine.constants import (
    Direction,
    ItemType,
    Machine,
)
from factoriax.engine.machines import run_arms, run_conveyor_belts
from factoriax.engine.state import EnvParams

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

    Args:
        shape: Grid (H, W).
        entries: Mapping of (y, x) -> (item_type, count).

    Returns:
        Tuple of (buffer_type, buffer_count) arrays, each shape (H, W).
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

    Args:
        state_factory: Pytest fixture that creates EnvState.
        machine_types: Machine type per tile.
        machine_direction: Direction per tile.
        buffer_type: Buffer item type per tile (optional).
        buffer_count: Buffer item count per tile (optional).

    Returns:
        Configured EnvState.
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

    Args:
        state: Environment state.
        y: Tile row.
        x: Tile column.

    Returns:
        Tuple of (item_type, count). Returns (0, 0) if no entity.
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


class TestArm:
    """Pick-and-place arm tests.

    Arms have no internal buffer. Each tick, an arm transfers 1 item from
    the entity behind it to the entity in front (facing direction).
    Layout: [SOURCE, ARM(RIGHT), DESTINATION] means the arm picks from
    SOURCE (behind) and deposits into DESTINATION (in front).
    """

    def test_transfers_from_miner_to_belt(self, state_factory) -> None:
        """Arm transfers 1 item from miner behind to belt in front."""
        # [MINER, ARM(RIGHT), BELT]
        types = jnp.array([[Machine.MINER, Machine.ARM, B]])
        dirs = jnp.array([[_NOOP, R, _NOOP]])
        bt, bc = _buf_grids((1, 3), {(0, 0): (ItemType.COAL, 5)})
        state = _make_state(
            state_factory,
            machine_types=types,
            machine_direction=dirs,
            buffer_type=bt,
            buffer_count=bc,
        )
        result = run_arms(state, _PARAMS)
        belt_type, belt_count = _get_buf(result, 0, 2)
        assert belt_type == int(ItemType.COAL)
        assert belt_count == 1
        _, miner_count = _get_buf(result, 0, 0)
        assert miner_count == 4

    def test_deposits_to_pallet(self, state_factory) -> None:
        """Arm transfers 1 item from belt behind to pallet in front."""
        # [BELT, ARM(RIGHT), PALLET]
        types = jnp.array([[B, Machine.ARM, Machine.PALLET]])
        dirs = jnp.array([[_NOOP, R, _NOOP]])
        bt, bc = _buf_grids((1, 3), {(0, 0): (ItemType.IRON_ORE, 3)})
        state = _make_state(
            state_factory,
            machine_types=types,
            machine_direction=dirs,
            buffer_type=bt,
            buffer_count=bc,
        )
        result = run_arms(state, _PARAMS)
        pallet_type, pallet_count = _get_buf(result, 0, 2)
        assert pallet_type == int(ItemType.IRON_ORE)
        assert pallet_count == 1
        _, belt_count = _get_buf(result, 0, 0)
        assert belt_count == 2

    def test_merges_same_item_into_destination(self, state_factory) -> None:
        """Arm merges item into destination holding the same type."""
        # [MINER, ARM(RIGHT), PALLET with existing COAL]
        types = jnp.array([[Machine.MINER, Machine.ARM, Machine.PALLET]])
        dirs = jnp.array([[_NOOP, R, _NOOP]])
        bt, bc = _buf_grids(
            (1, 3),
            {
                (0, 0): (ItemType.COAL, 5),
                (0, 2): (ItemType.COAL, 10),
            },
        )
        state = _make_state(
            state_factory,
            machine_types=types,
            machine_direction=dirs,
            buffer_type=bt,
            buffer_count=bc,
        )
        result = run_arms(state, _PARAMS)
        pallet_type, pallet_count = _get_buf(result, 0, 2)
        assert pallet_type == int(ItemType.COAL)
        assert pallet_count == 11
        _, miner_count = _get_buf(result, 0, 0)
        assert miner_count == 4

    def test_noop_when_dest_full(self, state_factory) -> None:
        """Arm does nothing when destination is at max stack."""
        # [MINER, ARM(RIGHT), BELT at max]
        types = jnp.array([[Machine.MINER, Machine.ARM, B]])
        dirs = jnp.array([[_NOOP, R, _NOOP]])
        bt, bc = _buf_grids(
            (1, 3),
            {
                (0, 0): (ItemType.COAL, 5),
                (0, 2): (ItemType.COAL, _BELT_MAX),
            },
        )
        state = _make_state(
            state_factory,
            machine_types=types,
            machine_direction=dirs,
            buffer_type=bt,
            buffer_count=bc,
        )
        result = run_arms(state, _PARAMS)
        _, miner_count = _get_buf(result, 0, 0)
        assert miner_count == 5
        _, belt_count = _get_buf(result, 0, 2)
        assert belt_count == _BELT_MAX

    def test_noop_when_source_empty(self, state_factory) -> None:
        """Arm with empty source does nothing."""
        # [empty MINER, ARM(RIGHT), BELT]
        types = jnp.array([[Machine.MINER, Machine.ARM, B]])
        dirs = jnp.array([[_NOOP, R, _NOOP]])
        state = _make_state(
            state_factory,
            machine_types=types,
            machine_direction=dirs,
        )
        result = run_arms(state, _PARAMS)
        assert _get_buf(result, 0, 0) == (0, 0)
        assert _get_buf(result, 0, 2) == (0, 0)

    def test_noop_when_no_dest_entity(self, state_factory) -> None:
        """Arm facing into empty tile does nothing."""
        # [MINER, ARM(RIGHT), NONE]
        types = jnp.array([[Machine.MINER, Machine.ARM, Machine.NONE]])
        dirs = jnp.array([[_NOOP, R, _NOOP]])
        bt, bc = _buf_grids((1, 3), {(0, 0): (ItemType.COAL, 3)})
        state = _make_state(
            state_factory,
            machine_types=types,
            machine_direction=dirs,
            buffer_type=bt,
            buffer_count=bc,
        )
        result = run_arms(state, _PARAMS)
        _, miner_count = _get_buf(result, 0, 0)
        assert miner_count == 3

    def test_does_not_corrupt_unrelated_entities(
        self,
        state_factory,
    ) -> None:
        """Arm transfer does not zero out unrelated entity buffers.

        Reproduces a bug where the pick phase's item-type clearing used
        a gather+where instead of a scatter, causing nearby entity
        buffers to be zeroed in the same tick.
        """
        # [PALLET, ARM(RIGHT), BELT]
        types = jnp.array([[Machine.PALLET, Machine.ARM, B]])
        dirs = jnp.array([[_NOOP, R, _NOOP]])
        bt, bc = _buf_grids(
            (1, 3),
            {(0, 0): (ItemType.COAL, 3)},
        )
        state = _make_state(
            state_factory,
            machine_types=types,
            machine_direction=dirs,
            buffer_type=bt,
            buffer_count=bc,
        )
        result = run_arms(state, _PARAMS)
        belt_type, belt_count = _get_buf(result, 0, 2)
        assert belt_type == int(ItemType.COAL), "belt item corrupted"
        assert belt_count == 1
        pallet_type, pallet_count = _get_buf(result, 0, 0)
        assert pallet_type == int(ItemType.COAL)
        assert pallet_count == 2
