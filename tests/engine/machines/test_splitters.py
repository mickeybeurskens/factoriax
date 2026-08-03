"""Tests for SPLITTER push semantics.

Splitter logic lives inside :func:`run_conveyor_belts` (folded from
the original standalone ``run_splitters`` pass for performance. The
4-iteration scatter-gather is shared with belts). These tests target
the splitter behaviour specifically by populating only splitter +
pallet entities. Belts can run alongside in the same pass without
affecting these assertions.

Splitter semantics under test:

1. Buffer of 2 with both perpendicular outputs receptive → fires one
   item to each, source drains to zero. Even-split is the load-bearing
   property under symmetric flow.
2. Buffer of <2 → never fires (pair-firing gating: the splitter only
   dispatches once it holds a pair, even if exactly one side is
   receptive).
3. Buffer of 2 with exactly one side blocked → fires the receptive
   side (consumes 1, leaves 1 in buffer). The blocked side does not
   choke the other.
4. Both outputs blocked → hold (no neighbour can accept).
5. Inactive direction (NONE) → no-op, regardless of buffer state.
6. Composition with belts upstream → buffer fills over time, fires
   on the tick it reaches stack=2.
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

# Splitter behaviour is exercised through the merged belt-network pass.
run_splitters = run_conveyor_belts

# Default params for run_conveyor_belts (splitter) calls.
_PARAMS = EnvParams()


def _eid(state: EnvState, y: int, x: int) -> int:
    eid = int(state.tile_entity[y, x])
    assert eid >= 0, f"No entity at ({y}, {x})"
    return eid


def _make_splitter_world(
    state_factory,
    *,
    facing: int,
    buf_count: int,
    left_pallet: bool = True,
    right_pallet: bool = True,
    up_pallet: bool = True,
    down_pallet: bool = True,
    item: int = int(ItemType.IRON_PLATE),
) -> EnvState:
    """Build a 3x3 world with a splitter at (1, 1) and optional
    pallets on each of the four sides.

    The pallets are receptive sinks for splitter outputs (large stack
    capacity, accepts any matching type). Setting a side to ``False``
    leaves the tile as DIRT, so the splitter has no neighbour there
    and that side is non-receptive.
    """
    shape = (3, 3)
    world = jnp.full(shape, int(BlockType.DIRT), dtype=jnp.int32)
    mt = jnp.full(shape, int(Machine.NONE), dtype=jnp.int32)
    md = jnp.zeros(shape, dtype=jnp.int8)
    bt = jnp.zeros(shape, dtype=jnp.int8)
    bc = jnp.zeros(shape, dtype=jnp.int16)

    mt = mt.at[1, 1].set(int(Machine.SPLITTER))
    md = md.at[1, 1].set(facing)
    bt = bt.at[1, 1].set(item)
    bc = bc.at[1, 1].set(buf_count)

    if up_pallet:
        mt = mt.at[0, 1].set(int(Machine.PALLET))
    if down_pallet:
        mt = mt.at[2, 1].set(int(Machine.PALLET))
    if left_pallet:
        mt = mt.at[1, 0].set(int(Machine.PALLET))
    if right_pallet:
        mt = mt.at[1, 2].set(int(Machine.PALLET))

    return state_factory(
        world_map=world,
        machine_types=mt,
        machine_direction=md,
        buffer_type=bt,
        buffer_count=bc,
    )


# ---------------------------------------------------------------------------
# Even-split behaviour (the primary contract)
# ---------------------------------------------------------------------------


def test_facing_up_with_full_buffer_fires_one_to_each_horizontal_output(
    state_factory,
) -> None:
    """Vertically-facing splitter (UP) with buf=2 sends 1 to LEFT and
    1 to RIGHT pallets simultaneously."""
    state = _make_splitter_world(state_factory, facing=int(Direction.UP), buf_count=2)
    out = run_splitters(state, _PARAMS)
    seid = _eid(out, 1, 1)
    left_eid = _eid(out, 1, 0)
    right_eid = _eid(out, 1, 2)

    assert int(out.ent_buf_count[seid]) == 0
    assert int(out.ent_buf_count[left_eid]) == 1
    assert int(out.ent_buf_count[right_eid]) == 1
    assert int(out.ent_buf_type[left_eid]) == int(ItemType.IRON_PLATE)
    assert int(out.ent_buf_type[right_eid]) == int(ItemType.IRON_PLATE)


def test_facing_down_outputs_match_facing_up(state_factory) -> None:
    """DOWN-facing splitters output to LEFT/RIGHT (same axis pair as
    UP-facing); only the visual orientation differs from the engine's
    point of view."""
    state = _make_splitter_world(state_factory, facing=int(Direction.DOWN), buf_count=2)
    out = run_splitters(state, _PARAMS)
    assert int(out.ent_buf_count[_eid(out, 1, 0)]) == 1
    assert int(out.ent_buf_count[_eid(out, 1, 2)]) == 1


def test_facing_left_fires_one_up_and_one_down(state_factory) -> None:
    """LEFT-facing splitters output to UP/DOWN (vertical axis pair)."""
    state = _make_splitter_world(state_factory, facing=int(Direction.LEFT), buf_count=2)
    out = run_splitters(state, _PARAMS)
    up_eid = _eid(out, 0, 1)
    down_eid = _eid(out, 2, 1)
    assert int(out.ent_buf_count[_eid(out, 1, 1)]) == 0
    assert int(out.ent_buf_count[up_eid]) == 1
    assert int(out.ent_buf_count[down_eid]) == 1


def test_facing_right_outputs_match_facing_left(state_factory) -> None:
    state = _make_splitter_world(
        state_factory, facing=int(Direction.RIGHT), buf_count=2
    )
    out = run_splitters(state, _PARAMS)
    assert int(out.ent_buf_count[_eid(out, 0, 1)]) == 1
    assert int(out.ent_buf_count[_eid(out, 2, 1)]) == 1


# ---------------------------------------------------------------------------
# Stack-of-2 gating
# ---------------------------------------------------------------------------


def test_buffer_of_one_does_not_fire(state_factory) -> None:
    """The splitter holds until its buffer reaches 2, so a single item
    must remain in place. Without this, the user's even-split contract
    is silently violated for low-throughput streams."""
    state = _make_splitter_world(state_factory, facing=int(Direction.UP), buf_count=1)
    out = run_splitters(state, _PARAMS)
    assert int(out.ent_buf_count[_eid(out, 1, 1)]) == 1
    assert int(out.ent_buf_count[_eid(out, 1, 0)]) == 0
    assert int(out.ent_buf_count[_eid(out, 1, 2)]) == 0


def test_empty_buffer_is_a_no_op(state_factory) -> None:
    """A buf=0 splitter must not touch the state of any entity."""
    state = _make_splitter_world(state_factory, facing=int(Direction.UP), buf_count=0)
    out = run_splitters(state, _PARAMS)
    assert int(out.ent_buf_count[_eid(out, 1, 1)]) == 0
    assert int(out.ent_buf_count[_eid(out, 1, 0)]) == 0
    assert int(out.ent_buf_count[_eid(out, 1, 2)]) == 0


# ---------------------------------------------------------------------------
# Atomic both-or-nothing back-pressure
# ---------------------------------------------------------------------------


def test_blocked_left_output_fires_right_only(state_factory) -> None:
    """If the LEFT output has no neighbour (DIRT) but RIGHT is
    receptive, the splitter fires RIGHT only. It consumes 1 from the
    buffer and leaves 1 in place. The blocked side cannot stall the
    receptive side."""
    state = _make_splitter_world(
        state_factory,
        facing=int(Direction.UP),
        buf_count=2,
        left_pallet=False,  # no entity on LEFT, so non-receptive
    )
    out = run_splitters(state, _PARAMS)
    assert int(out.ent_buf_count[_eid(out, 1, 1)]) == 1
    # RIGHT pallet got the single fired item.
    assert int(out.ent_buf_count[_eid(out, 1, 2)]) == 1


def test_blocked_right_output_fires_left_only(state_factory) -> None:
    """Mirror of the LEFT-blocked case."""
    state = _make_splitter_world(
        state_factory,
        facing=int(Direction.UP),
        buf_count=2,
        right_pallet=False,
    )
    out = run_splitters(state, _PARAMS)
    assert int(out.ent_buf_count[_eid(out, 1, 1)]) == 1
    assert int(out.ent_buf_count[_eid(out, 1, 0)]) == 1


def test_both_outputs_blocked_holds_both(state_factory) -> None:
    """Neither side has a receptive neighbour, so the splitter holds
    its pair in place. The outcome is the same as the old atomic hold.
    The other-side-blocked case is what changed, not this one."""
    state = _make_splitter_world(
        state_factory,
        facing=int(Direction.UP),
        buf_count=2,
        left_pallet=False,
        right_pallet=False,
    )
    out = run_splitters(state, _PARAMS)
    assert int(out.ent_buf_count[_eid(out, 1, 1)]) == 2


# ---------------------------------------------------------------------------
# Type guarding
# ---------------------------------------------------------------------------


def test_inactive_direction_is_no_op(state_factory) -> None:
    """A splitter with ``ent_direction == 0`` (NONE) must not fire.
    The decode table returns (0, 0) for that row. A regression here
    makes every NONE-direction splitter spam fictional pushes."""
    state = _make_splitter_world(state_factory, facing=0, buf_count=2)
    out = run_splitters(state, _PARAMS)
    assert int(out.ent_buf_count[_eid(out, 1, 1)]) == 2
    assert int(out.ent_buf_count[_eid(out, 1, 0)]) == 0
    assert int(out.ent_buf_count[_eid(out, 1, 2)]) == 0


def test_non_splitter_entities_untouched(state_factory) -> None:
    """A pallet at (0, 1) carrying 5 of an item must still hold 5
    afterward. run_splitters must not affect non-splitter entities
    in any way."""
    shape = (3, 3)
    world = jnp.full(shape, int(BlockType.DIRT), dtype=jnp.int32)
    mt = jnp.full(shape, int(Machine.NONE), dtype=jnp.int32)
    bt = jnp.zeros(shape, dtype=jnp.int8)
    bc = jnp.zeros(shape, dtype=jnp.int16)
    mt = mt.at[0, 1].set(int(Machine.PALLET))
    bt = bt.at[0, 1].set(int(ItemType.IRON_PLATE))
    bc = bc.at[0, 1].set(5)
    state = state_factory(
        world_map=world, machine_types=mt, buffer_type=bt, buffer_count=bc
    )
    out = run_splitters(state, _PARAMS)
    assert int(out.ent_buf_count[_eid(out, 0, 1)]) == 5
    assert int(out.ent_buf_type[_eid(out, 0, 1)]) == int(ItemType.IRON_PLATE)


# ---------------------------------------------------------------------------
# Multi-tick composition
# ---------------------------------------------------------------------------


def test_two_ticks_drains_then_holds(state_factory) -> None:
    """Tick 1 drains the buf-of-2 to zero. Tick 2 finds buf=0 and
    nothing more arrives, so the splitter no-ops."""
    state = _make_splitter_world(state_factory, facing=int(Direction.UP), buf_count=2)
    after1 = run_splitters(state, _PARAMS)
    after2 = run_splitters(after1, _PARAMS)
    assert int(after1.ent_buf_count[_eid(after1, 1, 1)]) == 0
    assert int(after2.ent_buf_count[_eid(after2, 1, 1)]) == 0
    # Output pallets keep their items across the second tick.
    assert int(after2.ent_buf_count[_eid(after2, 1, 0)]) == 1
    assert int(after2.ent_buf_count[_eid(after2, 1, 2)]) == 1


def test_left_pallet_wrong_type_fires_right_only(state_factory) -> None:
    """LEFT pallet is full of a *different* item type (non-receptive
    via the same-type-or-empty rule); RIGHT pallet is empty
    (receptive). The splitter fires RIGHT only and consumes 1."""
    shape = (3, 3)
    world = jnp.full(shape, int(BlockType.DIRT), dtype=jnp.int32)
    mt = jnp.full(shape, int(Machine.NONE), dtype=jnp.int32)
    md = jnp.zeros(shape, dtype=jnp.int8)
    bt = jnp.zeros(shape, dtype=jnp.int8)
    bc = jnp.zeros(shape, dtype=jnp.int16)

    mt = mt.at[1, 1].set(int(Machine.SPLITTER))
    md = md.at[1, 1].set(int(Direction.UP))
    bt = bt.at[1, 1].set(int(ItemType.IRON_PLATE))
    bc = bc.at[1, 1].set(2)

    # LEFT pallet holds COPPER_PLATE (different type, blocks merge).
    mt = mt.at[1, 0].set(int(Machine.PALLET))
    bt = bt.at[1, 0].set(int(ItemType.COPPER_PLATE))
    bc = bc.at[1, 0].set(50)
    # RIGHT pallet receptive (empty).
    mt = mt.at[1, 2].set(int(Machine.PALLET))

    state = state_factory(
        world_map=world,
        machine_types=mt,
        machine_direction=md,
        buffer_type=bt,
        buffer_count=bc,
    )
    out = run_splitters(state, _PARAMS)
    assert int(out.ent_buf_count[_eid(out, 1, 1)]) == 1
    # RIGHT pallet got the single fired item.
    assert int(out.ent_buf_count[_eid(out, 1, 2)]) == 1
    # LEFT pallet still holds its 50 COPPER. The splitter never pushed.
    assert int(out.ent_buf_count[_eid(out, 1, 0)]) == 50


def test_buffer_of_one_with_one_side_blocked_holds(state_factory) -> None:
    """Pair-firing gating: a single item never fires, even if the only
    receptive side is the one that receives it. The splitter waits
    for upstream to deliver a second item before it dispatches."""
    state = _make_splitter_world(
        state_factory,
        facing=int(Direction.UP),
        buf_count=1,
        left_pallet=False,  # only RIGHT is receptive
    )
    out = run_splitters(state, _PARAMS)
    assert int(out.ent_buf_count[_eid(out, 1, 1)]) == 1
    assert int(out.ent_buf_count[_eid(out, 1, 2)]) == 0
