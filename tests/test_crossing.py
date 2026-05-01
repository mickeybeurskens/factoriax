"""Tests for CROSSING tile semantics.

Crossing logic lives inside :func:`run_conveyor_belts` (folded with
belts and splitters into a single 4-iteration scatter-gather pass).
These tests exercise the crossing-specific behaviour:

1. Two independent axis flows: an item entering the vertical axis must
   exit on the vertical output side and never cross into the horizontal
   slot. Stream isolation is the load-bearing property.
2. Wrong-side rejection: a belt pushing onto a crossing's *output* side
   is blocked, so the crossing has well-defined input and output sides.
3. Saturated-chain throughput: a belt → crossing → belt chain runs at
   full single-tile-per-tick throughput. The per-axis cap of 2 gives
   one cell of slack so a same-tick refill (gather +1) and drain
   (scatter -1) leave the slot at its starting count of 1 in steady
   state.
4. Direction encoding: each of the four packed ``ent_direction`` values
   (1..4) maps to the documented (vertical_dir, horizontal_dir) pair.
"""

from __future__ import annotations

import jax.numpy as jnp

from factoriax.belts import CROSSING_HORIZ_SLOT, CROSSING_VERT_SLOT
from factoriax.constants import (
    BlockType,
    Direction,
    ItemType,
    MachineType,
)
from factoriax.machines import run_conveyor_belts
from factoriax.state import EnvParams, EnvState

# Default params — engine kernels read params.machine_config.max_stack
# for buffer caps; the default config matches what these tests already
# implicitly assumed.
_PARAMS = EnvParams()


def _eid(state: EnvState, y: int, x: int) -> int:
    eid = int(state.tile_entity[y, x])
    assert eid >= 0, f"No entity at ({y}, {x})"
    return eid


def _make_crossing_world(
    state_factory,
    *,
    encoding: int,
    vert_item: int = 0,
    vert_count: int = 0,
    horiz_item: int = 0,
    horiz_count: int = 0,
    pallets: tuple[bool, bool, bool, bool] = (True, True, True, True),
) -> EnvState:
    """Build a 3x3 world with a crossing at (1, 1).

    ``encoding`` is the packed ``ent_direction`` (1..4 — see
    :data:`factoriax.belts.CROSSING_AXIS_DIRS`).

    ``pallets`` selects which of the four sides has a receptive pallet:
    ``(up, down, left, right)``. Sides without a pallet are DIRT and so
    non-receptive — useful for testing single-axis flows.

    The crossing's ``ent_asm_in[idx, 0]`` is pre-loaded with
    ``vert_count`` of ``vert_item`` (the vertical-axis slot); ``[idx, 1]``
    with ``horiz_item`` * ``horiz_count`` for the horizontal-axis slot.
    """
    shape = (3, 3)
    world = jnp.full(shape, int(BlockType.DIRT), dtype=jnp.int32)
    mt = jnp.full(shape, int(MachineType.NONE), dtype=jnp.int32)
    md = jnp.zeros(shape, dtype=jnp.int8)
    ait = jnp.zeros((*shape, 2), dtype=jnp.int8)
    aic = jnp.zeros((*shape, 2), dtype=jnp.int16)

    mt = mt.at[1, 1].set(int(MachineType.CROSSING))
    md = md.at[1, 1].set(encoding)
    ait = ait.at[1, 1, CROSSING_VERT_SLOT].set(vert_item)
    aic = aic.at[1, 1, CROSSING_VERT_SLOT].set(vert_count)
    ait = ait.at[1, 1, CROSSING_HORIZ_SLOT].set(horiz_item)
    aic = aic.at[1, 1, CROSSING_HORIZ_SLOT].set(horiz_count)

    up_p, down_p, left_p, right_p = pallets
    if up_p:
        mt = mt.at[0, 1].set(int(MachineType.PALLET))
    if down_p:
        mt = mt.at[2, 1].set(int(MachineType.PALLET))
    if left_p:
        mt = mt.at[1, 0].set(int(MachineType.PALLET))
    if right_p:
        mt = mt.at[1, 2].set(int(MachineType.PALLET))

    return state_factory(
        world_map=world,
        machine_types=mt,
        machine_direction=md,
        asm_in_type=ait,
        asm_in_count=aic,
    )


# ---------------------------------------------------------------------------
# Output: each axis pushes from its slot to the encoded direction.
# ---------------------------------------------------------------------------


def test_vertical_axis_pushes_down_for_encoding_1(state_factory) -> None:
    """encoding=1 → vert_dir=DOWN. The vertical slot should drain to
    the S pallet, leaving the horizontal axis untouched."""
    state = _make_crossing_world(
        state_factory,
        encoding=1,
        vert_item=int(ItemType.IRON_PLATE),
        vert_count=1,
    )
    out = run_conveyor_belts(state, _PARAMS)
    cid = _eid(out, 1, 1)
    south_eid = _eid(out, 2, 1)

    # Vertical slot drained.
    assert int(out.ent_asm_in_count[cid, CROSSING_VERT_SLOT]) == 0
    # Horizontal slot still empty (it was empty to begin with).
    assert int(out.ent_asm_in_count[cid, CROSSING_HORIZ_SLOT]) == 0
    # South pallet received the iron plate.
    assert int(out.ent_buf_count[south_eid]) == 1
    assert int(out.ent_buf_type[south_eid]) == int(ItemType.IRON_PLATE)


def test_horizontal_axis_pushes_right_for_encoding_1(state_factory) -> None:
    """encoding=1 → horiz_dir=RIGHT. The horizontal slot should drain
    to the E pallet, leaving the vertical axis untouched."""
    state = _make_crossing_world(
        state_factory,
        encoding=1,
        horiz_item=int(ItemType.COPPER_PLATE),
        horiz_count=1,
    )
    out = run_conveyor_belts(state, _PARAMS)
    cid = _eid(out, 1, 1)
    east_eid = _eid(out, 1, 2)
    assert int(out.ent_asm_in_count[cid, CROSSING_HORIZ_SLOT]) == 0
    assert int(out.ent_asm_in_count[cid, CROSSING_VERT_SLOT]) == 0
    assert int(out.ent_buf_count[east_eid]) == 1
    assert int(out.ent_buf_type[east_eid]) == int(ItemType.COPPER_PLATE)


def test_encoding_2_horiz_left(state_factory) -> None:
    """encoding=2 → vert_dir=DOWN, horiz_dir=LEFT."""
    state = _make_crossing_world(
        state_factory,
        encoding=2,
        horiz_item=int(ItemType.COPPER_PLATE),
        horiz_count=1,
    )
    out = run_conveyor_belts(state, _PARAMS)
    west_eid = _eid(out, 1, 0)
    assert int(out.ent_buf_count[west_eid]) == 1
    assert int(out.ent_buf_type[west_eid]) == int(ItemType.COPPER_PLATE)


def test_encoding_3_vert_up(state_factory) -> None:
    """encoding=3 → vert_dir=UP, horiz_dir=RIGHT."""
    state = _make_crossing_world(
        state_factory,
        encoding=3,
        vert_item=int(ItemType.IRON_PLATE),
        vert_count=1,
    )
    out = run_conveyor_belts(state, _PARAMS)
    north_eid = _eid(out, 0, 1)
    assert int(out.ent_buf_count[north_eid]) == 1
    assert int(out.ent_buf_type[north_eid]) == int(ItemType.IRON_PLATE)


def test_encoding_4_full_dual_drain(state_factory) -> None:
    """encoding=4 → vert_dir=UP, horiz_dir=LEFT. Both slots populated;
    both drain in the same tick to opposite sides."""
    state = _make_crossing_world(
        state_factory,
        encoding=4,
        vert_item=int(ItemType.IRON_PLATE),
        vert_count=1,
        horiz_item=int(ItemType.COPPER_PLATE),
        horiz_count=1,
    )
    out = run_conveyor_belts(state, _PARAMS)
    cid = _eid(out, 1, 1)
    assert int(out.ent_asm_in_count[cid, CROSSING_VERT_SLOT]) == 0
    assert int(out.ent_asm_in_count[cid, CROSSING_HORIZ_SLOT]) == 0
    assert int(out.ent_buf_count[_eid(out, 0, 1)]) == 1  # N receives Fe
    assert int(out.ent_buf_type[_eid(out, 0, 1)]) == int(ItemType.IRON_PLATE)
    assert int(out.ent_buf_count[_eid(out, 1, 0)]) == 1  # W receives Cu
    assert int(out.ent_buf_type[_eid(out, 1, 0)]) == int(ItemType.COPPER_PLATE)


# ---------------------------------------------------------------------------
# Stream isolation: vertical and horizontal slots never mix.
# ---------------------------------------------------------------------------


def test_streams_do_not_mix(state_factory) -> None:
    """A crossing with iron in vert and copper in horiz must drain
    each item to its own axis output. Cross-axis leakage would mean
    the wrong type appears at one of the destination pallets."""
    state = _make_crossing_world(
        state_factory,
        encoding=1,  # vert→S, horiz→E
        vert_item=int(ItemType.IRON_PLATE),
        vert_count=1,
        horiz_item=int(ItemType.COPPER_PLATE),
        horiz_count=1,
    )
    out = run_conveyor_belts(state, _PARAMS)
    south_eid = _eid(out, 2, 1)
    east_eid = _eid(out, 1, 2)
    # South gets exactly the vertical-axis item (iron), not copper.
    assert int(out.ent_buf_type[south_eid]) == int(ItemType.IRON_PLATE)
    # East gets exactly the horizontal-axis item (copper), not iron.
    assert int(out.ent_buf_type[east_eid]) == int(ItemType.COPPER_PLATE)


# ---------------------------------------------------------------------------
# Wrong-side rejection.
# ---------------------------------------------------------------------------


def test_belt_pushing_into_output_side_is_rejected(state_factory) -> None:
    """A belt pushing UP onto a crossing whose vert flow is N→S
    (encoding=1, vert_dir=DOWN) is hitting the OUTPUT side. The push
    must fail — the belt's item stays put, the crossing slot stays
    empty."""
    shape = (3, 3)
    world = jnp.full(shape, int(BlockType.DIRT), dtype=jnp.int32)
    mt = jnp.full(shape, int(MachineType.NONE), dtype=jnp.int32)
    md = jnp.zeros(shape, dtype=jnp.int8)
    bt = jnp.zeros(shape, dtype=jnp.int8)
    bc = jnp.zeros(shape, dtype=jnp.int16)
    ait = jnp.zeros((*shape, 2), dtype=jnp.int8)
    aic = jnp.zeros((*shape, 2), dtype=jnp.int16)

    # Crossing at (1, 1), encoding=1: vert_dir=DOWN (S is output side).
    mt = mt.at[1, 1].set(int(MachineType.CROSSING))
    md = md.at[1, 1].set(1)

    # Belt at (2, 1) pushing UP (toward the crossing's S/output side).
    mt = mt.at[2, 1].set(int(MachineType.CONVEYOR_BELT))
    md = md.at[2, 1].set(int(Direction.UP))
    bt = bt.at[2, 1].set(int(ItemType.IRON_PLATE))
    bc = bc.at[2, 1].set(1)

    state = state_factory(
        world_map=world,
        machine_types=mt,
        machine_direction=md,
        buffer_type=bt,
        buffer_count=bc,
        asm_in_type=ait,
        asm_in_count=aic,
    )
    out = run_conveyor_belts(state, _PARAMS)
    cid = _eid(out, 1, 1)
    belt_eid = _eid(out, 2, 1)
    assert int(out.ent_asm_in_count[cid, CROSSING_VERT_SLOT]) == 0
    assert int(out.ent_buf_count[belt_eid]) == 1  # belt held its item
    assert int(out.ent_buf_type[belt_eid]) == int(ItemType.IRON_PLATE)


def test_belt_pushing_into_correct_input_side_lands_in_axis_slot(
    state_factory,
) -> None:
    """Mirror of the previous test: a belt pushing DOWN from N onto a
    crossing with vert_dir=DOWN is hitting the correct INPUT side and
    its item lands in the crossing's vertical slot."""
    shape = (3, 3)
    world = jnp.full(shape, int(BlockType.DIRT), dtype=jnp.int32)
    mt = jnp.full(shape, int(MachineType.NONE), dtype=jnp.int32)
    md = jnp.zeros(shape, dtype=jnp.int8)
    bt = jnp.zeros(shape, dtype=jnp.int8)
    bc = jnp.zeros(shape, dtype=jnp.int16)
    ait = jnp.zeros((*shape, 2), dtype=jnp.int8)
    aic = jnp.zeros((*shape, 2), dtype=jnp.int16)

    mt = mt.at[1, 1].set(int(MachineType.CROSSING))
    md = md.at[1, 1].set(1)

    # Belt at (0, 1) pushing DOWN — input side N for vert_dir=DOWN.
    mt = mt.at[0, 1].set(int(MachineType.CONVEYOR_BELT))
    md = md.at[0, 1].set(int(Direction.DOWN))
    bt = bt.at[0, 1].set(int(ItemType.IRON_PLATE))
    bc = bc.at[0, 1].set(1)

    state = state_factory(
        world_map=world,
        machine_types=mt,
        machine_direction=md,
        buffer_type=bt,
        buffer_count=bc,
        asm_in_type=ait,
        asm_in_count=aic,
    )
    out = run_conveyor_belts(state, _PARAMS)
    cid = _eid(out, 1, 1)
    belt_eid = _eid(out, 0, 1)
    assert int(out.ent_asm_in_count[cid, CROSSING_VERT_SLOT]) == 1
    assert int(out.ent_asm_in_type[cid, CROSSING_VERT_SLOT]) == int(ItemType.IRON_PLATE)
    assert int(out.ent_buf_count[belt_eid]) == 0  # belt drained


# ---------------------------------------------------------------------------
# Saturated-chain throughput — full one-tile-per-tick flow.
# ---------------------------------------------------------------------------


def test_belt_to_crossing_to_belt_chain_full_throughput(state_factory) -> None:
    """A pre-loaded belt feeding a vertically-flowing crossing whose
    output side has a receptive belt should advance items by *one
    tile per tick*. The crossing's per-axis cap of 2 is what enables
    this: gather adds the upstream item to the slot (1 → 2) before
    scatter removes the outgoing one (2 → 1), so the slot is never
    seen as ``full`` mid-pass."""
    shape = (4, 1)
    world = jnp.full(shape, int(BlockType.DIRT), dtype=jnp.int32)
    mt = jnp.full(shape, int(MachineType.NONE), dtype=jnp.int32)
    md = jnp.zeros(shape, dtype=jnp.int8)
    bt = jnp.zeros(shape, dtype=jnp.int8)
    bc = jnp.zeros(shape, dtype=jnp.int16)
    ait = jnp.zeros((*shape, 2), dtype=jnp.int8)
    aic = jnp.zeros((*shape, 2), dtype=jnp.int16)

    # (0, 0) belt pushing DOWN with iron, (1, 0) crossing vert_dir=DOWN
    # with slot 0 pre-filled, (2, 0) belt pushing DOWN, (3, 0) pallet
    # as a sink so the output-side belt has somewhere to drain.
    mt = mt.at[0, 0].set(int(MachineType.CONVEYOR_BELT))
    md = md.at[0, 0].set(int(Direction.DOWN))
    bt = bt.at[0, 0].set(int(ItemType.IRON_PLATE))
    bc = bc.at[0, 0].set(1)

    mt = mt.at[1, 0].set(int(MachineType.CROSSING))
    md = md.at[1, 0].set(1)  # vert_dir=DOWN, horiz_dir=RIGHT
    ait = ait.at[1, 0, CROSSING_VERT_SLOT].set(int(ItemType.IRON_PLATE))
    aic = aic.at[1, 0, CROSSING_VERT_SLOT].set(1)

    mt = mt.at[2, 0].set(int(MachineType.CONVEYOR_BELT))
    md = md.at[2, 0].set(int(Direction.DOWN))

    mt = mt.at[3, 0].set(int(MachineType.PALLET))

    state = state_factory(
        world_map=world,
        machine_types=mt,
        machine_direction=md,
        buffer_type=bt,
        buffer_count=bc,
        asm_in_type=ait,
        asm_in_count=aic,
    )
    out = run_conveyor_belts(state, _PARAMS)

    upstream_belt = _eid(out, 0, 0)
    crossing = _eid(out, 1, 0)
    downstream_belt = _eid(out, 2, 0)

    # Upstream belt drained: its iron flowed into the crossing's slot.
    assert int(out.ent_buf_count[upstream_belt]) == 0
    # Crossing slot still has 1 (drained one to downstream, received
    # one from upstream — the will-drain optimisation enables the
    # in-flight slot reuse).
    assert int(out.ent_asm_in_count[crossing, CROSSING_VERT_SLOT]) == 1
    # Downstream belt now holds the iron the crossing pushed out.
    assert int(out.ent_buf_count[downstream_belt]) == 1
    assert int(out.ent_buf_type[downstream_belt]) == int(ItemType.IRON_PLATE)


# ---------------------------------------------------------------------------
# Empty / inactive crossing.
# ---------------------------------------------------------------------------


def test_crossing_with_empty_slots_is_no_op(state_factory) -> None:
    state = _make_crossing_world(state_factory, encoding=1)
    out = run_conveyor_belts(state, _PARAMS)
    cid = _eid(out, 1, 1)
    assert int(out.ent_asm_in_count[cid, CROSSING_VERT_SLOT]) == 0
    assert int(out.ent_asm_in_count[cid, CROSSING_HORIZ_SLOT]) == 0
    # All four pallets stay empty.
    for tile in [(0, 1), (2, 1), (1, 0), (1, 2)]:
        assert int(out.ent_buf_count[_eid(out, *tile)]) == 0


def test_inactive_direction_is_no_op(state_factory) -> None:
    """A crossing with ent_direction=0 decodes to (0, 0) outputs in
    CROSSING_AXIS_DIRS — neither axis has a valid direction so no
    pushes should fire even with full slots."""
    state = _make_crossing_world(
        state_factory,
        encoding=0,
        vert_item=int(ItemType.IRON_PLATE),
        vert_count=1,
        horiz_item=int(ItemType.COPPER_PLATE),
        horiz_count=1,
    )
    out = run_conveyor_belts(state, _PARAMS)
    cid = _eid(out, 1, 1)
    assert int(out.ent_asm_in_count[cid, CROSSING_VERT_SLOT]) == 1
    assert int(out.ent_asm_in_count[cid, CROSSING_HORIZ_SLOT]) == 1
    # No pallet receives anything.
    for tile in [(0, 1), (2, 1), (1, 0), (1, 2)]:
        assert int(out.ent_buf_count[_eid(out, *tile)]) == 0


# ---------------------------------------------------------------------------
# Backpressure on output side.
# ---------------------------------------------------------------------------


def test_blocked_output_holds_axis_item(state_factory) -> None:
    """If the vertical output side has no receiver (DIRT, no entity),
    the vertical slot's item stays put. The horizontal axis is
    unaffected and should still drain."""
    state = _make_crossing_world(
        state_factory,
        encoding=1,  # vert→S, horiz→E
        vert_item=int(ItemType.IRON_PLATE),
        vert_count=1,
        horiz_item=int(ItemType.COPPER_PLATE),
        horiz_count=1,
        pallets=(True, False, True, True),  # no S pallet
    )
    out = run_conveyor_belts(state, _PARAMS)
    cid = _eid(out, 1, 1)
    # Vert slot held its item.
    assert int(out.ent_asm_in_count[cid, CROSSING_VERT_SLOT]) == 1
    # Horiz still drained.
    assert int(out.ent_asm_in_count[cid, CROSSING_HORIZ_SLOT]) == 0
    east_eid = _eid(out, 1, 2)
    assert int(out.ent_buf_count[east_eid]) == 1
    assert int(out.ent_buf_type[east_eid]) == int(ItemType.COPPER_PLATE)
