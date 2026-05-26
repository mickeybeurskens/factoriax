"""Unit tests for :func:`build_inter_cell_chain` and
:func:`inter_cell_chain_inventory`.

These are pure-Python (no env rollout) and run in milliseconds.
"""

from __future__ import annotations

import pytest

from baselines.rocket.scripted.goals import (
    PlaceMachineAt,
    build_inter_cell_chain,
    inter_cell_chain_inventory,
)
from factoriax.constants import Direction, ItemType, Machine

# ---------------------------------------------------------------------------
# Inventory helper
# ---------------------------------------------------------------------------


def test_inventory_offset_2_is_one_arm_zero_belts() -> None:
    """The minimum cell-chain bridge needs 1 ARM and 0 BELTs."""
    cost = inter_cell_chain_inventory((10, 10), (12, 10))
    assert cost == {int(ItemType.ARM): 1}


def test_inventory_offset_5_is_one_arm_three_belts() -> None:
    """Offset 5 = arm + 3 belts (2 .. 4)."""
    cost = inter_cell_chain_inventory((10, 10), (15, 10))
    assert cost == {int(ItemType.ARM): 1, int(ItemType.CONVEYOR_BELT): 3}


def test_inventory_left_direction() -> None:
    """LEFT direction works symmetrically to RIGHT."""
    cost = inter_cell_chain_inventory((15, 10), (12, 10), direction=int(Direction.LEFT))
    assert cost == {int(ItemType.ARM): 1, int(ItemType.CONVEYOR_BELT): 1}


def test_inventory_down_direction_for_vertical_chain() -> None:
    """DOWN direction works for vertical chains (e.g. tier-4 drops)."""
    cost = inter_cell_chain_inventory((10, 5), (10, 9), direction=int(Direction.DOWN))
    assert cost == {int(ItemType.ARM): 1, int(ItemType.CONVEYOR_BELT): 2}


def test_inventory_offset_1_raises() -> None:
    """Offset 1 leaves no room for the arm — must error."""
    with pytest.raises(ValueError, match="offset 1"):
        inter_cell_chain_inventory((10, 10), (11, 10))


def test_inventory_off_axis_raises() -> None:
    """next_input_tile must be axis-aligned with crate_tile."""
    with pytest.raises(ValueError, match="not on the same row"):
        inter_cell_chain_inventory((10, 10), (15, 11))


def test_inventory_backward_direction_raises() -> None:
    """next_input behind crate_tile fails (would imply arm runs the wrong way)."""
    with pytest.raises(ValueError, match="not forward"):
        inter_cell_chain_inventory((15, 10), (10, 10))  # default RIGHT


def test_inventory_invalid_direction_raises() -> None:
    """Direction must be one of the four cardinals."""
    with pytest.raises(ValueError, match="must be UP/DOWN/LEFT/RIGHT"):
        inter_cell_chain_inventory((10, 10), (12, 10), direction=999)


# ---------------------------------------------------------------------------
# Goal-list builder
# ---------------------------------------------------------------------------


def test_build_offset_2_emits_one_arm_no_belts() -> None:
    """Crate(10,10) -> next_input(12,10): arm at (11,10) RIGHT, no belts."""
    goals = build_inter_cell_chain((10, 10), (12, 10))
    assert len(goals) == 1
    assert isinstance(goals[0], PlaceMachineAt)
    assert goals[0].machine_type == int(Machine.ARM)
    assert goals[0].target == (11, 10)
    assert goals[0].facing == int(Direction.RIGHT)


def test_build_offset_5_emits_arm_and_three_belts_in_order() -> None:
    """Crate(10,10) -> next_input(15,10): arm + 3 belts at 12,13,14."""
    goals = build_inter_cell_chain((10, 10), (15, 10))
    assert len(goals) == 4
    arm, belt0, belt1, belt2 = goals
    assert arm.machine_type == int(Machine.ARM)
    assert arm.target == (11, 10)
    belts = (belt0, belt1, belt2)
    assert all(b.machine_type == int(Machine.CONVEYOR_BELT) for b in belts)
    assert [b.target for b in belts] == [(12, 10), (13, 10), (14, 10)]
    assert all(b.facing == int(Direction.RIGHT) for b in belts)


def test_build_left_direction_mirrors_right() -> None:
    """LEFT direction: arm at crate-1, belts walking west to next_input+1."""
    goals = build_inter_cell_chain((15, 10), (12, 10), direction=int(Direction.LEFT))
    assert len(goals) == 2
    assert goals[0].target == (14, 10)
    assert goals[0].facing == int(Direction.LEFT)
    assert goals[1].target == (13, 10)
    assert goals[1].facing == int(Direction.LEFT)


def test_build_down_direction_for_tier_4_vertical_drop() -> None:
    """DOWN direction: chain runs along a column."""
    goals = build_inter_cell_chain((10, 5), (10, 9), direction=int(Direction.DOWN))
    assert len(goals) == 3
    assert [(g.machine_type, g.target, g.facing) for g in goals] == [
        (int(Machine.ARM), (10, 6), int(Direction.DOWN)),
        (int(Machine.CONVEYOR_BELT), (10, 7), int(Direction.DOWN)),
        (int(Machine.CONVEYOR_BELT), (10, 8), int(Direction.DOWN)),
    ]


def test_build_inventory_matches_emitted_goals() -> None:
    """Sum of emitted PlaceMachineAt by item-type equals the inventory call."""
    crate, sink = (10, 10), (16, 10)
    goals = build_inter_cell_chain(crate, sink)
    cost = inter_cell_chain_inventory(crate, sink)
    counts: dict[int, int] = {}
    for g in goals:
        # Map Machine -> ItemType (1:1 for ARM, CONVEYOR_BELT).
        if g.machine_type == int(Machine.ARM):
            counts[int(ItemType.ARM)] = counts.get(int(ItemType.ARM), 0) + 1
        elif g.machine_type == int(Machine.CONVEYOR_BELT):
            counts[int(ItemType.CONVEYOR_BELT)] = (
                counts.get(int(ItemType.CONVEYOR_BELT), 0) + 1
            )
    assert counts == cost


def test_build_off_axis_raises() -> None:
    """next_input not on the same row/column raises."""
    with pytest.raises(ValueError, match="not on the same row"):
        build_inter_cell_chain((10, 10), (13, 11))


def test_build_offset_1_raises() -> None:
    """Adjacent crate + input has no room for the arm."""
    with pytest.raises(ValueError, match="offset 1"):
        build_inter_cell_chain((10, 10), (11, 10))


def test_build_offset_zero_raises() -> None:
    """Same tile is rejected as 'not forward'."""
    with pytest.raises(ValueError, match="not forward"):
        build_inter_cell_chain((10, 10), (10, 10))


def test_build_with_occupied_tile_in_chain_raises() -> None:
    """Any tile in the arm/belt path must be dirt at place time."""
    with pytest.raises(ValueError, match="collides with an occupied tile"):
        build_inter_cell_chain((10, 10), (15, 10), occupied={(13, 10)})


def test_build_out_of_bounds_belt_raises() -> None:
    """Map-size bounds check fires when a belt would land off-map.

    Crate at (30, 10) with offset 4 puts arm at (31, 10) (in bounds)
    and belts at (32, 10), (33, 10) — both off the 32x32 map.
    """
    with pytest.raises(ValueError, match="out of map bounds"):
        build_inter_cell_chain((30, 10), (34, 10), map_size=(32, 32))


def test_build_out_of_bounds_arm_raises() -> None:
    """Map-size bounds check fires when the arm tile itself is off-map."""
    with pytest.raises(ValueError, match="out of map bounds"):
        build_inter_cell_chain((31, 10), (33, 10), map_size=(32, 32))


# ---------------------------------------------------------------------------
# Realistic gap=5 cell-chain scenario from the M9-M15 plan
# ---------------------------------------------------------------------------


def test_realistic_motor_to_engine_chain() -> None:
    """MOTOR crate (20, 13) -> ENGINE input_b (22, 13): one bridge arm.

    MOTOR cell at (18, 13) RIGHT puts its crate at (20, 13). ENGINE at
    (23, 13) RIGHT puts its input_b at (22, 13). Spacing 5 between
    assemblers means a single arm at (21, 13) bridges them — no
    belts.
    """
    goals = build_inter_cell_chain((20, 13), (22, 13))
    assert len(goals) == 1
    assert goals[0].target == (21, 13)
    assert goals[0].facing == int(Direction.RIGHT)
    assert goals[0].machine_type == int(Machine.ARM)
