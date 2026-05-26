"""Unit tests for :func:`build_assembler_module_at` and
:func:`assembler_module_inventory`."""

from __future__ import annotations

import pytest

from baselines.rocket.scripted.goals import (
    PlaceMachineAt,
    assembler_module_inventory,
    build_assembler_module_at,
)
from factoriax.engine.constants import Direction, ItemType, Machine


def _placements(goals: list) -> list[tuple[int, tuple[int, int], int]]:
    out = []
    for goal in goals:
        assert isinstance(goal, PlaceMachineAt)
        out.append((goal.machine_type, goal.target, goal.facing))
    return out


def test_inventory_2_input_one_pallet_two_belts_one_arm_one_assembler() -> None:
    """Inputs are feeder belts (north + west); output is a pallet."""
    cost = assembler_module_inventory(input_b=True)
    assert cost[int(ItemType.ASSEMBLER)] == 1
    assert cost[int(ItemType.CONVEYOR_BELT)] == 2  # input_a + input_b feeders
    assert cost[int(ItemType.PALLET)] == 1  # output only
    assert cost[int(ItemType.ARM)] == 1


def test_inventory_1_input_drops_one_belt() -> None:
    cost = assembler_module_inventory(input_b=False)
    assert cost[int(ItemType.CONVEYOR_BELT)] == 1  # input_a feeder only
    assert cost[int(ItemType.PALLET)] == 1  # output


def test_2_input_layout_around_center() -> None:
    """Standard layout: input_a north feeder belt, input_b west
    feeder belt, arm east, output east-east."""
    goals = build_assembler_module_at((16, 9))
    placements = _placements(goals)
    assert placements == [
        # output east-east, facing DOWN
        (int(Machine.PALLET), (18, 9), int(Direction.DOWN)),
        # input_b west feeder belt, facing RIGHT (pushes east into
        # the assembler; Phase 0 directional pull picks it up).
        (int(Machine.CONVEYOR_BELT), (15, 9), int(Direction.RIGHT)),
        # arm east, facing RIGHT
        (int(Machine.ARM), (17, 9), int(Direction.RIGHT)),
        # center assembler, facing DOWN
        (int(Machine.ASSEMBLER), (16, 9), int(Direction.DOWN)),
        # input_a north feeder belt, facing DOWN — last because its
        # stand tile (one further north) is unaffected by anything
        # else.
        (int(Machine.CONVEYOR_BELT), (16, 8), int(Direction.DOWN)),
    ]


def test_1_input_skips_input_b() -> None:
    goals = build_assembler_module_at((16, 9), input_b=False)
    placements = _placements(goals)
    # 4 placements: output, arm, center, input_a (no input_b).
    assert len(placements) == 4
    targets = [p[1] for p in placements]
    assert (15, 9) not in targets  # input_b skipped


def test_arm_precedes_center() -> None:
    """Arm must be placed before center because the arm's stand tile
    (one west of arm = center) needs to be walkable dirt."""
    goals = build_assembler_module_at((16, 9))
    types = [g.machine_type for g in goals]
    arm_idx = types.index(int(Machine.ARM))
    center_idx = types.index(int(Machine.ASSEMBLER))
    assert arm_idx < center_idx


def test_center_precedes_input_a() -> None:
    """Input_a is placed last; the center's stand tile (one north of
    center = input_a's eventual location) must be dirt at center-
    place time."""
    goals = build_assembler_module_at((16, 9))
    targets = [g.target for g in goals]
    center_idx = targets.index((16, 9))
    input_a_idx = targets.index((16, 8))
    assert center_idx < input_a_idx


def test_collision_with_occupied_raises_with_label() -> None:
    with pytest.raises(ValueError, match=r"output tile \(18, 9\)"):
        build_assembler_module_at((16, 9), occupied={(18, 9)})


def test_out_of_bounds_input_a_raises() -> None:
    """Center at y=0 puts input_a at y=-1 → out of bounds."""
    with pytest.raises(ValueError, match="input_a tile"):
        build_assembler_module_at((16, 0), map_size=(32, 32))


def test_overlapping_with_existing_module_via_occupied() -> None:
    """Two modules placed too close share a tile and the helper
    rejects the second one with the offending role labelled."""
    module_a = build_assembler_module_at((16, 9))
    a_tiles = {g.target for g in module_a}
    # Module B at (19, 9): input_b at (18, 9) collides with A's output.
    with pytest.raises(ValueError, match=r"input_b tile \(18, 9\)"):
        build_assembler_module_at((19, 9), occupied=a_tiles)
