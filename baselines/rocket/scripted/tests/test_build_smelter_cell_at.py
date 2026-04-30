"""Unit tests for :func:`build_smelter_cell_at` and
:func:`smelter_cell_inventory`.

Pure-Python — no env rollout — so the suite runs in milliseconds and
catches layout / collision / inventory errors before the integration
tests pay the ~30s rollout cost.
"""

from __future__ import annotations

import pytest

from baselines.rocket.scripted.goals import (
    PlaceMachineAt,
    build_smelter_cell_at,
    smelter_cell_inventory,
)
from factoriax.constants import Direction, ItemType, MachineType


def _placements(goals: list) -> list[tuple[int, tuple[int, int], int]]:
    """Pull (machine_type, target, facing) tuples out of a goal list."""
    out = []
    for goal in goals:
        assert isinstance(goal, PlaceMachineAt)
        out.append((goal.machine_type, goal.target, goal.facing))
    return out


def test_inventory_costs_two_pallets_one_arm_one_furnace() -> None:
    """One smelter cell consumes 2 PALLET + 1 ARM + 1 FURNACE."""
    cost = smelter_cell_inventory()
    assert cost[int(ItemType.PALLET)] == 2
    assert cost[int(ItemType.ARM)] == 1
    assert cost[int(ItemType.FURNACE)] == 1
    assert sum(cost.values()) == 4


def test_canonical_cell_emits_four_placements_in_order() -> None:
    """The four pieces land in placement-safe order with correct facings."""
    goals = build_smelter_cell_at((8, 11))
    placements = _placements(goals)
    assert placements == [
        # coal_buffer south of furnace, faces UP toward the furnace
        (int(MachineType.PALLET), (8, 12), int(Direction.UP)),
        # plate-bus east of arm, faces DOWN so its stand tile is south
        (int(MachineType.PALLET), (10, 11), int(Direction.DOWN)),
        # arm east of furnace, faces RIGHT (pulls plate east)
        (int(MachineType.ARM), (9, 11), int(Direction.RIGHT)),
        # furnace last — its stand tile is one west, kept walkable
        # because the arm doesn't share that column
        (int(MachineType.FURNACE), (8, 11), int(Direction.RIGHT)),
    ]


def test_arm_precedes_furnace_to_keep_arm_stand_tile_walkable() -> None:
    """Arm placement must come before furnace placement.

    The arm's stand tile (one west of the arm, i.e. the furnace's
    eventual location) must be walkable dirt at arm-place time.
    Reversing the order would put the arm's stand tile on a freshly-
    placed furnace and fail at runtime.
    """
    goals = build_smelter_cell_at((8, 11))
    types = [g.machine_type for g in goals]
    arm_idx = types.index(int(MachineType.ARM))
    furnace_idx = types.index(int(MachineType.FURNACE))
    assert arm_idx < furnace_idx


def test_collision_with_occupied_tile_raises_with_label() -> None:
    """The error message names the offending cell tile and its role."""
    with pytest.raises(ValueError, match=r"arm tile \(9, 11\)"):
        build_smelter_cell_at((8, 11), occupied={(9, 11)})


def test_collision_with_coal_buffer_tile_raises() -> None:
    with pytest.raises(ValueError, match=r"coal_buffer tile \(8, 12\)"):
        build_smelter_cell_at((8, 11), occupied={(8, 12)})


def test_out_of_bounds_furnace_raises() -> None:
    """Placing the cell at the east edge pushes plate-bus off-map."""
    with pytest.raises(ValueError, match="plate_bus tile"):
        # furnace at (30, 11) → plate-bus at (32, 11), off the 32x32 map
        build_smelter_cell_at((30, 11), map_size=(32, 32))


def test_out_of_bounds_coal_buffer_raises() -> None:
    """Cell at the south edge pushes coal-buffer off-map."""
    with pytest.raises(ValueError, match="coal_buffer tile"):
        # furnace at (8, 31) → coal_buffer at (8, 32), off the 32x32 map
        build_smelter_cell_at((8, 31), map_size=(32, 32))


def test_negative_coordinates_raise() -> None:
    with pytest.raises(ValueError, match="furnace tile"):
        build_smelter_cell_at((-1, 5), map_size=(32, 32))


def test_disjoint_cells_via_occupied_set() -> None:
    """Build two cells; pass cell A's tiles as occupied to cell B."""
    cell_a = build_smelter_cell_at((8, 11))
    a_tiles = {g.target for g in cell_a}
    # Cell B is far enough away that none of its tiles overlap.
    cell_b = build_smelter_cell_at((23, 11), occupied=a_tiles)
    b_tiles = {g.target for g in cell_b}
    assert a_tiles.isdisjoint(b_tiles)


def test_overlapping_cells_via_occupied_set_raises() -> None:
    """Cell A's plate-bus collides with Cell B's furnace."""
    cell_a = build_smelter_cell_at((8, 11))  # plate_bus at (10, 11)
    a_tiles = {g.target for g in cell_a}
    with pytest.raises(ValueError, match=r"furnace tile \(10, 11\)"):
        build_smelter_cell_at((10, 11), occupied=a_tiles)


def test_facing_left_mirrors_arm_and_plate_bus_to_west() -> None:
    """``facing=LEFT`` flips the arm and plate-bus to the *west* side
    while keeping the coal-buffer south."""
    goals = build_smelter_cell_at((23, 11), facing=int(Direction.LEFT))
    placements = _placements(goals)
    assert placements == [
        # coal_buffer south, unchanged from RIGHT case
        (int(MachineType.PALLET), (23, 12), int(Direction.UP)),
        # plate_bus two tiles WEST of furnace, still facing DOWN
        (int(MachineType.PALLET), (21, 11), int(Direction.DOWN)),
        # arm one tile WEST of furnace, facing LEFT
        (int(MachineType.ARM), (22, 11), int(Direction.LEFT)),
        # furnace facing LEFT
        (int(MachineType.FURNACE), (23, 11), int(Direction.LEFT)),
    ]


def test_facing_other_than_left_or_right_raises() -> None:
    with pytest.raises(ValueError, match="facing must be LEFT or RIGHT"):
        build_smelter_cell_at((8, 11), facing=int(Direction.UP))


def test_extract_facing_adds_extractor_arm_before_plate_bus() -> None:
    """Extractor arm sits at plate_bus + unit(extract_facing) and
    is placed *between* coal_buffer and plate_bus so its stand tile
    (= plate_bus's eventual location) is still walkable dirt."""
    goals = build_smelter_cell_at(
        (8, 11),
        facing=int(Direction.RIGHT),
        extract_facing=int(Direction.RIGHT),
    )
    placements = _placements(goals)
    assert placements == [
        (int(MachineType.PALLET), (8, 12), int(Direction.UP)),  # coal
        # Extractor at (11, 11) facing RIGHT — plate_bus is (10, 11),
        # extractor sits one step further east.
        (int(MachineType.ARM), (11, 11), int(Direction.RIGHT)),
        (int(MachineType.PALLET), (10, 11), int(Direction.DOWN)),
        (int(MachineType.ARM), (9, 11), int(Direction.RIGHT)),
        (int(MachineType.FURNACE), (8, 11), int(Direction.RIGHT)),
    ]


def test_extract_facing_for_mirrored_copper_cell() -> None:
    """Mirrored copper cell with extractor: extractor is one tile
    further WEST of the plate-bus."""
    goals = build_smelter_cell_at(
        (23, 11),
        facing=int(Direction.LEFT),
        extract_facing=int(Direction.LEFT),
    )
    placements = _placements(goals)
    # Order: coal -> extractor -> plate_bus -> arm -> furnace
    assert placements[0][1] == (23, 12)  # coal
    assert placements[1] == (
        int(MachineType.ARM),
        (20, 11),
        int(Direction.LEFT),
    )  # extractor
    assert placements[2][1] == (21, 11)  # plate_bus
    assert placements[3][1] == (22, 11)  # cell arm
    assert placements[4][1] == (23, 11)  # furnace


def test_extract_facing_opposite_of_facing_raises() -> None:
    """Extracting in the opposite direction would put the extractor
    on top of the cell arm — rejected."""
    with pytest.raises(ValueError, match="opposite"):
        build_smelter_cell_at(
            (8, 11),
            facing=int(Direction.RIGHT),
            extract_facing=int(Direction.LEFT),
        )


def test_extract_facing_invalid_direction_raises() -> None:
    with pytest.raises(ValueError, match="UP/DOWN/LEFT/RIGHT"):
        build_smelter_cell_at(
            (8, 11),
            extract_facing=99,
        )


def test_inventory_with_extractor_adds_one_arm() -> None:
    cost = smelter_cell_inventory(with_extractor=True)
    assert cost[int(ItemType.PALLET)] == 2
    assert cost[int(ItemType.ARM)] == 2  # cell arm + extractor
    assert cost[int(ItemType.FURNACE)] == 1
