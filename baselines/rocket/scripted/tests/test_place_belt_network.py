"""Unit tests for :func:`place_belt_network` and
:func:`belt_network_inventory`.

Covers the conflict-resolution rules: single-path tiles emit BELTs,
two perpendicular paths emit a CROSSING with the right encoding, and
every other multi-path collision raises ``ValueError`` at goal-
construction time so a misrouted trunk surfaces while building the
goal list rather than 5000 ticks into the rollout.
"""

from __future__ import annotations

import pytest

from baselines.rocket.scripted.goals import (
    BeltPath,
    PlaceMachineAt,
    belt_network_inventory,
    place_belt_network,
)
from factoriax.engine.constants import Direction, ItemType, Machine


def _placements(goals: list) -> list[tuple[int, tuple[int, int], int]]:
    out = []
    for goal in goals:
        assert isinstance(goal, PlaceMachineAt)
        out.append((goal.machine_type, goal.target, goal.facing))
    return out


def _belts_and_crossings(
    goals: list,
) -> tuple[
    list[tuple[tuple[int, int], int]],
    list[tuple[tuple[int, int], int]],
]:
    """Split a goal list into (belt_tile_dir_pairs, crossing_tile_enc_pairs)."""
    belts: list[tuple[tuple[int, int], int]] = []
    crossings: list[tuple[tuple[int, int], int]] = []
    for goal in goals:
        assert isinstance(goal, PlaceMachineAt)
        if goal.machine_type == int(Machine.CONVEYOR_BELT):
            belts.append((goal.target, goal.facing))
        elif goal.machine_type == int(Machine.CROSSING):
            crossings.append((goal.target, goal.facing))
        else:
            raise AssertionError(f"unexpected goal type: {goal.machine_type}")
    return belts, crossings


# ---------------------------------------------------------------------------
# Single-path equivalence to place_belt_path
# ---------------------------------------------------------------------------


def test_single_path_emits_belts_only_no_crossings() -> None:
    """A single path through the network is just a chain of belts."""
    path = BeltPath([(0, 0), (3, 0)], label="solo")
    goals = place_belt_network([path])
    belts, crossings = _belts_and_crossings(goals)
    assert crossings == []
    # Sink at (3, 0); belts at (0, 0), (1, 0), (2, 0).
    assert {tile for tile, _ in belts} == {(0, 0), (1, 0), (2, 0)}
    assert all(d == int(Direction.RIGHT) for _, d in belts)


def test_single_path_belt_directions_follow_corners() -> None:
    """A path that bends mid-way has the corner belt take the
    outgoing segment's direction (matches place_belt_path semantics)."""
    path = BeltPath([(0, 0), (0, 2), (2, 2)], label="L")
    goals = place_belt_network([path])
    belts, _ = _belts_and_crossings(goals)
    by_tile = dict(belts)
    assert by_tile[(0, 0)] == int(Direction.DOWN)
    assert by_tile[(0, 1)] == int(Direction.DOWN)
    # Corner tile takes the outgoing direction (RIGHT).
    assert by_tile[(0, 2)] == int(Direction.RIGHT)
    assert by_tile[(1, 2)] == int(Direction.RIGHT)


# ---------------------------------------------------------------------------
# Crossings — encoding correctness for all 4 (vert, horiz) combinations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "vert_path, horiz_path, expected_encoding",
    [
        # vert path goes UP→DOWN (DOWN flow), horiz LEFT→RIGHT
        (
            [(2, 0), (2, 4)],
            [(0, 2), (4, 2)],
            1,  # vert=DOWN, horiz=RIGHT
        ),
        # vert UP→DOWN, horiz RIGHT→LEFT
        (
            [(2, 0), (2, 4)],
            [(4, 2), (0, 2)],
            2,  # vert=DOWN, horiz=LEFT
        ),
        # vert DOWN→UP, horiz LEFT→RIGHT
        (
            [(2, 4), (2, 0)],
            [(0, 2), (4, 2)],
            3,  # vert=UP, horiz=RIGHT
        ),
        # vert DOWN→UP, horiz RIGHT→LEFT
        (
            [(2, 4), (2, 0)],
            [(4, 2), (0, 2)],
            4,  # vert=UP, horiz=LEFT
        ),
    ],
)
def test_perpendicular_intersection_emits_crossing_with_correct_encoding(
    vert_path: list[tuple[int, int]],
    horiz_path: list[tuple[int, int]],
    expected_encoding: int,
) -> None:
    """Two perpendicular paths intersect at exactly one tile; that
    tile becomes a CROSSING with encoding derived from the (vert, horiz)
    direction pair per CROSSING_AXIS_DIRS."""
    goals = place_belt_network(
        [
            BeltPath(vert_path, label="vert"),
            BeltPath(horiz_path, label="horiz"),
        ],
    )
    belts, crossings = _belts_and_crossings(goals)
    assert len(crossings) == 1
    crossing_tile, encoding = crossings[0]
    assert crossing_tile == (2, 2)
    assert encoding == expected_encoding


# ---------------------------------------------------------------------------
# Conflict cases — must raise ValueError with informative message
# ---------------------------------------------------------------------------


def test_parallel_collision_raises() -> None:
    """Two paths pushing the same direction through the same tile.

    Path A is a long east-bound trunk at row 0. Path B enters row 0
    from the north at column 1 and exits to the north at column 4 —
    its belts at (1, 0), (2, 0), (3, 0) all push RIGHT, just like
    path A's belts at the same tiles. Sinks are off the conflict
    tiles so the parallel-collision check fires first."""
    paths = [
        BeltPath([(0, 0), (5, 0)], label="A"),
        BeltPath(
            [(1, -1), (1, 0), (4, 0), (4, -1)],
            label="B",
        ),
    ]
    with pytest.raises(ValueError, match="parallel collision"):
        place_belt_network(paths)


def test_anti_parallel_collision_raises() -> None:
    """Two paths pushing opposite directions on the same axis.

    Path A drops south through column 1 then bends east; path B
    rises north through column 1 then bends east. Sinks land at
    column 3 (different rows) so anti-parallel fires first."""
    paths = [
        BeltPath([(1, -1), (1, 4), (3, 4)], label="south-flow"),
        BeltPath([(1, 6), (1, 1), (3, 1)], label="north-flow"),
    ]
    with pytest.raises(ValueError, match="anti-parallel"):
        place_belt_network(paths)


def test_three_way_junction_raises() -> None:
    """Three paths share a single tile — only 2-axis crossings supported.

    East/south/north paths all use (5, 5). The east path is inserted
    into the per-tile use map first, so when classification iterates,
    (5, 5) is the first multi-use tile encountered and the 3-way
    error fires before any anti-parallel error from the
    south/north overlap on (5, 6..8)."""
    paths = [
        BeltPath([(0, 5), (10, 5)], label="east"),
        BeltPath([(5, 0), (5, 10)], label="south"),
        BeltPath([(5, 8), (5, -1)], label="north"),
    ]
    with pytest.raises(ValueError, match="3-way"):
        place_belt_network(paths)


def test_belt_passes_through_another_paths_sink_raises() -> None:
    """A path's belt-cell coincides with another path's sink tile."""
    paths = [
        BeltPath([(0, 0), (4, 0)], label="east-trunk"),
        BeltPath([(2, -2), (2, 0)], label="north-feeder"),
    ]
    # north-feeder sinks at (2, 0); east-trunk's belt at (2, 0).
    with pytest.raises(ValueError, match="sinks cannot be passed through"):
        place_belt_network(paths)


def test_self_intersecting_path_raises() -> None:
    """A path that doubles back on itself."""
    paths = [
        BeltPath(
            [(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)],
            label="loop",
        ),
    ]
    with pytest.raises(ValueError, match="self-intersects"):
        place_belt_network(paths)


def test_collision_with_occupied_tile_raises() -> None:
    paths = [BeltPath([(0, 0), (3, 0)], label="east")]
    occupied = {(1, 0)}
    with pytest.raises(ValueError, match="collides with an occupied tile"):
        place_belt_network(paths, occupied=occupied)


def test_out_of_bounds_belt_raises() -> None:
    paths = [BeltPath([(0, 0), (5, 0)], label="east")]
    with pytest.raises(ValueError, match="out of map bounds"):
        place_belt_network(paths, map_size=(4, 4))


def test_unique_sink_constraint() -> None:
    """Two paths that share a sink tile must raise."""
    paths = [
        BeltPath([(0, 0), (3, 0)], label="A"),
        BeltPath([(3, -2), (3, 0)], label="B"),  # also sinks at (3, 0)
    ]
    with pytest.raises(ValueError, match="sinks must be unique"):
        place_belt_network(paths)


# ---------------------------------------------------------------------------
# Topological sort — placement order keeps stand tiles walkable
# ---------------------------------------------------------------------------


def test_belt_chain_emits_in_path_order() -> None:
    """A simple east-bound chain emits belts in west→east order so
    each belt's stand tile (its west neighbour) is the previously
    placed belt."""
    path = BeltPath([(5, 0), (10, 0)], label="east")
    goals = place_belt_network([path])
    tiles_in_order = [g.target for g in goals]
    # First belt at (5, 0); each subsequent belt is one east.
    assert tiles_in_order == [(5, 0), (6, 0), (7, 0), (8, 0), (9, 0)]


def test_two_paths_with_crossing_topological_order_is_valid() -> None:
    """For two paths intersecting at a crossing, the emit order must
    place each tile only when its stand tile is walkable (dirt or an
    already-placed belt/crossing)."""
    paths = [
        BeltPath([(2, 0), (2, 4)], label="vert"),  # DOWN
        BeltPath([(0, 2), (4, 2)], label="horiz"),  # RIGHT
    ]
    goals = place_belt_network(paths)
    placed: set[tuple[int, int]] = set()
    direction_offset = {
        int(Direction.UP): (0, -1),
        int(Direction.DOWN): (0, 1),
        int(Direction.LEFT): (-1, 0),
        int(Direction.RIGHT): (1, 0),
    }
    for goal in goals:
        assert isinstance(goal, PlaceMachineAt)
        tile = goal.target
        if goal.machine_type == int(Machine.CONVEYOR_BELT):
            dx, dy = direction_offset[goal.facing]
            stand = (tile[0] - dx, tile[1] - dy)
            assert stand in placed or stand not in {
                g.target for g in goals if g.target != tile
            }, f"belt at {tile} placed before its stand tile {stand} was walkable"
        placed.add(tile)


# ---------------------------------------------------------------------------
# Inventory helper
# ---------------------------------------------------------------------------


def test_inventory_counts_belts_only_for_solo_path() -> None:
    paths = [BeltPath([(0, 0), (5, 0)], label="solo")]
    cost = belt_network_inventory(paths)
    assert cost == {int(ItemType.CONVEYOR_BELT): 5}
    assert int(ItemType.CROSSING) not in cost


def test_inventory_counts_belts_and_crossings_for_two_path_intersection() -> None:
    paths = [
        BeltPath([(2, 0), (2, 4)], label="vert"),  # 4 cells, sink at (2, 4)
        BeltPath([(0, 2), (4, 2)], label="horiz"),  # 4 cells, sink at (4, 2)
    ]
    cost = belt_network_inventory(paths)
    # 4 + 4 = 8 cells, but (2, 2) is shared → 7 distinct tiles.
    # 1 of them is a crossing, the rest are belts.
    assert cost[int(ItemType.CROSSING)] == 1
    assert cost[int(ItemType.CONVEYOR_BELT)] == 6


def test_inventory_validates_inputs() -> None:
    """A malformed input (parallel collision) raises through the
    inventory helper too — pre-flight at bootstrap-sizing time."""
    paths = [
        BeltPath([(0, 0), (5, 0)], label="A"),
        BeltPath(
            [(1, -1), (1, 0), (4, 0), (4, -1)],
            label="B",
        ),
    ]
    with pytest.raises(ValueError, match="parallel collision"):
        belt_network_inventory(paths)


# ---------------------------------------------------------------------------
# place_belt_path equivalence
# ---------------------------------------------------------------------------


def test_single_path_matches_place_belt_path_output() -> None:
    """For a single path, place_belt_network and place_belt_path must
    emit equivalent goals (same belts + directions). place_belt_path
    is the documented entry point for single-path callers."""
    from baselines.rocket.scripted.goals import place_belt_path

    waypoints = [(5, 5), (5, 8), (8, 8)]
    via_network = place_belt_network([BeltPath(waypoints)])
    via_path = place_belt_path(waypoints)
    # Same set of (tile, direction) pairs (ordering may differ
    # because of topological-sort tie-breaking).
    assert {(g.target, g.facing) for g in via_network} == {
        (g.target, g.facing) for g in via_path
    }
