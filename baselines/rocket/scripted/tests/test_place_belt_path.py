"""Unit tests for :func:`place_belt_path`.

Pure-Python — no env rollout — so the suite runs in milliseconds and
catches segment / direction / collision errors before the integration
tests pay the ~30s rollout cost.
"""

from __future__ import annotations

import pytest

from baselines.rocket.scripted.goals import (
    PlaceMachineAt,
    place_belt_path,
)
from factoriax.engine.constants import Direction, Machine


def _belt_specs(goals: list) -> list[tuple[tuple[int, int], int]]:
    """Pull (target, facing) pairs out of a list of PlaceMachineAt goals."""
    out = []
    for g in goals:
        assert isinstance(g, PlaceMachineAt)
        assert g.machine_type == int(Machine.CONVEYOR_BELT)
        out.append((g.target, g.facing))
    return out


def test_straight_horizontal_path_emits_right_facing_belts() -> None:
    goals = place_belt_path([(2, 5), (5, 5)])
    assert _belt_specs(goals) == [
        ((2, 5), int(Direction.RIGHT)),
        ((3, 5), int(Direction.RIGHT)),
        ((4, 5), int(Direction.RIGHT)),
    ]


def test_straight_vertical_path_emits_up_facing_belts() -> None:
    goals = place_belt_path([(7, 21), (7, 18)])
    assert _belt_specs(goals) == [
        ((7, 21), int(Direction.UP)),
        ((7, 20), int(Direction.UP)),
        ((7, 19), int(Direction.UP)),
    ]


def test_corner_belt_takes_outgoing_segment_direction() -> None:
    """Iron-trunk shape: south-to-north, then west to a sink one west."""
    goals = place_belt_path([(10, 14), (10, 12), (8, 12)])
    specs = _belt_specs(goals)
    # Corner at (10, 12) faces LEFT (the direction of the next segment).
    assert ((10, 12), int(Direction.LEFT)) in specs
    # Sink (8, 12) is NOT in the belt list — it's the receiver tile.
    assert all(target != (8, 12) for target, _ in specs)
    assert len(specs) == 4  # (10, 14), (10, 13), (10, 12), (9, 12)


def test_silicon_route_belt_count_matches_manhattan_distance() -> None:
    """Silicon trunk waypoints produce 21 belts (= path length - 1)."""
    goals = place_belt_path([(7, 21), (7, 8), (14, 8), (15, 8)])
    assert len(goals) == 21


def test_collision_with_occupied_tile_raises() -> None:
    with pytest.raises(ValueError, match="collides"):
        place_belt_path([(0, 0), (4, 0)], occupied={(2, 0)})


def test_self_intersecting_path_raises() -> None:
    """A path that revisits a belt tile (non-sink) is rejected."""
    # Right to (3,0), down to (3,2), left to (1,2), up to (1,0) — the
    # tile (1, 0) was already a belt in the first segment.
    with pytest.raises(ValueError, match="self-intersects"):
        place_belt_path([(0, 0), (3, 0), (3, 2), (1, 2), (1, 0), (4, 0)])


def test_path_loop_back_to_start_raises_sink_clash() -> None:
    """A loop whose sink coincides with the start belt is rejected."""
    with pytest.raises(ValueError, match="already on belt path"):
        place_belt_path([(0, 0), (3, 0), (3, 1), (0, 1), (0, 0)])


def test_degenerate_segment_raises() -> None:
    with pytest.raises(ValueError, match="degenerate"):
        place_belt_path([(3, 0), (3, 0)])


def test_diagonal_segment_raises() -> None:
    with pytest.raises(ValueError, match="not axis-aligned"):
        place_belt_path([(0, 0), (3, 3)])


def test_too_few_waypoints_raises() -> None:
    with pytest.raises(ValueError, match=">=2"):
        place_belt_path([(0, 0)])


def test_sink_on_path_raises() -> None:
    """The sink waypoint must not coincide with an earlier path tile."""
    with pytest.raises(ValueError, match="self-intersects|already on belt path"):
        # Path goes right from (0,0) to (3,0) then back to (1,0) — sink (1,0)
        # was already a belt in segment 1.
        place_belt_path([(0, 0), (3, 0), (1, 0)])
