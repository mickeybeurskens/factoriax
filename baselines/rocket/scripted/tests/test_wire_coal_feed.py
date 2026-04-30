"""Unit tests for :func:`wire_coal_feed` and :func:`coal_feed_inventory`."""

from __future__ import annotations

import pytest

from baselines.rocket.scripted.goals import (
    PlaceMachineAt,
    coal_feed_inventory,
    wire_coal_feed,
)
from factoriax.constants import Direction, ItemType, MachineType


def _types(goals: list) -> list[int]:
    return [g.machine_type for g in goals if isinstance(g, PlaceMachineAt)]


def _targets(goals: list) -> list[tuple[int, int]]:
    return [g.target for g in goals if isinstance(g, PlaceMachineAt)]


def test_iron_layout_emits_miner_first_then_belts() -> None:
    """Iron's RIGHT-facing miner at (9, 24) doesn't block the trunk's
    first belt at (10, 24) UP (stand (10, 25) = dirt), so the helper
    emits miner first and trunk after."""
    goals = wire_coal_feed(
        miner_tile=(9, 24),
        miner_facing=int(Direction.RIGHT),
        trunk_waypoints=[(10, 24), (10, 12), (8, 12)],
    )
    types = _types(goals)
    # First placement is the miner; everything after is belts.
    assert types[0] == int(MachineType.MINER)
    assert all(t == int(MachineType.CONVEYOR_BELT) for t in types[1:])
    # Total: 1 miner + 14 belts (matches iron's documented trunk length).
    assert len(types) == 15


def test_tin_layout_emits_trunk_first_then_miner() -> None:
    """Tin's DOWN-facing miner at (7, 24) sits on the first belt's
    stand tile ((7, 25) DOWN → stand (7, 24)), so the helper emits
    the trunk first and the miner last."""
    goals = wire_coal_feed(
        miner_tile=(7, 24),
        miner_facing=int(Direction.DOWN),
        trunk_waypoints=[(7, 25), (7, 26), (22, 26), (22, 27), (23, 27)],
    )
    types = _types(goals)
    # Last placement is the miner; everything before is belts.
    assert types[-1] == int(MachineType.MINER)
    assert all(t == int(MachineType.CONVEYOR_BELT) for t in types[:-1])
    # Total: 18 belts + 1 miner.
    assert len(types) == 19


def test_silicon_layout_trunk_first() -> None:
    """Silicon's UP-facing miner at (7, 22) collides with first belt's
    stand tile ((7, 21) UP → stand (7, 22)) — same trunk-first branch."""
    goals = wire_coal_feed(
        miner_tile=(7, 22),
        miner_facing=int(Direction.UP),
        trunk_waypoints=[(7, 21), (7, 8), (14, 8), (15, 8)],
    )
    types = _types(goals)
    assert types[-1] == int(MachineType.MINER)
    assert sum(1 for t in types if t == int(MachineType.CONVEYOR_BELT)) == 21


def test_trunk_start_must_match_miner_push_tile() -> None:
    with pytest.raises(ValueError, match="pushes onto"):
        wire_coal_feed(
            miner_tile=(9, 24),
            miner_facing=int(Direction.RIGHT),
            # Trunk starts at (11, 24), but miner pushes onto (10, 24).
            trunk_waypoints=[(11, 24), (11, 12)],
        )


def test_invalid_facing_raises() -> None:
    with pytest.raises(ValueError, match="UP/DOWN/LEFT/RIGHT"):
        wire_coal_feed(
            miner_tile=(9, 24),
            miner_facing=99,
            trunk_waypoints=[(10, 24), (10, 12)],
        )


def test_too_short_waypoints_raises() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        wire_coal_feed(
            miner_tile=(9, 24),
            miner_facing=int(Direction.RIGHT),
            trunk_waypoints=[(10, 24)],
        )


def test_miner_tile_collides_with_occupied_raises() -> None:
    with pytest.raises(ValueError, match=r"miner tile \(9, 24\)"):
        wire_coal_feed(
            miner_tile=(9, 24),
            miner_facing=int(Direction.RIGHT),
            trunk_waypoints=[(10, 24), (10, 12)],
            occupied={(9, 24)},
        )


def test_trunk_avoids_miner_in_miner_first_branch() -> None:
    """When the miner is placed first, subsequent belts must not land
    on the miner's tile. The helper adds miner_tile to the trunk's
    occupied set automatically."""
    # Miner at (10, 23) facing DOWN pushes onto (10, 24) — trunk
    # entry. From there a westbound trunk to (8, 24) goes through
    # (9, 24), which sits on top of nothing problematic on its own,
    # but if we instead have miner at (9, 24) facing RIGHT and a
    # waypoint chain that loops back west, the helper must reject
    # the conflict. Use a straight east-to-west trunk for the
    # cleanest case: belt at (9, 24) would land on the miner tile.
    with pytest.raises(ValueError, match=r"belt at \(9, 24\) collides"):
        wire_coal_feed(
            miner_tile=(9, 24),
            miner_facing=int(Direction.RIGHT),
            # Trunk starts at the miner's push tile (10, 24) then
            # heads west — the (9, 24) belt would sit on the miner.
            trunk_waypoints=[(10, 24), (8, 24)],
        )


def test_inventory_iron_returns_miner_plus_14_belts() -> None:
    cost = coal_feed_inventory([(10, 24), (10, 12), (8, 12)])
    assert cost[int(ItemType.MINER)] == 1
    assert cost[int(ItemType.CONVEYOR_BELT)] == 14


def test_inventory_silicon_returns_miner_plus_21_belts() -> None:
    cost = coal_feed_inventory([(7, 21), (7, 8), (14, 8), (15, 8)])
    assert cost[int(ItemType.CONVEYOR_BELT)] == 21


def test_out_of_bounds_trunk_raises() -> None:
    with pytest.raises(ValueError, match="trunk waypoint|out of"):
        wire_coal_feed(
            miner_tile=(30, 9),
            miner_facing=int(Direction.RIGHT),
            trunk_waypoints=[(31, 9), (35, 9)],
            map_size=(32, 32),
        )
