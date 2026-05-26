"""Unit tests for :func:`place_ore_node` and :func:`ore_node_inventory`."""

from __future__ import annotations

import pytest

from baselines.rocket.scripted.goals import (
    PlaceMachineAt,
    ore_node_inventory,
    place_ore_node,
)
from factoriax.constants import Direction, ItemType, Machine


def _placements(goals: list) -> list[tuple[int, tuple[int, int], int]]:
    out = []
    for goal in goals:
        assert isinstance(goal, PlaceMachineAt)
        out.append((goal.machine_type, goal.target, goal.facing))
    return out


def test_inventory_with_pallet_costs_one_miner_one_pallet() -> None:
    cost = ore_node_inventory(with_pallet=True)
    assert cost[int(ItemType.MINER)] == 1
    assert cost[int(ItemType.PALLET)] == 1
    assert sum(cost.values()) == 2


def test_inventory_without_pallet_costs_one_miner() -> None:
    cost = ore_node_inventory(with_pallet=False)
    assert cost == {int(ItemType.MINER): 1}


def test_default_down_node_emits_pallet_then_miner() -> None:
    """South-edge DOWN-facing node: pallet south of miner, miner south-edge."""
    goals = place_ore_node((8, 9))  # iron south-edge centre
    placements = _placements(goals)
    assert placements == [
        (int(Machine.PALLET), (8, 10), int(Direction.DOWN)),
        (int(Machine.MINER), (8, 9), int(Direction.DOWN)),
    ]


def test_pallet_precedes_miner() -> None:
    """Pallet first so the miner's stand tile remains walkable ore.

    Reversed order would place the miner first, then try to place
    the pallet from a stand tile that's now adjacent to the miner —
    fine in this layout, but the documented order is pallet-first
    and the helper preserves it.
    """
    goals = place_ore_node((8, 9))
    types = [g.machine_type for g in goals]
    assert types.index(int(Machine.PALLET)) < types.index(int(Machine.MINER))


def test_right_facing_node_puts_pallet_east() -> None:
    goals = place_ore_node((9, 24), direction=int(Direction.RIGHT))
    placements = _placements(goals)
    assert placements == [
        (int(Machine.PALLET), (10, 24), int(Direction.RIGHT)),
        (int(Machine.MINER), (9, 24), int(Direction.RIGHT)),
    ]


def test_up_facing_node_puts_pallet_north() -> None:
    goals = place_ore_node((7, 22), direction=int(Direction.UP))
    placements = _placements(goals)
    assert placements == [
        (int(Machine.PALLET), (7, 21), int(Direction.UP)),
        (int(Machine.MINER), (7, 22), int(Direction.UP)),
    ]


def test_with_pallet_false_emits_only_miner() -> None:
    """Coal-node pattern: miner pushes onto a belt, no pallet."""
    goals = place_ore_node(
        (9, 24),
        direction=int(Direction.RIGHT),
        with_pallet=False,
    )
    placements = _placements(goals)
    assert placements == [
        (int(Machine.MINER), (9, 24), int(Direction.RIGHT)),
    ]


def test_invalid_direction_raises() -> None:
    with pytest.raises(ValueError, match="UP/DOWN/LEFT/RIGHT"):
        place_ore_node((8, 9), direction=99)


def test_collision_with_miner_tile_raises() -> None:
    with pytest.raises(ValueError, match=r"miner tile \(8, 9\)"):
        place_ore_node((8, 9), occupied={(8, 9)})


def test_collision_with_pallet_tile_raises() -> None:
    with pytest.raises(ValueError, match=r"pallet tile \(8, 10\)"):
        place_ore_node((8, 9), occupied={(8, 10)})


def test_out_of_bounds_pallet_raises() -> None:
    """Miner at south edge of 32x32 map → pallet off the map."""
    with pytest.raises(ValueError, match="pallet tile"):
        place_ore_node((8, 31), map_size=(32, 32))


def test_out_of_bounds_miner_raises() -> None:
    with pytest.raises(ValueError, match="miner tile"):
        place_ore_node((-1, 5), map_size=(32, 32))
