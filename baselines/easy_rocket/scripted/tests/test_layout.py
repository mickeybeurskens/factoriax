"""Tests for the factory layout planner.

Pure DAG-analysis helpers run on the recipe table; the ``plan_factory``
interface test builds a real easy_rocket state (array construction,
no env-step JIT) and asserts structural invariants of the plan.
"""

from __future__ import annotations

import jax
import pytest

import factoriax
from baselines.easy_rocket.scripted.layout import (
    HAND_CRAFTED_FACTORY_ITEMS,
    _consumers_per_ore,
    _crossing_direction,
    _walk_dag,
    plan_factory,
)
from factoriax.constants import Direction, ItemType
from factoriax.levels import build_state
from factoriax.scenarios.easy_rocket import (
    EASY_ROCKET_RECIPE_TABLE,
    build_easy_rocket_level,
    easy_rocket_conditions,
)

_DXY = {
    int(Direction.LEFT): (-1, 0),
    int(Direction.RIGHT): (1, 0),
    int(Direction.UP): (0, -1),
    int(Direction.DOWN): (0, 1),
}


# --- Pure DAG helpers -------------------------------------------------------


def test_crossing_direction_table() -> None:
    d, r, u, lft = (
        int(Direction.DOWN),
        int(Direction.RIGHT),
        int(Direction.UP),
        int(Direction.LEFT),
    )
    assert _crossing_direction(d, r) == 1
    assert _crossing_direction(d, lft) == 2
    assert _crossing_direction(u, r) == 3
    assert _crossing_direction(u, lft) == 4
    # Order-independent (one horizontal, one vertical, either order).
    assert _crossing_direction(r, d) == _crossing_direction(d, r)


def test_walk_dag_finds_automated_chain_and_ores() -> None:
    automated, ores = _walk_dag(
        EASY_ROCKET_RECIPE_TABLE,
        int(ItemType.ROCKET),
        HAND_CRAFTED_FACTORY_ITEMS,
    )
    # Three automated assemblers; the target is last (topological).
    assert automated == [
        int(ItemType.HULL),
        int(ItemType.ENGINE_UNIT),
        int(ItemType.ROCKET),
    ]
    # All six easy_rocket ores are reachable (TIN via the ARM recipe).
    assert ores == {
        int(ItemType.IRON_ORE),
        int(ItemType.COPPER_ORE),
        int(ItemType.TIN_ORE),
        int(ItemType.SILICON),
        int(ItemType.COAL),
        int(ItemType.LIMESTONE),
    }


def test_consumers_per_ore() -> None:
    automated, _ = _walk_dag(
        EASY_ROCKET_RECIPE_TABLE,
        int(ItemType.ROCKET),
        HAND_CRAFTED_FACTORY_ITEMS,
    )
    consumers = _consumers_per_ore(EASY_ROCKET_RECIPE_TABLE, automated)
    assert consumers[int(ItemType.IRON_ORE)] == [int(ItemType.HULL)]
    assert consumers[int(ItemType.COPPER_ORE)] == [int(ItemType.ENGINE_UNIT)]
    assert sorted(consumers[int(ItemType.LIMESTONE)]) == sorted(
        [int(ItemType.HULL), int(ItemType.ENGINE_UNIT)]
    )
    # Ores used only by hand-crafted machines have no automated consumer.
    assert int(ItemType.COAL) not in consumers
    assert int(ItemType.TIN_ORE) not in consumers


# --- plan_factory interface -------------------------------------------------


def _easy_rocket_state(seed: int):
    level = build_easy_rocket_level(jax.random.PRNGKey(seed))
    _, params = factoriax.make(
        level, obs="global", achievement_fn=easy_rocket_conditions
    )
    params = params.replace(
        num_players=1,
        max_timesteps=2000,
        recipe_table=EASY_ROCKET_RECIPE_TABLE,
    )
    return build_state(level, params)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_plan_factory_valid_and_structured(seed: int) -> None:
    state = _easy_rocket_state(seed)
    layout = plan_factory(state, EASY_ROCKET_RECIPE_TABLE)

    assert layout.valid, layout.error

    # Three automated assemblers.
    outputs = {a.recipe_output for a in layout.assemblers}
    assert outputs == {
        int(ItemType.HULL),
        int(ItemType.ENGINE_UNIT),
        int(ItemType.ROCKET),
    }

    # One manual miner per ore (6), plus one pallet each.
    manual = [m for m in layout.miners if m.role == "manual"]
    factory = [m for m in layout.miners if m.role == "factory"]
    assert len(manual) == 6
    assert len(layout.pallets) == len(manual)
    # IRON + COPPER (1 each) + LIMESTONE (2) factory miners.
    assert len(factory) == 4

    # Every machine sits on a distinct tile.
    tiles = (
        [m.pos for m in layout.miners]
        + [a.pos for a in layout.assemblers]
        + [arm.pos for arm in layout.arms]
        + [b.pos for b in layout.belts]
        + [c.pos for c in layout.crossings]
        + [p.pos for p in layout.pallets]
        + [layout.rocket_tile]
    )
    assert len(tiles) == len(set(tiles)), "overlapping entity positions"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_manual_miner_faces_its_pallet(seed: int) -> None:
    state = _easy_rocket_state(seed)
    layout = plan_factory(state, EASY_ROCKET_RECIPE_TABLE)
    pallet_pos = {p.pos for p in layout.pallets}
    for m in layout.miners:
        if m.role != "manual":
            continue
        dx, dy = _DXY[m.facing]
        target = (m.pos[0] + dx, m.pos[1] + dy)
        assert target in pallet_pos, (
            f"manual miner at {m.pos} faces {target}, not a pallet"
        )
