"""Unit tests for the easy-rocket scenario."""

from __future__ import annotations

import jax
import numpy as np
import pytest

from factoriax.constants import BlockType, ItemType
from factoriax.levels import Level
from factoriax.recipes import RecipeBook, RecipeTable
from factoriax.scenarios.easy_rocket import (
    EASY_ROCKET_RECIPE_BOOK,
    EASY_ROCKET_RECIPE_TABLE,
    build_easy_rocket_level,
)

_SPAWN: tuple[int, int] = (8, 8)
_FORBID_RADIUS: int = 1
_MAP_SIZE: int = 16

_ORE_BLOCKS: frozenset[int] = frozenset(
    {
        int(BlockType.IRON),
        int(BlockType.COPPER),
        int(BlockType.TIN),
        int(BlockType.SILICON),
        int(BlockType.COAL),
        int(BlockType.LIMESTONE),
    }
)


def _levels_equal(a: Level, b: Level) -> bool:
    if (a.name, a.map_width, a.map_height) != (b.name, b.map_width, b.map_height):
        return False
    for field in (
        "block_map",
        "block_resources",
        "machine_types",
        "machine_directions",
        "machine_inventory",
        "machine_selected_recipe",
    ):
        av = getattr(a, field)
        bv = getattr(b, field)
        if (av is None) != (bv is None):
            return False
        if av is not None and not np.array_equal(av, bv):
            return False
    return a.player_positions == b.player_positions


_EXPECTED_OUTPUTS: frozenset[int] = frozenset(
    {
        int(ItemType.ASSEMBLER),
        int(ItemType.CONVEYOR_BELT),
        int(ItemType.SPLITTER),
        int(ItemType.CROSSING),
        int(ItemType.MINER),
        int(ItemType.HULL),
        int(ItemType.ENGINE_UNIT),
        int(ItemType.ROCKET),
    }
)


def test_recipe_book_constructs() -> None:
    assert isinstance(EASY_ROCKET_RECIPE_BOOK, RecipeBook)
    assert isinstance(EASY_ROCKET_RECIPE_TABLE, RecipeTable)

    actual_outputs = frozenset(r.output for r in EASY_ROCKET_RECIPE_BOOK.recipes)
    assert actual_outputs == _EXPECTED_OUTPUTS
    assert len(EASY_ROCKET_RECIPE_BOOK.recipes) == 8


def test_recipe_book_rocket_takes_hull_and_engine_unit() -> None:
    rocket_recipes = [
        r for r in EASY_ROCKET_RECIPE_BOOK.recipes if r.output == int(ItemType.ROCKET)
    ]
    assert len(rocket_recipes) == 1
    rocket = rocket_recipes[0]

    input_items = {item for item, _ in rocket.inputs}
    assert input_items == {int(ItemType.HULL), int(ItemType.ENGINE_UNIT)}

    forbidden = {int(ItemType.AVIONICS), int(ItemType.ROCKET_CORE)}

    output_items = {r.output for r in EASY_ROCKET_RECIPE_BOOK.recipes}
    assert not (output_items & forbidden), (
        "AVIONICS / ROCKET_CORE must not be outputs in the easy-rocket book"
    )

    for recipe in EASY_ROCKET_RECIPE_BOOK.recipes:
        input_set = {item for item, _ in recipe.inputs}
        assert not (input_set & forbidden), (
            f"Recipe {recipe.name!r} references forbidden input {forbidden & input_set}"
        )


def test_build_level_dimensions() -> None:
    level = build_easy_rocket_level(jax.random.PRNGKey(0))
    assert level.map_width == _MAP_SIZE
    assert level.map_height == _MAP_SIZE
    assert level.player_positions == [_SPAWN]
    assert level.machine_types is None


def test_build_level_determinism() -> None:
    key = jax.random.PRNGKey(42)
    assert _levels_equal(build_easy_rocket_level(key), build_easy_rocket_level(key))


def test_build_level_keys_vary() -> None:
    base = build_easy_rocket_level(jax.random.PRNGKey(0))
    found_difference = False
    for seed in (1, 2, 3, 4, 5):
        other = build_easy_rocket_level(jax.random.PRNGKey(seed))
        if not np.array_equal(base.block_map, other.block_map):
            found_difference = True
            break
    assert found_difference


def test_build_level_has_all_ore_types() -> None:
    level = build_easy_rocket_level(jax.random.PRNGKey(7))
    present = {int(b) for b in np.unique(level.block_map).tolist()}
    assert _ORE_BLOCKS.issubset(present)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_build_level_patches_avoid_spawn(seed: int) -> None:
    level = build_easy_rocket_level(jax.random.PRNGKey(seed))
    sx, sy = _SPAWN
    for ty in range(sy - _FORBID_RADIUS, sy + _FORBID_RADIUS + 1):
        for tx in range(sx - _FORBID_RADIUS, sx + _FORBID_RADIUS + 1):
            block = int(level.block_map[ty, tx])
            assert block not in _ORE_BLOCKS, (
                f"Patch overlaps spawn zone at ({tx}, {ty}); seed={seed}"
            )
