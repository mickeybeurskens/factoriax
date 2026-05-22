"""Unit tests for the easy-rocket scenario."""

from __future__ import annotations

from factoriax.constants import ItemType
from factoriax.recipes import RecipeBook, RecipeTable
from factoriax.scenarios.easy_rocket import (
    EASY_ROCKET_RECIPE_BOOK,
    EASY_ROCKET_RECIPE_TABLE,
)

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
