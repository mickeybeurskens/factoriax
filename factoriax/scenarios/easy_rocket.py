"""Easy Rocket scenario — small recipe book covering eight machines and the rocket."""

from __future__ import annotations

from factoriax.constants import ItemType
from factoriax.recipes import Recipe, RecipeBook, RecipeTable

EASY_ROCKET_RECIPES: tuple[Recipe, ...] = (
    Recipe(
        output=int(ItemType.MINER),
        inputs=((int(ItemType.IRON_PLATE), 1), (int(ItemType.WIRE), 1)),
        ticks=6,
        name="Miner",
    ),
    Recipe(
        output=int(ItemType.ASSEMBLER),
        inputs=((int(ItemType.FRAME), 1), (int(ItemType.CIRCUIT), 1)),
        ticks=8,
        name="Assembler",
    ),
    Recipe(
        output=int(ItemType.CONVEYOR_BELT),
        inputs=((int(ItemType.IRON_PLATE), 1), (int(ItemType.COPPER_PLATE), 1)),
        ticks=4,
        name="Conveyor Belt",
    ),
    Recipe(
        output=int(ItemType.SPLITTER),
        inputs=((int(ItemType.TIN_PLATE), 1), (int(ItemType.COAL), 1)),
        ticks=4,
        name="Splitter",
    ),
    Recipe(
        output=int(ItemType.CROSSING),
        inputs=((int(ItemType.COPPER_PLATE), 1), (int(ItemType.COAL), 1)),
        ticks=4,
        name="Crossing",
    ),
    Recipe(
        output=int(ItemType.HULL),
        inputs=((int(ItemType.FRAME), 2), (int(ItemType.IRON_PLATE), 2)),
        ticks=8,
        name="Hull",
    ),
    Recipe(
        output=int(ItemType.ENGINE_UNIT),
        inputs=((int(ItemType.MOTOR), 2), (int(ItemType.WIRE), 1)),
        ticks=8,
        name="Engine Unit",
    ),
    Recipe(
        output=int(ItemType.ROCKET),
        inputs=((int(ItemType.HULL), 1), (int(ItemType.ENGINE_UNIT), 1)),
        ticks=50,
        name="Rocket",
    ),
)

EASY_ROCKET_RECIPE_BOOK: RecipeBook = RecipeBook(recipes=EASY_ROCKET_RECIPES)

EASY_ROCKET_RECIPE_TABLE: RecipeTable = RecipeTable.from_book(EASY_ROCKET_RECIPE_BOOK)
