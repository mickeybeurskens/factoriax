"""Easy Rocket scenario — small recipe book covering eight machines and the rocket."""

from __future__ import annotations

import jax

from factoriax.constants import BlockType, ItemType
from factoriax.levels import Level, LevelBuilder
from factoriax.recipes import Recipe, RecipeBook, RecipeTable

_MAP_SIZE: int = 16
_SPAWN: tuple[int, int] = (_MAP_SIZE // 2, _MAP_SIZE // 2)
_PATCH_SIZE: int = 2
_FORBID_RADIUS: int = 1
_ORE_RESOURCES_PER_TILE: int = 3000
_MAX_SAMPLE_ATTEMPTS: int = 1000

_PATCH_BLOCKS: tuple[BlockType, ...] = (
    BlockType.IRON,
    BlockType.COPPER,
    BlockType.TIN,
    BlockType.SILICON,
    BlockType.COAL,
    BlockType.LIMESTONE,
)


def _patch_touches_spawn_zone(px: int, py: int) -> bool:
    sx, sy = _SPAWN
    for dy in range(_PATCH_SIZE):
        for dx in range(_PATCH_SIZE):
            tx, ty = px + dx, py + dy
            if abs(tx - sx) <= _FORBID_RADIUS and abs(ty - sy) <= _FORBID_RADIUS:
                return True
    return False


def _patches_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) < _PATCH_SIZE and abs(a[1] - b[1]) < _PATCH_SIZE


def _sample_patch_corner(
    key: jax.Array, placed: list[tuple[int, int]]
) -> tuple[int, int]:
    max_corner = _MAP_SIZE - _PATCH_SIZE
    for _ in range(_MAX_SAMPLE_ATTEMPTS):
        key, subkey = jax.random.split(key)
        coords = jax.random.randint(subkey, shape=(2,), minval=0, maxval=max_corner + 1)
        corner = (int(coords[0]), int(coords[1]))
        if _patch_touches_spawn_zone(*corner):
            continue
        if any(_patches_overlap(corner, other) for other in placed):
            continue
        return corner
    raise RuntimeError(
        f"build_easy_rocket_level: failed to place a patch in "
        f"{_MAX_SAMPLE_ATTEMPTS} attempts."
    )


def build_easy_rocket_level(key: jax.Array) -> Level:
    """Build a 16x16 easy-rocket level with six 2x2 ore patches placed by PRNG.

    Patches do not overlap each other and stay outside the 3x3 ring centered on
    the player spawn at (8, 8). Same key returns equal Levels; different keys
    produce different layouts.
    """
    builder = LevelBuilder(_MAP_SIZE, _MAP_SIZE)
    placed: list[tuple[int, int]] = []
    patch_keys = jax.random.split(key, len(_PATCH_BLOCKS))
    for patch_key, block in zip(patch_keys, _PATCH_BLOCKS, strict=True):
        corner = _sample_patch_corner(patch_key, placed)
        placed.append(corner)
        builder.fill_rect(
            corner[0],
            corner[1],
            _PATCH_SIZE,
            _PATCH_SIZE,
            block,
            resources=_ORE_RESOURCES_PER_TILE,
        )
    builder.set_player_position(*_SPAWN)
    return builder.build("easy_rocket_v1")


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
