"""Easy Rocket scenario — small recipe book covering eight machines and the rocket."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from factoriax.constants import (
    MAX_ACHIEVEMENTS,
    BlockType,
    ItemType,
    Machine,
)
from factoriax.levels import Level, LevelBuilder
from factoriax.recipes import Recipe, RecipeBook, RecipeTable
from factoriax.rewards import achievement_reward
from factoriax.scenarios.core import LevelResult, ScenarioLevel
from factoriax.state import EnvParams, EnvState

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
        inputs=((int(ItemType.LIMESTONE), 1), (int(ItemType.SILICON), 1)),
        ticks=6,
        name="Miner",
    ),
    Recipe(
        output=int(ItemType.ASSEMBLER),
        inputs=((int(ItemType.COAL), 1), (int(ItemType.SILICON), 1)),
        ticks=8,
        name="Assembler",
    ),
    Recipe(
        output=int(ItemType.CONVEYOR_BELT),
        inputs=((int(ItemType.COAL), 1), (int(ItemType.IRON_ORE), 1)),
        ticks=4,
        name="Conveyor Belt",
    ),
    Recipe(
        output=int(ItemType.SPLITTER),
        inputs=((int(ItemType.COPPER_ORE), 1), (int(ItemType.IRON_ORE), 1)),
        ticks=4,
        name="Splitter",
    ),
    Recipe(
        output=int(ItemType.CROSSING),
        inputs=((int(ItemType.TIN_ORE), 1), (int(ItemType.IRON_ORE), 1)),
        ticks=4,
        name="Crossing",
    ),
    Recipe(
        output=int(ItemType.ARM),
        inputs=((int(ItemType.TIN_ORE), 1), (int(ItemType.COPPER_ORE), 1)),
        ticks=4,
        name="Arm",
    ),
    Recipe(
        output=int(ItemType.PALLET),
        inputs=((int(ItemType.SILICON), 1), (int(ItemType.IRON_ORE), 1)),
        ticks=4,
        name="Pallet",
    ),
    Recipe(
        output=int(ItemType.HULL),
        inputs=((int(ItemType.IRON_ORE), 2), (int(ItemType.LIMESTONE), 2)),
        ticks=8,
        name="Hull",
    ),
    Recipe(
        output=int(ItemType.ENGINE_UNIT),
        inputs=((int(ItemType.COPPER_ORE), 2), (int(ItemType.LIMESTONE), 1)),
        ticks=8,
        name="Engine Unit",
    ),
    Recipe(
        output=int(ItemType.ROCKET),
        inputs=((int(ItemType.HULL), 100), (int(ItemType.ENGINE_UNIT), 100)),
        ticks=5,
        name="Rocket",
    ),
)

EASY_ROCKET_RECIPE_BOOK: RecipeBook = RecipeBook(recipes=EASY_ROCKET_RECIPES)

EASY_ROCKET_RECIPE_TABLE: RecipeTable = RecipeTable.from_book(EASY_ROCKET_RECIPE_BOOK)


NUM_EASY_ROCKET_ACHIEVEMENTS: int = 13

_RAW_ORE_ITEMS: tuple[int, ...] = (
    int(ItemType.IRON_ORE),
    int(ItemType.COPPER_ORE),
    int(ItemType.TIN_ORE),
    int(ItemType.SILICON),
    int(ItemType.COAL),
    int(ItemType.LIMESTONE),
)

# Ores needed to craft a miner: trace MINER <- IRON_PLATE + WIRE, where
# IRON_PLATE <- IRON_ORE + COAL and WIRE <- COPPER_PLATE + TIN_PLATE.
_MINER_CRAFT_ORES: tuple[int, ...] = (
    int(ItemType.IRON_ORE),
    int(ItemType.COPPER_ORE),
    int(ItemType.TIN_ORE),
    int(ItemType.COAL),
)

_ORE_BLOCKS: tuple[int, ...] = (
    int(BlockType.IRON),
    int(BlockType.COPPER),
    int(BlockType.TIN),
    int(BlockType.SILICON),
    int(BlockType.COAL),
    int(BlockType.LIMESTONE),
)


def _holds_item(state: EnvState, item: int) -> jax.Array:
    return jnp.sum(state.player_inventory[:, item]) >= 1


def _holds_at_least(state: EnvState, item: int, threshold: int) -> jax.Array:
    return jnp.sum(state.player_inventory[:, item]) >= threshold


def _count_machines(state: EnvState, machine_type: int) -> jax.Array:
    return jnp.sum(state.machine_types == machine_type)


def _blocks_under_active_miners(state: EnvState) -> tuple[jax.Array, jax.Array]:
    """Return (active_mask, block_at_pos) over all entity slots.

    ``active_mask`` is True for slots that hold an active miner. ``block_at_pos``
    is the block under the entity's ``(ent_y, ent_x)`` tile, computed with
    clamped indices so inactive slots stay JIT-safe.
    """
    active = (state.ent_type == int(Machine.MINER)) & (state.ent_y >= 0)
    safe_y = jnp.maximum(state.ent_y, 0)
    safe_x = jnp.maximum(state.ent_x, 0)
    blocks = state.map[safe_y, safe_x]
    return active, blocks


def _any_miner_on_ore(state: EnvState) -> jax.Array:
    active, blocks = _blocks_under_active_miners(state)
    is_ore = jnp.zeros_like(blocks, dtype=jnp.bool_)
    for ore_block in _ORE_BLOCKS:
        is_ore = is_ore | (blocks == ore_block)
    return jnp.any(active & is_ore)


def _distinct_ore_types_under_miners(state: EnvState) -> jax.Array:
    active, blocks = _blocks_under_active_miners(state)
    presence = jnp.stack(
        [jnp.any(active & (blocks == ore_block)) for ore_block in _ORE_BLOCKS]
    )
    result: jax.Array = jnp.sum(presence.astype(jnp.int32)) >= 3
    return result


def easy_rocket_conditions(state: EnvState) -> jax.Array:
    """Compute the 13 easy-rocket achievement bits, zero-padded to MAX_ACHIEVEMENTS.

    The four belt-network achievements (indices 7, 9, 10, 11) are stubs that
    always read False; they unlock once the entity connection graph lands.
    """
    has_any_raw_ore = jnp.any(
        jnp.stack([_holds_item(state, item) for item in _RAW_ORE_ITEMS])
    )
    has_each_raw_ore = jnp.all(
        jnp.stack([_holds_item(state, item) for item in _RAW_ORE_ITEMS])
    )
    has_ten_of_each_miner_craft_ore = jnp.all(
        jnp.stack([_holds_at_least(state, item, 10) for item in _MINER_CRAFT_ORES])
    )

    has_miner_in_inventory = _holds_item(state, int(ItemType.MINER))
    has_assembler_in_inventory = _holds_item(state, int(ItemType.ASSEMBLER))
    has_belt_in_inventory = _holds_item(state, int(ItemType.CONVEYOR_BELT))

    miner_on_ore = _any_miner_on_ore(state)
    three_ore_types = _distinct_ore_types_under_miners(state)
    rocket_placed = _count_machines(state, int(Machine.ROCKET)) >= 1

    stub = jnp.bool_(False)
    conditions = jnp.stack(
        [
            has_any_raw_ore,
            has_each_raw_ore,
            has_ten_of_each_miner_craft_ore,
            has_miner_in_inventory,
            has_assembler_in_inventory,
            has_belt_in_inventory,
            miner_on_ore,
            stub,
            three_ore_types,
            stub,
            stub,
            stub,
            rocket_placed,
        ]
    )
    return jnp.concatenate(
        [
            conditions,
            jnp.zeros(MAX_ACHIEVEMENTS - NUM_EASY_ROCKET_ACHIEVEMENTS, dtype=jnp.bool_),
        ]
    )


EASY_ROCKET_ACHIEVEMENT_WEIGHTS: jax.Array = (
    jnp.zeros(MAX_ACHIEVEMENTS, dtype=jnp.float32)
    .at[:NUM_EASY_ROCKET_ACHIEVEMENTS]
    .set(1.0)
)

MAX_EASY_ROCKET_SCORE: float = float(NUM_EASY_ROCKET_ACHIEVEMENTS)


def easy_rocket_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    return achievement_reward(
        prev_state, new_state, params, weights=EASY_ROCKET_ACHIEVEMENT_WEIGHTS
    )


class EasyRocketScenario:
    """16x16 rocket scenario with 2000-step budget for fast RL iteration."""

    name: str = "easy_rocket"
    num_players: int = 1
    achievement_fn = staticmethod(easy_rocket_conditions)
    blocked_actions: frozenset[int] = frozenset()

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed

    def levels(self) -> list[ScenarioLevel]:
        params = EnvParams(
            max_timesteps=2000,
            map_width=_MAP_SIZE,
            map_height=_MAP_SIZE,
            num_players=1,
            recipe_table=EASY_ROCKET_RECIPE_TABLE,
            # The full automated factory runs to ~80 entities, above the
            # auto default of max(64, area // 4) = 64 for a 16x16 map, so
            # the rocket section would overflow the entity arrays
            # mid-build. Budget the whole factory with headroom.
            max_machines=100,
        )
        return [
            ScenarioLevel(
                name="easy_rocket_v1",
                description=(
                    "Build a rocket from raw ore on a 16x16 procgen map. "
                    "Episode ends at T=2000; score is the unweighted sum of "
                    "unlocked achievements."
                ),
                level=build_easy_rocket_level(jax.random.PRNGKey(self._seed)),
                env_params=params,
            )
        ]

    def score_level(
        self, scenario_level: ScenarioLevel, items_mined: dict[str, int]
    ) -> float:
        return 0.0

    def score(self, level_results: list[LevelResult]) -> float:
        weights = jnp.asarray(EASY_ROCKET_ACHIEVEMENT_WEIGHTS)
        scores: list[float] = []
        for result in level_results:
            mask = result.achievements_unlocked
            if mask is None:
                scores.append(0.0)
                continue
            scores.append(float(jnp.sum(weights * jnp.asarray(mask))))
        if not scores:
            return 0.0
        return sum(scores) / len(scores)
