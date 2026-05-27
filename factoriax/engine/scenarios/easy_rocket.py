"""Easy Rocket scenario — small recipe book covering eight machines and the rocket."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from factoriax.engine.constants import (
    MAX_ACHIEVEMENTS,
    BlockType,
    ItemType,
    Machine,
)
from factoriax.engine.envs.factoriax_env import FactoriaXEnv
from factoriax.engine.envs.hooks import achievement_hook
from factoriax.engine.levels import Level, LevelBuilder, initial_state
from factoriax.engine.recipes import Recipe, RecipeBook, RecipeTable
from factoriax.engine.rewards import achievement_reward
from factoriax.engine.state import EnvParams, EnvState

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


_MAX_CORNER: int = _MAP_SIZE - _PATCH_SIZE
#: Every 2x2 patch corner that does not touch the spawn zone, computed once.
#: The JAX generator draws non-overlapping patches from this fixed set, so the
#: spawn-avoidance rule matches the host builder exactly.
_VALID_PATCH_CORNERS: np.ndarray = np.array(
    [
        (cx, cy)
        for cx in range(_MAX_CORNER + 1)
        for cy in range(_MAX_CORNER + 1)
        if not _patch_touches_spawn_zone(cx, cy)
    ],
    dtype=np.int32,
)


def _easy_rocket_terrain(key: jax.Array, params: EnvParams) -> jax.Array:
    """Build a dirt map with one non-overlapping 2x2 patch per ore block.

    Jittable, vmappable port of :func:`build_easy_rocket_level`'s placement:
    shuffle the valid (spawn-avoiding) corners with ``key`` and greedily take
    the first ``len(_PATCH_BLOCKS)`` that do not overlap an already-placed
    patch. With ~200 candidates and six patches this always succeeds, so no
    rejection-failure branch is needed.
    """
    corners = jnp.asarray(_VALID_PATCH_CORNERS)
    shuffled = corners[jax.random.permutation(key, corners.shape[0])]
    n_patches = len(_PATCH_BLOCKS)
    placed0 = jnp.full((n_patches, 2), -_PATCH_SIZE, dtype=jnp.int32)

    def place(
        carry: tuple[jax.Array, jax.Array], cand: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array], None]:
        placed, count = carry
        dx = jnp.abs(placed[:, 0] - cand[0])
        dy = jnp.abs(placed[:, 1] - cand[1])
        filled = jnp.arange(n_patches) < count
        overlaps = jnp.any(filled & (dx < _PATCH_SIZE) & (dy < _PATCH_SIZE))
        do_place = (count < n_patches) & ~overlaps
        slot = jnp.minimum(count, n_patches - 1)
        placed = placed.at[slot].set(jnp.where(do_place, cand, placed[slot]))
        return (placed, count + do_place.astype(jnp.int32)), None

    (placed, _count), _ = jax.lax.scan(place, (placed0, jnp.int32(0)), shuffled)

    world = jnp.full((params.map_height, params.map_width), jnp.int8(BlockType.DIRT))
    for i, block in enumerate(_PATCH_BLOCKS):
        patch = jnp.full((_PATCH_SIZE, _PATCH_SIZE), jnp.int8(int(block)))
        world = jax.lax.dynamic_update_slice(world, patch, (placed[i, 1], placed[i, 0]))
    return world


def generate_easy_rocket_state(key: jax.Array, params: EnvParams) -> EnvState:
    """Generate an easy-rocket initial state from a PRNG key.

    JAX-native and JIT/vmap-compatible: the six ore patches are placed from
    ``key`` (see :func:`_easy_rocket_terrain`), then :func:`initial_state`
    assembles the full :class:`EnvState` (player at centre, ore resources from
    ``params.base_resources``, empty machines/inventory). Suitable as a
    scenario ``reset_fn`` so every reset/episode draws a fresh layout.
    """
    return initial_state(_easy_rocket_terrain(key, params), params)


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


_RAW_ORE_ITEMS: tuple[int, ...] = (
    int(ItemType.IRON_ORE),
    int(ItemType.COPPER_ORE),
    int(ItemType.TIN_ORE),
    int(ItemType.SILICON),
    int(ItemType.COAL),
    int(ItemType.LIMESTONE),
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


def _producing_miners(state: EnvState) -> tuple[jax.Array, jax.Array]:
    """Return (producing_mask, block_at_pos) over all entity slots.

    Like :func:`_blocks_under_active_miners` but the mask also requires a
    non-empty output buffer, so a slot counts only once its miner has
    actually mined ore rather than merely being placed on an ore tile.
    """
    active, blocks = _blocks_under_active_miners(state)
    producing = active & (state.ent_buf_count > 0)
    return producing, blocks


def _producing_ore_presence(state: EnvState) -> jax.Array:
    """Per-ore-block presence: True where a producing miner sits on it."""
    producing, blocks = _producing_miners(state)
    return jnp.stack(
        [jnp.any(producing & (blocks == ore_block)) for ore_block in _ORE_BLOCKS]
    )


def _distinct_producing_ore_types(state: EnvState) -> jax.Array:
    """At least three distinct ore blocks sit under producing miners."""
    result: jax.Array = jnp.sum(_producing_ore_presence(state).astype(jnp.int32)) >= 3
    return result


def _all_ore_types_covered(state: EnvState) -> jax.Array:
    """Every one of the six raw ores sits under a producing miner."""
    result: jax.Array = jnp.all(_producing_ore_presence(state))
    return result


def _assembler_holds_inputs(state: EnvState, item_a: int, item_b: int) -> jax.Array:
    """An active assembler holds both ``item_a`` and ``item_b`` in its inputs.

    Each input item must occupy one of the two input slots with a
    non-empty count. Requiring both inputs in the same assembler keeps
    the hull, engine, and rocket feeds distinct despite their shared
    limestone input.
    """
    is_asm = (state.ent_type == int(Machine.ASSEMBLER)) & (state.ent_y >= 0)
    in_type = state.ent_asm_in_type
    in_count = state.ent_asm_in_count
    has_a = jnp.any((in_type == item_a) & (in_count > 0), axis=-1)
    has_b = jnp.any((in_type == item_b) & (in_count > 0), axis=-1)
    return jnp.any(is_asm & has_a & has_b)


def _assembler_outputs_item(state: EnvState, item: int) -> jax.Array:
    """An active assembler carries ``item`` in its output or buffer slot.

    Reads both ``ent_asm_out`` and ``ent_buf`` because the engine drains
    a finished output into the buffer on the next tick; checking only the
    output slot would blink off for that tick.
    """
    is_asm = (state.ent_type == int(Machine.ASSEMBLER)) & (state.ent_y >= 0)
    out_has = (state.ent_asm_out_type == item) & (state.ent_asm_out_count > 0)
    buf_has = (state.ent_buf_type == item) & (state.ent_buf_count > 0)
    return jnp.any(is_asm & (out_has | buf_has))


def _has_any_raw_ore(state: EnvState) -> jax.Array:
    """Player holds at least one of any raw ore type."""
    return jnp.any(jnp.stack([_holds_item(state, item) for item in _RAW_ORE_ITEMS]))


def _has_each_raw_ore(state: EnvState) -> jax.Array:
    """Player holds at least one of every raw ore type."""
    return jnp.all(jnp.stack([_holds_item(state, item) for item in _RAW_ORE_ITEMS]))


def _any_producing_miner(state: EnvState) -> jax.Array:
    """At least one placed miner has ore in its output buffer."""
    return jnp.any(_producing_miners(state)[0])


def _has_machine(state: EnvState, machine: int) -> jax.Array:
    """At least one machine of ``machine`` type is placed on the map."""
    return _count_machines(state, machine) >= 1


#: Achievement bits in curriculum order, each paired with a stable name. This
#: is the single source of truth for the bit set: the condition tuple, the
#: public name list, and :data:`NUM_EASY_ROCKET_ACHIEVEMENTS` all derive from
#: it. Hand-skill bits read the player inventory; production bits read machine
#: buffers, so hand crafting cannot unlock them. The four sections are raw ore,
#: hulls, engines, and final assembly.
_EASY_ROCKET_ACHIEVEMENTS: tuple[tuple[str, Callable[..., jax.Array]], ...] = (
    ("mine_1_ore", _has_any_raw_ore),
    ("prospector", _has_each_raw_ore),
    ("craft_miner", partial(_holds_item, item=int(ItemType.MINER))),
    ("automated_mining", _any_producing_miner),
    ("ore_fields", _distinct_producing_ore_types),
    ("full_supply", _all_ore_types_covered),
    ("assembler_online", partial(_has_machine, machine=int(Machine.ASSEMBLER))),
    (
        "hull_line_fed",
        partial(
            _assembler_holds_inputs,
            item_a=int(ItemType.IRON_ORE),
            item_b=int(ItemType.LIMESTONE),
        ),
    ),
    ("hull_production", partial(_assembler_outputs_item, item=int(ItemType.HULL))),
    (
        "engine_line_fed",
        partial(
            _assembler_holds_inputs,
            item_a=int(ItemType.COPPER_ORE),
            item_b=int(ItemType.LIMESTONE),
        ),
    ),
    (
        "engine_production",
        partial(_assembler_outputs_item, item=int(ItemType.ENGINE_UNIT)),
    ),
    (
        "rocket_line_fed",
        partial(
            _assembler_holds_inputs,
            item_a=int(ItemType.HULL),
            item_b=int(ItemType.ENGINE_UNIT),
        ),
    ),
    ("rocket_assembled", partial(_assembler_outputs_item, item=int(ItemType.ROCKET))),
    ("liftoff", partial(_has_machine, machine=int(Machine.ROCKET))),
)

#: Stable per-bit names in curriculum order, for display and logging.
EASY_ROCKET_ACHIEVEMENT_NAMES: tuple[str, ...] = tuple(
    name for name, _ in _EASY_ROCKET_ACHIEVEMENTS
)

_EASY_ROCKET_CONDITIONS: tuple[Callable[..., jax.Array], ...] = tuple(
    condition for _, condition in _EASY_ROCKET_ACHIEVEMENTS
)

NUM_EASY_ROCKET_ACHIEVEMENTS: int = len(_EASY_ROCKET_ACHIEVEMENTS)


def easy_rocket_conditions(state: EnvState) -> jax.Array:
    """Compute the easy-rocket achievement bits, zero-padded to MAX_ACHIEVEMENTS.

    The bits walk a four-section production curriculum: raw ore, hulls,
    engines, and final assembly. Automated-production bits read
    machine-internal buffers, which only the simulation fills; hand actions
    deposit into the player inventory, so those bits cannot be unlocked by
    hand crafting.
    """
    conditions = jnp.stack([condition(state) for condition in _EASY_ROCKET_CONDITIONS])
    padding = jnp.zeros(MAX_ACHIEVEMENTS - conditions.shape[0], dtype=jnp.bool_)
    return jnp.concatenate([conditions, padding])


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


def easy_rocket() -> tuple[FactoriaXEnv, EnvParams]:
    """Return the easy-rocket env (keyed procgen reset) and its params.

    Binds the keyed generator as ``reset_fn`` (a fresh layout per reset), the
    production achievement conditions as a step hook, and the achievement
    reward. Loaded via ``factoriax.make("EasyRocket-v1")``. The 16x16 factory
    runs to ~80 entities, so ``max_machines`` is budgeted to 100 (above the
    auto default of 64) to avoid overflowing the entity arrays mid-build.
    """
    env = FactoriaXEnv(
        reset_fn=generate_easy_rocket_state,
        step_hooks=(achievement_hook(easy_rocket_conditions),),
        reward_fn=easy_rocket_reward,
    )
    params = EnvParams(
        map_width=_MAP_SIZE,
        map_height=_MAP_SIZE,
        num_players=1,
        max_timesteps=2000,
        max_machines=100,
        recipe_table=EASY_ROCKET_RECIPE_TABLE,
        base_resources=_ORE_RESOURCES_PER_TILE,
    )
    return env, params
