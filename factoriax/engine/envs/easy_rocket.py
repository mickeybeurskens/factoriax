"""Easy Rocket scenario — small recipe book covering eight machines and the rocket."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp

from factoriax.engine.achievements import (
    Achievement,
    achievement_fn,
    achievement_weights,
    count_machines,
    holds_item,
    max_score,
)
from factoriax.engine.constants import (
    BlockType,
    ItemType,
    Machine,
)
from factoriax.engine.envs.base import FactoriaxEnv, achievement_hook
from factoriax.engine.envs.common import (
    MAP_SIZE,
    ORE_RESOURCES_PER_TILE,
    PATCH_BLOCKS,
    PATCH_SIZE,
    SPAWN,
    producing_miners,
    sample_patch_corner,
    six_patch_terrain,
)
from factoriax.engine.levels import Level, LevelBuilder
from factoriax.engine.recipes import Recipe, RecipeBook, RecipeTable
from factoriax.engine.rewards import achievement_reward
from factoriax.engine.state import EnvParams


def build_easy_rocket_level(key: jax.Array) -> Level:
    """Build a 16x16 easy-rocket level with six 2x2 ore patches placed by PRNG.

    Patches do not overlap each other, stay clear of the outer 1-cell dirt
    ring, and stay clear of the 4x4 zone covering the 2x2 spawn area plus its
    1-cell inner-ring buffer. Same key returns equal Levels; different keys
    produce different layouts. The JAX-native equivalent is
    :func:`factoriax.engine.envs.common.six_patch_terrain`.

    Parameters
    ----------
    key: jax.Array :


    Returns
    -------

    """
    builder = LevelBuilder(MAP_SIZE, MAP_SIZE)
    placed: list[tuple[int, int]] = []
    patch_keys = jax.random.split(key, len(PATCH_BLOCKS))
    for patch_key, block in zip(patch_keys, PATCH_BLOCKS, strict=True):
        corner = sample_patch_corner(patch_key, placed)
        placed.append(corner)
        builder.fill_rect(
            corner[0],
            corner[1],
            PATCH_SIZE,
            PATCH_SIZE,
            block,
            resources=ORE_RESOURCES_PER_TILE,
        )
    builder.set_player_position(*SPAWN)
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


def _producing_ore_presence(state: EnvState) -> jax.Array:
    """Per-ore-block presence: True where a producing miner sits on it.

    Parameters
    ----------
    state: EnvState :


    Returns
    -------

    """
    producing, blocks = producing_miners(state)
    return jnp.stack(
        [jnp.any(producing & (blocks == ore_block)) for ore_block in _ORE_BLOCKS]
    )


def _distinct_producing_ore_types(state: EnvState) -> jax.Array:
    """At least three distinct ore blocks sit under producing miners.

    Parameters
    ----------
    state: EnvState :


    Returns
    -------

    """
    result: jax.Array = jnp.sum(_producing_ore_presence(state).astype(jnp.int32)) >= 3
    return result


def _all_ore_types_covered(state: EnvState) -> jax.Array:
    """Every one of the six raw ores sits under a producing miner.

    Parameters
    ----------
    state: EnvState :


    Returns
    -------

    """
    result: jax.Array = jnp.all(_producing_ore_presence(state))
    return result


def _assembler_holds_inputs(state: EnvState, item_a: int, item_b: int) -> jax.Array:
    """An active assembler holds both ``item_a`` and ``item_b`` in its inputs.

    Each input item must occupy one of the two input slots with a
    non-empty count. Requiring both inputs in the same assembler keeps
    the hull, engine, and rocket feeds distinct despite their shared
    limestone input.

    Parameters
    ----------
    state: EnvState :

    item_a: int :

    item_b: int :


    Returns
    -------

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

    Parameters
    ----------
    state: EnvState :

    item: int :


    Returns
    -------

    """
    is_asm = (state.ent_type == int(Machine.ASSEMBLER)) & (state.ent_y >= 0)
    out_has = (state.ent_asm_out_type == item) & (state.ent_asm_out_count > 0)
    buf_has = (state.ent_buf_type == item) & (state.ent_buf_count > 0)
    return jnp.any(is_asm & (out_has | buf_has))


def _has_any_raw_ore(state: EnvState) -> jax.Array:
    """Player holds at least one of any raw ore type.

    Parameters
    ----------
    state: EnvState :


    Returns
    -------

    """
    return jnp.any(jnp.stack([holds_item(state, item) for item in _RAW_ORE_ITEMS]))


def _has_each_raw_ore(state: EnvState) -> jax.Array:
    """Player holds at least one of every raw ore type.

    Parameters
    ----------
    state: EnvState :


    Returns
    -------

    """
    return jnp.all(jnp.stack([holds_item(state, item) for item in _RAW_ORE_ITEMS]))


def _any_producing_miner(state: EnvState) -> jax.Array:
    """At least one placed miner has ore in its output buffer.

    Parameters
    ----------
    state: EnvState :


    Returns
    -------

    """
    return jnp.any(producing_miners(state)[0])


def _has_machine(state: EnvState, machine: int) -> jax.Array:
    """At least one machine of ``machine`` type is placed on the map.

    Parameters
    ----------
    state: EnvState :

    machine: int :


    Returns
    -------

    """
    return count_machines(state, machine) >= 1


def _has_n_machines(state: EnvState, machine: int, n: int) -> jax.Array:
    """At least ``n`` machines of ``machine`` type are placed on the map.

    Parameters
    ----------
    state: EnvState :

    machine: int :

    n: int :


    Returns
    -------

    """
    return count_machines(state, machine) >= n


def _has_n_machines_pair(
    state: EnvState, machine_a: int, machine_b: int, n: int
) -> jax.Array:
    """At least ``n`` of ``machine_a`` AND at least ``n`` of ``machine_b``.

    Parameters
    ----------
    state: EnvState :

    machine_a: int :

    machine_b: int :

    n: int :


    Returns
    -------

    """
    return (count_machines(state, machine_a) >= n) & (
        count_machines(state, machine_b) >= n
    )


def _has_n_raw_ore_types(state: EnvState, n: int) -> jax.Array:
    """Player inventory holds at least one of ``n`` distinct raw ore types.

    Sibling of :func:`_has_any_raw_ore` (n=1) and
    :func:`_has_each_raw_ore` (n=len(_RAW_ORE_ITEMS)); use this when
    you want a mid-curriculum variety milestone such as "half the ore
    types collected".

    Parameters
    ----------
    state: EnvState :

    n: int :


    Returns
    -------

    """
    held_types = jnp.stack([holds_item(state, item) for item in _RAW_ORE_ITEMS])
    return jnp.sum(held_types.astype(jnp.int32)) >= n


#: Achievement bits in curriculum order, each paired with a stable name. This
#: is the single source of truth for the bit set: the condition tuple, the
#: public name list, and :data:`NUM_EASY_ROCKET_ACHIEVEMENTS` all derive from
#: it. Hand-skill bits read the player inventory; production bits read machine
#: buffers, so hand crafting cannot unlock them. The four sections are raw ore,
#: hulls, engines, and final assembly.
EASY_ROCKET_ACHIEVEMENTS: tuple[Achievement, ...] = (
    # ---- Bootstrap ----
    Achievement("mine_1_ore", _has_any_raw_ore),
    Achievement("mine_3_ore", partial(_has_n_raw_ore_types, n=3)),
    Achievement("prospector", _has_each_raw_ore),
    Achievement("place_miner", partial(_has_machine, machine=int(Machine.MINER))),
    Achievement(
        "place_3_miners",
        partial(_has_n_machines, machine=int(Machine.MINER), n=3),
    ),
    Achievement(
        "metal_in_motion",
        partial(_has_n_machines, machine=int(Machine.MINER), n=6),
    ),
    # ---- Miners up ----
    Achievement("automated_mining", _any_producing_miner),
    Achievement("ore_fields", _distinct_producing_ore_types),
    Achievement("mining_master", _all_ore_types_covered),
    # ---- Storage ----
    Achievement(
        "one_pallet", partial(_has_n_machines, machine=int(Machine.PALLET), n=1)
    ),
    Achievement(
        "three_pallets", partial(_has_n_machines, machine=int(Machine.PALLET), n=3)
    ),
    Achievement("stacked", partial(_has_n_machines, machine=int(Machine.PALLET), n=6)),
    # ---- Hull ----
    Achievement(
        "assembler_online", partial(_has_machine, machine=int(Machine.ASSEMBLER))
    ),
    Achievement(
        "assembler_and_arm",
        partial(
            _has_n_machines_pair,
            machine_a=int(Machine.ASSEMBLER),
            machine_b=int(Machine.ARM),
            n=1,
        ),
    ),
    Achievement(
        "one_belt",
        partial(_has_n_machines, machine=int(Machine.CONVEYOR_BELT), n=1),
    ),
    Achievement(
        "three_belts",
        partial(_has_n_machines, machine=int(Machine.CONVEYOR_BELT), n=3),
    ),
    Achievement(
        "hull_production", partial(_assembler_outputs_item, item=int(ItemType.HULL))
    ),
    # ---- Engine ----
    Achievement(
        "two_assemblers",
        partial(_has_n_machines, machine=int(Machine.ASSEMBLER), n=2),
    ),
    Achievement(
        "an_arm_and_a_leg",
        partial(
            _has_n_machines_pair,
            machine_a=int(Machine.ASSEMBLER),
            machine_b=int(Machine.ARM),
            n=2,
        ),
    ),
    Achievement(
        "six_belts",
        partial(_has_n_machines, machine=int(Machine.CONVEYOR_BELT), n=6),
    ),
    Achievement(
        "engine_production",
        partial(_assembler_outputs_item, item=int(ItemType.ENGINE_UNIT)),
    ),
    # ---- Rocket production ----
    Achievement(
        "auto_bots", partial(_has_n_machines, machine=int(Machine.ASSEMBLER), n=3)
    ),
    Achievement(
        "belt_spaghetti",
        partial(_has_n_machines, machine=int(Machine.CONVEYOR_BELT), n=10),
    ),
    Achievement(
        "rocket_assembled", partial(_assembler_outputs_item, item=int(ItemType.ROCKET))
    ),
    Achievement("liftoff", partial(_has_machine, machine=int(Machine.ROCKET))),
)

#: Stable per-bit names in curriculum order, for display and logging.
#: These double as W&B metric keys in ``baselines/easy_rocket`` — treat
#: them as a wire format, same as the bit order itself.
EASY_ROCKET_ACHIEVEMENT_NAMES: tuple[str, ...] = tuple(
    a.name for a in EASY_ROCKET_ACHIEVEMENTS
)

NUM_EASY_ROCKET_ACHIEVEMENTS: int = len(EASY_ROCKET_ACHIEVEMENTS)

#: Bits walk a four-section production curriculum: raw ore, hulls, engines,
#: and final assembly. Automated-production bits read machine-internal
#: buffers, which only the simulation fills; hand actions deposit into the
#: player inventory, so those bits cannot be unlocked by hand crafting.
easy_rocket_conditions = achievement_fn(EASY_ROCKET_ACHIEVEMENTS)

EASY_ROCKET_ACHIEVEMENT_WEIGHTS: jax.Array = achievement_weights(
    EASY_ROCKET_ACHIEVEMENTS
)

MAX_EASY_ROCKET_SCORE: float = max_score(EASY_ROCKET_ACHIEVEMENTS)


def easy_rocket_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """

    Parameters
    ----------
    prev_state: EnvState :

    new_state: EnvState :

    params: EnvParams :


    Returns
    -------

    """
    return achievement_reward(
        prev_state, new_state, params, weights=EASY_ROCKET_ACHIEVEMENT_WEIGHTS
    )


def easy_rocket(
    *,
    obs: str = "superficial_global",
    obs_radius: int = 7,
) -> tuple[FactoriaxEnv, EnvParams]:
    """Build the EasyRocket-v1 env with achievement hook and reward bound.

    Binds the keyed generator as ``reset_fn`` (fresh layout each episode),
    the production achievement conditions as a step hook, and the achievement
    reward. ``max_machines`` is 100 to avoid overflowing entity arrays on a
    fully-built 16x16 factory (~80 entities at peak).

    Parameters
    ----------
    obs :
        Observation variant passed to :class:`FactoriaxEnv`.
    obs_radius :
        Half-width of the local observation window.
    """
    env = FactoriaxEnv(
        terrain_fn=six_patch_terrain,
        step_hooks=(achievement_hook(easy_rocket_conditions),),
        reward_fn=easy_rocket_reward,
        obs=obs,
        obs_radius=obs_radius,
        map_width=MAP_SIZE,
        map_height=MAP_SIZE,
        num_players=1,
        max_machines=100,
    )
    params = EnvParams(
        max_timesteps=2000,
        recipe_table=EASY_ROCKET_RECIPE_TABLE,
        base_resources=ORE_RESOURCES_PER_TILE,
    )
    return env, params
