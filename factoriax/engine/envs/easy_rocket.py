"""The Easy Rocket scenario: a small recipe book for eight machines and a rocket."""

from __future__ import annotations

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
from factoriax.engine.state import EnvParams, EnvState


def build_easy_rocket_level(key: jax.Array) -> Level:
    """Build a 16x16 easy-rocket level with six 2x2 ore patches from a PRNG.

    The patches do not overlap each other. They stay clear of the outer
    one-cell dirt ring, and clear of the 4x4 zone that covers the 2x2 spawn
    area and its one-cell inner ring. The same key returns equal levels, and
    two different keys give two different layouts.
    :func:`factoriax.engine.envs.common.six_patch_terrain` is the JAX-native
    version of this function.

    Parameters
    ----------
    key
        PRNG key for the patch placement. The function runs on the host, so
        JAX cannot trace it. Call it before you enter ``jit``.

    Returns
    -------
    Level
        A 16x16 level with six 2x2 ore patches. The same key always returns an
        equal level.
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
    """Report, for each ore block, whether a producing miner stands on it.

    Parameters
    ----------
    state
        State to read.

    Returns
    -------
    jax.Array
        Shape ``(6,)``, bool. There is one flag for each raw ore block, in
        ``_ORE_BLOCKS`` order. A flag is True where a producing miner stands
        on that ore.
    """
    producing, blocks = producing_miners(state)
    return jnp.stack(
        [jnp.any(producing & (blocks == ore_block)) for ore_block in _ORE_BLOCKS]
    )


def _distinct_producing_ore_types(state: EnvState) -> jax.Array:
    """Test whether producing miners stand on three different ore blocks or more.

    Parameters
    ----------
    state
        State to read.

    Returns
    -------
    jax.Array
        Scalar bool. The value is True when three or more of the six raw ores
        each hold a producing miner.
    """
    result: jax.Array = jnp.sum(_producing_ore_presence(state).astype(jnp.int32)) >= 3
    return result


def _all_ore_types_covered(state: EnvState) -> jax.Array:
    """Test whether a producing miner stands on each of the six raw ores.

    Parameters
    ----------
    state
        State to read.

    Returns
    -------
    jax.Array
        Scalar bool.
    """
    result: jax.Array = jnp.all(_producing_ore_presence(state))
    return result


def _assembler_holds_inputs(state: EnvState, item_a: int, item_b: int) -> jax.Array:
    """Report whether one assembler holds both items in its input slots.

    Each input item must sit in one of the two input slots, with a count above
    zero. The test needs both inputs in the same assembler. The hull feed, the
    engine feed, and the rocket feed therefore stay separate, although all
    three take limestone.

    Parameters
    ----------
    state
        State to read.
    item_a
        First required input item id.
    item_b
        Second required input item id.

    Returns
    -------
    jax.Array
        Scalar bool. The value is True when one placed assembler holds both
        items, in either input slot.
    """
    is_asm = (state.ent_type == int(Machine.ASSEMBLER)) & (state.ent_y >= 0)
    in_type = state.ent_asm_in_type
    in_count = state.ent_asm_in_count
    has_a = jnp.any((in_type == item_a) & (in_count > 0), axis=-1)
    has_b = jnp.any((in_type == item_b) & (in_count > 0), axis=-1)
    return jnp.any(is_asm & has_a & has_b)


def _assembler_outputs_item(state: EnvState, item: int) -> jax.Array:
    """Report whether one assembler holds an item that is on its way out.

    The function reads both ``ent_asm_out`` and ``ent_buf``, because the engine
    moves a finished output into the buffer on the next tick. A test of the
    output slot alone therefore returns False for that one tick.

    Parameters
    ----------
    state
        State to read.
    item
        Item id to look for.

    Returns
    -------
    jax.Array
        Scalar bool.
    """
    is_asm = (state.ent_type == int(Machine.ASSEMBLER)) & (state.ent_y >= 0)
    out_has = (state.ent_asm_out_type == item) & (state.ent_asm_out_count > 0)
    buf_has = (state.ent_buf_type == item) & (state.ent_buf_count > 0)
    return jnp.any(is_asm & (out_has | buf_has))


def _has_any_raw_ore(state: EnvState) -> jax.Array:
    """Test whether the player holds one unit or more of any raw ore type.

    Parameters
    ----------
    state
        State to read.

    Returns
    -------
    jax.Array
        Scalar bool.
    """
    return jnp.any(jnp.stack([holds_item(state, item) for item in _RAW_ORE_ITEMS]))


def _has_each_raw_ore(state: EnvState) -> jax.Array:
    """Test whether the player holds one unit or more of every raw ore type.

    Parameters
    ----------
    state
        State to read.

    Returns
    -------
    jax.Array
        Scalar bool.
    """
    return jnp.all(jnp.stack([holds_item(state, item) for item in _RAW_ORE_ITEMS]))


def _any_producing_miner(state: EnvState) -> jax.Array:
    """Test whether one placed miner or more holds ore in its output buffer.

    Parameters
    ----------
    state
        State to read.

    Returns
    -------
    jax.Array
        Scalar bool.
    """
    return jnp.any(producing_miners(state)[0])


def _has_machine(state: EnvState, machine: int) -> jax.Array:
    """Test whether the map holds one machine of type ``machine`` or more.

    Parameters
    ----------
    state
        State to read.
    machine
        ``Machine`` value to count.

    Returns
    -------
    jax.Array
        Scalar bool.
    """
    return count_machines(state, machine) >= 1


def _has_n_machines(state: EnvState, machine: int, n: int) -> jax.Array:
    """Test whether the map holds ``n`` machines of type ``machine`` or more.

    Parameters
    ----------
    state
        State to read.
    machine
        ``Machine`` value to count.
    n
        Threshold. A count equal to ``n`` passes.

    Returns
    -------
    jax.Array
        Scalar bool.
    """
    return count_machines(state, machine) >= n


def _has_n_machines_pair(
    state: EnvState, machine_a: int, machine_b: int, n: int
) -> jax.Array:
    """Test for ``n`` or more of ``machine_a``, and ``n`` or more of ``machine_b``.

    Parameters
    ----------
    state
        State to read.
    machine_a
        First ``Machine`` value to count.
    machine_b
        Second ``Machine`` value to count.
    n
        Threshold for each kind on its own. A count equal to ``n`` passes.

    Returns
    -------
    jax.Array
        Scalar bool.
    """
    return (count_machines(state, machine_a) >= n) & (
        count_machines(state, machine_b) >= n
    )


def _has_n_raw_ore_types(state: EnvState, n: int) -> jax.Array:
    """Test whether the player inventory holds ``n`` different raw ore types.

    This function generalises :func:`_has_any_raw_ore`, where ``n`` is 1, and
    :func:`_has_each_raw_ore`, where ``n`` is ``len(_RAW_ORE_ITEMS)``. Use it
    for a milestone in the middle of a curriculum, such as "half the ore types
    collected".

    Parameters
    ----------
    state
        State to read.
    n
        Number of different raw ore types that the player must hold. A count
        equal to ``n`` passes.

    Returns
    -------
    jax.Array
        Scalar bool.
    """
    held_types = jnp.stack([holds_item(state, item) for item in _RAW_ORE_ITEMS])
    return jnp.sum(held_types.astype(jnp.int32)) >= n


#: The achievement bits in curriculum order, each one with a stable name. This
#: tuple is the one definition of the bit set. The condition tuple, the public
#: name list, and :data:`NUM_EASY_ROCKET_ACHIEVEMENTS` all come from it. A
#: hand-skill bit reads the player inventory. A production bit reads a machine
#: buffer, so a hand craft cannot unlock one. The four sections are raw ore,
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

#: Stable name of each bit, in curriculum order, for the display and the log. A
#: training run writes these as metric keys.
#:
#: CAUTION: Treat these names as a wire format, the same as the bit order. A
#: new name breaks every log that an earlier run wrote.
EASY_ROCKET_ACHIEVEMENT_NAMES: tuple[str, ...] = tuple(
    a.name for a in EASY_ROCKET_ACHIEVEMENTS
)

NUM_EASY_ROCKET_ACHIEVEMENTS: int = len(EASY_ROCKET_ACHIEVEMENTS)

#: The bits walk a production curriculum of four sections: raw ore, hulls,
#: engines, and final assembly. An automated-production bit reads a buffer
#: inside a machine, and only the simulation fills such a buffer. A hand action
#: writes to the player inventory, so a hand craft cannot unlock those bits.
easy_rocket_conditions = achievement_fn(EASY_ROCKET_ACHIEVEMENTS)

EASY_ROCKET_ACHIEVEMENT_WEIGHTS: jax.Array = achievement_weights(
    EASY_ROCKET_ACHIEVEMENTS
)

MAX_EASY_ROCKET_SCORE: float = max_score(EASY_ROCKET_ACHIEVEMENTS)


def easy_rocket_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Score a step by the EasyRocket achievements that it unlocked.

    Parameters
    ----------
    prev_state
        State just before the step.
    new_state
        State just after the step.
    params
        The function does not read this argument. It is present for the shared
        reward signature.

    Returns
    -------
    jax.Array
        Scalar float32, the weighted sum of the achievements that this step
        unlocked. The value is zero on most steps.
    """
    return achievement_reward(
        prev_state, new_state, params, weights=EASY_ROCKET_ACHIEVEMENT_WEIGHTS
    )


def easy_rocket(
    *,
    obs: str = "superficial_global",
    obs_radius: int = 7,
) -> tuple[FactoriaxEnv, EnvParams]:
    """Build the EasyRocket-v1 environment, with its achievement hook and reward.

    The function binds the keyed generator as ``terrain_fn``, so each episode
    gets a new layout. It binds the production achievement conditions as a step
    hook, and the achievement reward as ``reward_fn``. ``max_machines`` is 100,
    which keeps the entity arrays large enough for a full 16x16 factory. Such a
    factory holds about 80 entities at its largest.

    Parameters
    ----------
    obs :
        Observation variant. The function passes it to :class:`FactoriaxEnv`.
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
