"""Level definitions for the basic skills benchmark.

Five levels testing progressively harder agent skills:

1. **mine_ores** -- Mine ore from three patches spread across the map.
2. **craft_all** -- Mine iron and copper, then craft all five recipe types.
3. **fuel_miner** -- Mine coal, deposit it into a pre-placed miner, let it
   produce ore.
4. **deploy_miners** -- Full deployment loop: mine resources, craft miners,
   place them on ore, fuel them.
5. **assembler_science** -- Feed a pre-placed assembler to produce science
   packs.

Each level defines a custom achievement function that returns a boolean
array of shape ``(MAX_ACHIEVEMENTS,)`` padded with zeros beyond the
level's own milestones. These drive both reward (via
``achievement_reward``) and scoring (fraction of achievements unlocked).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from factoriax.achievements import (
    count_total_items,
)
from factoriax.benchmarks.core import BenchmarkLevel
from factoriax.constants import (
    MAX_ACHIEVEMENTS,
    BlockType,
    Direction,
    ItemType,
    MachineType,
)
from factoriax.levels import LevelBuilder
from factoriax.state import EnvParams, EnvState

_NUM_PLAYERS: int = 1

# Number of custom achievements per level, used by scoring.
ACHIEVEMENT_COUNTS: dict[str, int] = {
    "mine_ores": 5,
    "craft_all": 10,
    "fuel_miner": 5,
    "deploy_miners": 7,
    "assembler_science": 7,
}

# Per-level reward weights for achievement_reward. Later (harder)
# milestones receive higher weight to encourage full completion.
ACHIEVEMENT_WEIGHTS: dict[str, jax.Array] = {}


def _params(width: int, height: int, max_timesteps: int) -> EnvParams:
    """Build EnvParams for a single-player level.

    Args:
        width: Map width in tiles.
        height: Map height in tiles.
        max_timesteps: Episode step budget.

    Returns:
        Configured ``EnvParams``.
    """
    return EnvParams(
        num_players=_NUM_PLAYERS,
        max_timesteps=max_timesteps,
        map_width=width,
        map_height=height,
    )


def _pad_conditions(conditions: jax.Array) -> jax.Array:
    """Pad a boolean condition array to ``MAX_ACHIEVEMENTS``.

    Args:
        conditions: Boolean array of shape ``(N,)`` where N <= MAX_ACHIEVEMENTS.

    Returns:
        Boolean array of shape ``(MAX_ACHIEVEMENTS,)``.
    """
    n = conditions.shape[0]
    return jnp.concatenate(
        [conditions, jnp.zeros(MAX_ACHIEVEMENTS - n, dtype=jnp.bool_)]
    )


def _pad_weights(weights: list[float]) -> jax.Array:
    """Build a padded weight vector from a list of per-achievement weights.

    Args:
        weights: Reward weight per achievement.

    Returns:
        Float32 array of shape ``(MAX_ACHIEVEMENTS,)``.
    """
    padded = jnp.zeros(MAX_ACHIEVEMENTS, dtype=jnp.float32)
    return padded.at[: len(weights)].set(jnp.array(weights, dtype=jnp.float32))


# ---------------------------------------------------------------------------
# Level 1 -- mine_ores
# ---------------------------------------------------------------------------
# 10x10 dirt map. Iron 3x3 at (2,1), copper 2x2 at (6,2), coal 2x2 at
# (7,7). Player spawns at (4,4). 100 resources per ore tile.
# Achievements: mined >= 1, 10, 25, 50, 100.


def _mine_ores_achievements(state: EnvState) -> jax.Array:
    """Achievement conditions for the mine_ores level.

    Args:
        state: Current environment state.

    Returns:
        Boolean array of shape ``(MAX_ACHIEVEMENTS,)``.
    """
    total_mined = (
        state.items_mined[ItemType.IRON_ORE]
        + state.items_mined[ItemType.COPPER_ORE]
        + state.items_mined[ItemType.COAL]
    )
    conditions = jnp.array(
        [
            total_mined >= 1,
            total_mined >= 5,
            total_mined >= 15,
            total_mined >= 30,
            total_mined >= 45,
        ],
        dtype=jnp.bool_,
    )
    return _pad_conditions(conditions)


_LEVEL_MINE = BenchmarkLevel(
    name="mine_ores",
    description=(
        "10x10 map with iron, copper, and coal patches. Mine as many "
        "resources as possible in 200 ticks. Tests movement and mining "
        "across multiple ore types."
    ),
    level=(
        LevelBuilder(10, 10)
        .fill_rect(2, 1, 3, 3, BlockType.IRON, resources=3)
        .fill_rect(6, 2, 2, 2, BlockType.COPPER, resources=3)
        .fill_rect(7, 7, 2, 2, BlockType.COAL, resources=3)
        .set_player_position(4, 4)
        .build("basic_mine_ores")
    ),
    env_params=_params(10, 10, 200),

)

ACHIEVEMENT_WEIGHTS["mine_ores"] = _pad_weights([1.0, 1.0, 2.0, 3.0, 5.0])


# ---------------------------------------------------------------------------
# Level 2 -- craft_all
# ---------------------------------------------------------------------------
# 12x12 dirt map. Iron 4x4 at (1,1) with 200 resources, copper 4x4 at
# (7,1) with 200 resources. Player at (6,5).
# Achievements: crafted 1 of each type (5), hold 5 of each type (5).


def _craft_all_achievements(state: EnvState) -> jax.Array:
    """Achievement conditions for the craft_all level.

    Args:
        state: Current environment state.

    Returns:
        Boolean array of shape ``(MAX_ACHIEVEMENTS,)``.
    """
    miner_count = count_total_items(state, ItemType.MINER)
    pallet_count = count_total_items(state, ItemType.PALLET)
    belt_count = count_total_items(state, ItemType.CONVEYOR_BELT)
    assembler_count = count_total_items(state, ItemType.ASSEMBLER)

    conditions = jnp.array(
        [
            # Crafted at least 1 of each
            miner_count >= 1,
            pallet_count >= 1,
            belt_count >= 1,
            assembler_count >= 1,
            # Hold 5 of each
            miner_count >= 5,
            pallet_count >= 5,
            belt_count >= 5,
            assembler_count >= 5,
        ],
        dtype=jnp.bool_,
    )
    return _pad_conditions(conditions)


_LEVEL_CRAFT = BenchmarkLevel(
    name="craft_all",
    description=(
        "12x12 map with large iron and copper patches. Craft all five "
        "recipe types: miner, pallet, belt, arm, assembler. Tests recipe "
        "selection, resource gathering, and inventory management across "
        "500 ticks."
    ),
    level=(
        LevelBuilder(12, 12)
        .fill_rect(1, 1, 4, 4, BlockType.IRON, resources=200)
        .fill_rect(7, 1, 4, 4, BlockType.COPPER, resources=200)
        .set_player_position(6, 5)
        .build("basic_craft_all")
    ),
    env_params=_params(12, 12, 500),

)

ACHIEVEMENT_WEIGHTS["craft_all"] = _pad_weights(
    [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0]
)


# ---------------------------------------------------------------------------
# Level 3 -- fuel_miner
# ---------------------------------------------------------------------------
# 8x8 dirt map. Iron 3x3 at (2,2) with 100 resources, coal 3x2 at (1,6)
# with 50 resources. Pre-placed MINER at (3,3) on iron, facing DOWN.
# Player at (5,7).
# Achievements: mined coal, coal in miner fuel, miner produced ore,
#               miner output >= 10, miner output >= 30.


def _fuel_miner_achievements(state: EnvState) -> jax.Array:
    """Achievement conditions for the fuel_miner level.

    Args:
        state: Current environment state.

    Returns:
        Boolean array of shape ``(MAX_ACHIEVEMENTS,)``.
    """
    mined_coal = state.items_mined[ItemType.COAL] >= 1

    # Any miner has coal in its pouch (fuel).
    is_miner = state.machine_types == MachineType.MINER
    coal_counts = state.machine_inventory[..., ItemType.COAL]
    coal_in_miner = jnp.any(is_miner & (coal_counts > 0))

    # Miner output: sum of all non-coal items in the pouch.
    # Coal is fuel; everything else is mined ore.
    all_items = jnp.sum(state.machine_inventory, axis=-1)
    output_counts = all_items - coal_counts
    has_output = output_counts > 0
    miner_produced = jnp.any(is_miner & has_output)

    # Total miner output across all miners (non-coal items).
    total_output = jnp.sum(jnp.where(is_miner, output_counts, 0))

    conditions = jnp.array(
        [
            mined_coal,
            coal_in_miner,
            miner_produced,
            total_output >= 10,
            total_output >= 30,
        ],
        dtype=jnp.bool_,
    )
    return _pad_conditions(conditions)


_LEVEL_FUEL = BenchmarkLevel(
    name="fuel_miner",
    description=(
        "8x8 map with a pre-placed miner on iron and a nearby coal "
        "patch. Mine coal, deposit it into the miner's fuel slot, and "
        "let the miner produce ore. Tests fuel mechanics and machine "
        "interaction."
    ),
    level=(
        LevelBuilder(8, 8)
        .fill_rect(2, 2, 3, 3, BlockType.IRON, resources=100)
        .fill_rect(1, 6, 3, 2, BlockType.COAL, resources=50)
        .place_machine(3, 3, MachineType.MINER, Direction.DOWN)
        .set_player_position(5, 7)
        .build("basic_fuel_miner")
    ),
    env_params=_params(8, 8, 200),

)

ACHIEVEMENT_WEIGHTS["fuel_miner"] = _pad_weights([1.0, 2.0, 3.0, 4.0, 5.0])


# ---------------------------------------------------------------------------
# Level 4 -- deploy_miners
# ---------------------------------------------------------------------------
# 12x12 dirt map. Iron 3x4 at (1,1), copper 3x4 at (8,1), coal 4x3 at
# (3,6). All 100 resources per tile. Player at (5,5).
# Achievements: crafted miner, placed miner on ore, fueled a miner,
#               miner produced ore, 2 miners on map, both miners fueled,
#               total automated ore >= 20.


def _deploy_miners_achievements(state: EnvState) -> jax.Array:
    """Achievement conditions for the deploy_miners level.

    Args:
        state: Current environment state.

    Returns:
        Boolean array of shape ``(MAX_ACHIEVEMENTS,)``.
    """
    # Count miners held in inventory (evidence of crafting).
    miners_held = count_total_items(state, ItemType.MINER)

    # Count placed miners and their properties.
    is_miner = state.machine_types == MachineType.MINER
    num_miners = jnp.sum(is_miner)

    # A miner is "on ore" if the block beneath it is a mineable type.
    is_ore = (
        (state.map == BlockType.IRON)
        | (state.map == BlockType.COPPER)
        | (state.map == BlockType.COAL)
    )
    miner_on_ore = jnp.any(is_miner & is_ore)

    # Any miner has coal (fuel) in its pouch.
    coal_counts = state.machine_inventory[..., ItemType.COAL]
    has_fuel = coal_counts > 0
    any_fueled = jnp.any(is_miner & has_fuel)

    # Any miner produced ore (non-coal items in pouch).
    all_items = jnp.sum(state.machine_inventory, axis=-1)
    output_counts = all_items - coal_counts
    any_output = jnp.any(is_miner & (output_counts > 0))

    # Count fueled miners.
    num_fueled = jnp.sum(is_miner & has_fuel)

    # Total automated ore output.
    total_output = jnp.sum(jnp.where(is_miner, output_counts, 0))

    # "Crafted a miner" detected by having held or placed one.
    crafted_miner = (miners_held >= 1) | (num_miners >= 1)

    conditions = jnp.array(
        [
            crafted_miner,
            miner_on_ore,
            any_fueled,
            any_output,
            num_miners >= 2,
            num_fueled >= 2,
            total_output >= 20,
        ],
        dtype=jnp.bool_,
    )
    return _pad_conditions(conditions)


_LEVEL_DEPLOY = BenchmarkLevel(
    name="deploy_miners",
    description=(
        "12x12 map with iron, copper, and coal patches. Start from "
        "nothing: mine resources, craft miners, place them on ore, "
        "fuel them with coal. Tests the full deployment loop over "
        "500 ticks."
    ),
    level=(
        LevelBuilder(12, 12)
        .fill_rect(1, 1, 3, 4, BlockType.IRON, resources=100)
        .fill_rect(8, 1, 3, 4, BlockType.COPPER, resources=100)
        .fill_rect(3, 6, 4, 3, BlockType.COAL, resources=100)
        .set_player_position(5, 5)
        .build("basic_deploy_miners")
    ),
    env_params=_params(12, 12, 500),

)

ACHIEVEMENT_WEIGHTS["deploy_miners"] = _pad_weights([1.0, 2.0, 3.0, 4.0, 3.0, 5.0, 6.0])


# ---------------------------------------------------------------------------
# Level 5 -- assembler_science
# ---------------------------------------------------------------------------
# 10x10 dirt map. Iron 3x3 at (1,1), copper 3x3 at (6,2). Pre-placed
# ASSEMBLER at (5,4), recipe set to Basic Science Pack (index 3).
# Player at (4,5).
# Achievements: mined iron, mined copper, deposited into assembler,
#               assembler produced, hold 1 science pack, hold 5, hold 10.


def _assembler_science_achievements(state: EnvState) -> jax.Array:
    """Achievement conditions for the assembler_science level.

    Args:
        state: Current environment state.

    Returns:
        Boolean array of shape ``(MAX_ACHIEVEMENTS,)``.
    """
    mined_iron = state.items_mined[ItemType.IRON_ORE] >= 1
    mined_copper = state.items_mined[ItemType.COPPER_ORE] >= 1

    # Any assembler has items in its pouch (evidence of deposit).
    is_asm = state.machine_types == MachineType.ASSEMBLER
    inv_total = jnp.sum(state.machine_inventory, axis=-1)
    deposited = jnp.any(is_asm & (inv_total > 0))

    # Assembler produced output (science packs in pouch).
    has_output = (
        state.machine_inventory[..., ItemType.BASIC_SCIENCE_PACK] > 0
    )
    produced = jnp.any(is_asm & has_output)

    # Science packs held by player.
    packs_held = count_total_items(state, ItemType.BASIC_SCIENCE_PACK)

    conditions = jnp.array(
        [
            mined_iron,
            mined_copper,
            deposited,
            produced,
            packs_held >= 1,
            packs_held >= 5,
            packs_held >= 10,
        ],
        dtype=jnp.bool_,
    )
    return _pad_conditions(conditions)


_LEVEL_ASSEMBLER = BenchmarkLevel(
    name="assembler_science",
    description=(
        "10x10 map with iron and copper patches and a pre-placed "
        "assembler set to craft Basic Science Packs. Mine resources, "
        "deposit into the assembler, and collect science packs. Tests "
        "multi-slot machine interaction over 300 ticks."
    ),
    level=(
        LevelBuilder(10, 10)
        .fill_rect(1, 1, 3, 3, BlockType.IRON, resources=100)
        .fill_rect(6, 2, 3, 3, BlockType.COPPER, resources=100)
        .place_machine(5, 4, MachineType.ASSEMBLER)
        .set_machine_recipe(5, 4, 3)
        .set_player_position(4, 5)
        .build("basic_assembler_science")
    ),
    env_params=_params(10, 10, 300),

)

ACHIEVEMENT_WEIGHTS["assembler_science"] = _pad_weights(
    [1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
)


# ---------------------------------------------------------------------------
# Exported level list
# ---------------------------------------------------------------------------

BASIC_SKILLS_LEVELS: list[BenchmarkLevel] = [
    _LEVEL_MINE,
    _LEVEL_CRAFT,
    _LEVEL_FUEL,
    _LEVEL_DEPLOY,
    _LEVEL_ASSEMBLER,
]
