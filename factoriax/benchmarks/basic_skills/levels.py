"""Level definitions for the basic skills benchmark.

Three levels testing isolated agent skills:

1. **mine_resources** — Mine as much ore as possible from two patches.
2. **craft_chests** — Craft chests from iron (pre-stocked + mineable).
3. **fill_chest** — Mine ore, navigate to a pre-placed chest, and deposit
   full stacks into it.

All levels use one player and are sized to be completable within their
timestep budget by a competent agent.
"""

from __future__ import annotations

from factoriax.benchmarks.core import BenchmarkLevel
from factoriax.constants import BlockType, ItemType, MachineType
from factoriax.levels import LevelBuilder
from factoriax.state import EnvParams

_NUM_PLAYERS: int = 1


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


# ---------------------------------------------------------------------------
# Level 1 — mine_resources
# ---------------------------------------------------------------------------
# 10x10 dirt map. A 3x3 iron patch at (2, 2) and a 3x3 coal patch at (6, 2).
# Player spawns at centre (5, 5). 200 ticks to mine as much as possible.
# Each ore tile has 1 resource so the agent must navigate between tiles.
# Tests: movement + mine action.

_LEVEL_MINE = BenchmarkLevel(
    name="mine_resources",
    description=(
        "10x10 map with iron and coal patches. Mine as many resources "
        "as possible in 200 ticks. Tests basic movement and mining."
    ),
    level=(
        LevelBuilder(10, 10)
        .fill_rect(2, 2, 3, 3, BlockType.IRON, resources=1)
        .fill_rect(6, 2, 3, 3, BlockType.COAL, resources=1)
        .build("basic_mine_resources")
    ),
    env_params=_params(10, 10, 200),
)


# ---------------------------------------------------------------------------
# Level 2 — craft_chests
# ---------------------------------------------------------------------------
# 7x7 dirt map. A 3x3 iron patch at (0, 0) with 5 ore per tile (45 total).
# Player starts with 30 iron in inventory, enough for 6 chests outright.
# Each chest costs 5 iron and takes 2 ticks to craft (recipe index 1).
# Mining more iron from the map allows additional chests. 200 ticks gives
# plenty of time to craft all possible chests.
# Tests: recipe selection + craft action, optional mining for more materials.

_LEVEL_CRAFT = BenchmarkLevel(
    name="craft_chests",
    description=(
        "7x7 map with starting iron inventory and an iron patch. "
        "Craft as many chests as possible. Tests recipe selection "
        "and crafting, with optional mining for extra materials."
    ),
    level=(
        LevelBuilder(7, 7)
        .fill_rect(0, 0, 3, 3, BlockType.IRON, resources=5)
        .build("basic_craft_chests")
    ),
    env_params=_params(7, 7, 200),
)
_LEVEL_CRAFT.level.player_inventory = [(int(ItemType.IRON), 30)]


# ---------------------------------------------------------------------------
# Level 3 — fill_chest
# ---------------------------------------------------------------------------
# 7x7 dirt map. A 3x3 iron patch at (1, 1) with 100 ore per tile (900 total).
# One chest pre-placed at (5, 3). Player must mine iron, walk to the chest,
# and deposit. Scoring rewards full stacks (64 items per slot, 8 slots).
# 300 ticks is tight but allows filling several slots if the agent is
# efficient about its mine-walk-deposit loop.
# Tests: mining + navigation + deposit action.

_LEVEL_FILL = BenchmarkLevel(
    name="fill_chest",
    description=(
        "7x7 map with an iron patch and a pre-placed chest. "
        "Mine iron, walk to the chest, and deposit full stacks. "
        "Tests mining, navigation, and the deposit action."
    ),
    level=(
        LevelBuilder(7, 7)
        .fill_rect(1, 1, 3, 3, BlockType.IRON, resources=100)
        .place_machine(5, 3, MachineType.CHEST)
        .build("basic_fill_chest")
    ),
    env_params=_params(7, 7, 300),
)


BASIC_SKILLS_LEVELS: list[BenchmarkLevel] = [
    _LEVEL_MINE,
    _LEVEL_CRAFT,
    _LEVEL_FILL,
]
