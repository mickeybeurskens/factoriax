"""Level definitions for the single-agent mining benchmark.

Five levels of increasing difficulty, each a 200-tick mining challenge.
Difficulty scales along three axes: map size (exploration cost), resource
density, and the spatial relationship between spawn and high-value deposits.

All levels use one player and cap at 200 timesteps. Resources default to
``BLOCK_MAX_RESOURCES`` (100) per ore tile unless stated otherwise.

Spawn placement is handled by ``build_state`` — players always appear at
the horizontal centre of the map, vertically centred. The exact spawn tile
is forced to dirt even if a resource patch overlaps it.

Level overview:
    1. adjacent_coal     — 10x10, dense coal patch adjacent to spawn.
    2. coal_and_iron     — 12x12, coal and iron equidistant from spawn.
    3. three_patches     — 15x15, three ore types; copper is closest.
    4. iron_near_copper_far — 18x18, cheap ore nearby, valuable ore far.
    5. two_corners       — 22x22, iron and copper equidistant in opposite
                           corners; agent must commit to one.
"""

from __future__ import annotations

from factoriax.benchmarks.core import BenchmarkLevel
from factoriax.constants import BlockType
from factoriax.levels import LevelBuilder
from factoriax.state import EnvParams

# All benchmark levels use these constraints.
_MAX_TIMESTEPS: int = 200
_NUM_PLAYERS: int = 1


def _params(width: int, height: int) -> EnvParams:
    """Build EnvParams matching the given map dimensions.

    Args:
        width: Map width in tiles.
        height: Map height in tiles.

    Returns:
        ``EnvParams`` with the correct dimensions, 1 player, and a 200-step
        episode budget.
    """
    return EnvParams(
        num_players=_NUM_PLAYERS,
        max_timesteps=_MAX_TIMESTEPS,
        map_width=width,
        map_height=height,
    )


# ---------------------------------------------------------------------------
# Level 1 — adjacent_coal
# ---------------------------------------------------------------------------
# 10x10 map. Spawn at (5, 5). A 4x4 coal patch at (2, 2) means the nearest
# ore is 1 tile from spawn. No navigation required — just mine. Sets a
# baseline for what a good miner can do with zero travel overhead.

_LEVEL_1 = BenchmarkLevel(
    name="adjacent_coal",
    description=(
        "10x10 map with a 4x4 coal patch immediately adjacent to spawn. "
        "No navigation required. Establishes a mining-speed baseline."
    ),
    level=(
        LevelBuilder(10, 10)
        .fill_rect(2, 2, 4, 4, BlockType.COAL)
        .build("mining_l1_adjacent_coal")
    ),
    env_params=_params(10, 10),
)

# ---------------------------------------------------------------------------
# Level 2 — coal_and_iron
# ---------------------------------------------------------------------------
# 12x12 map. Spawn at (6, 6). Coal 3x3 at (3, 3) and iron 3x3 at (7, 3)
# are roughly equidistant from spawn (~3 and ~4 tiles respectively). The
# agent must decide between coal (1 pt, slightly closer) and iron (2 pts).
# A greedy by-value agent should prefer iron; a naive agent may mine coal.

_LEVEL_2 = BenchmarkLevel(
    name="coal_and_iron",
    description=(
        "12x12 map with coal and iron patches equidistant from spawn. "
        "Tests whether the agent prefers higher-value resources."
    ),
    level=(
        LevelBuilder(12, 12)
        .fill_rect(3, 3, 3, 3, BlockType.COAL)
        .fill_rect(7, 3, 3, 3, BlockType.IRON)
        .build("mining_l2_coal_and_iron")
    ),
    env_params=_params(12, 12),
)

# ---------------------------------------------------------------------------
# Level 3 — three_patches
# ---------------------------------------------------------------------------
# 15x15 map. Spawn at (7, 7). Three ore patches in an asymmetric triangle:
#   Coal    4x4 at (0, 0)  — top-left,    ~14 tiles from spawn.
#   Iron    4x4 at (11, 0) — top-right,   ~11 tiles from spawn.
#   Copper  4x4 at (5, 11) — bottom-mid,   ~6 tiles from spawn.
# Copper is both closest and most valuable — good agents go there first.
# The question is whether they mine enough copper before branching or not.

_LEVEL_3 = BenchmarkLevel(
    name="three_patches",
    description=(
        "15x15 map with coal, iron, and copper in an asymmetric triangle. "
        "Copper is closest and most valuable; tests multi-resource prioritisation."
    ),
    level=(
        LevelBuilder(15, 15)
        .fill_rect(0, 0, 4, 4, BlockType.COAL)
        .fill_rect(11, 0, 4, 4, BlockType.IRON)
        .fill_rect(5, 11, 4, 4, BlockType.COPPER)
        .build("mining_l3_three_patches")
    ),
    env_params=_params(15, 15),
)

# ---------------------------------------------------------------------------
# Level 4 — iron_near_copper_far
# ---------------------------------------------------------------------------
# 18x18 map. Spawn at (9, 9). Iron 4x4 at (6, 6) is ~3 tiles away.
# Copper 3x3 at (14, 1) is ~13 tiles away (top-right corner).
# The tension: iron mines quickly with no travel cost; copper is worth
# 1.5× more per item but takes ~13 steps just to reach. A competent agent
# should calculate that travelling to copper and mining there is worth it
# given the 200-tick budget.

_LEVEL_4 = BenchmarkLevel(
    name="iron_near_copper_far",
    description=(
        "18x18 map with iron adjacent to spawn and copper in the far corner. "
        "Tests whether the agent commits to the higher-value distant resource."
    ),
    level=(
        LevelBuilder(18, 18)
        .fill_rect(6, 6, 4, 4, BlockType.IRON)
        .fill_rect(14, 1, 3, 3, BlockType.COPPER)
        .build("mining_l4_iron_near_copper_far")
    ),
    env_params=_params(18, 18),
)

# ---------------------------------------------------------------------------
# Level 5 — two_corners
# ---------------------------------------------------------------------------
# 22x22 map. Spawn at (11, 11). Iron 4x4 at (1, 17) and copper 4x4 at
# (17, 1) are both ~16 Manhattan tiles from spawn and in opposite corners.
# There is not enough time to mine both meaningfully. The right answer is
# to commit to copper (3 pts) over iron (2 pts) and stay there.
# An agent that oscillates or chooses iron will score significantly lower.

_LEVEL_5 = BenchmarkLevel(
    name="two_corners",
    description=(
        "22x22 map with iron and copper in opposite corners, equidistant from "
        "spawn. Tests commitment to the higher-value resource under time pressure."
    ),
    level=(
        LevelBuilder(22, 22)
        .fill_rect(1, 17, 4, 4, BlockType.IRON)
        .fill_rect(17, 1, 4, 4, BlockType.COPPER)
        .build("mining_l5_two_corners")
    ),
    env_params=_params(22, 22),
)

# Ordered from easiest to hardest.
MINING_LEVELS: list[BenchmarkLevel] = [
    _LEVEL_1,
    _LEVEL_2,
    _LEVEL_3,
    _LEVEL_4,
    _LEVEL_5,
]
