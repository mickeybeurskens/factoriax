"""Level definitions for the basic skills benchmark.

Six levels testing progressively harder agent skills:

1. **mine_resources** — Mine as much ore as possible from two patches.
2. **craft_chests** — Craft chests from iron (pre-stocked + mineable).
3. **fill_chest** — Mine ore, navigate to a pre-placed chest, and deposit
   full stacks into it.
4. **craft_miners** — Mine iron and copper, then craft miner machines.
5. **deploy_miner** — Place a miner on ore and fuel it with coal.
6. **mining_factory** — Full loop: mine, craft a miner, place, fuel, and
   maximise total ore output.

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
# 7x7 dirt map. A 3x3 iron patch at (0, 0) with 20 ore per tile (180 total).
# Player starts with 30 iron in inventory, enough for 6 chests outright.
# Each chest costs 5 iron (instant craft). Mining more iron from the map
# allows additional chests. 200 ticks gives plenty of time.
# Tests: crafting action, optional mining for more materials.

_LEVEL_CRAFT = BenchmarkLevel(
    name="craft_chests",
    description=(
        "7x7 map with starting iron inventory and an iron patch. "
        "Craft as many chests as possible. Tests recipe selection "
        "and crafting, with optional mining for extra materials."
    ),
    level=(
        LevelBuilder(7, 7)
        .fill_rect(0, 0, 3, 3, BlockType.IRON, resources=20)
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


# ---------------------------------------------------------------------------
# Level 4 — craft_miners
# ---------------------------------------------------------------------------
# 9x9 dirt map. A 3x3 iron patch at (1, 1) and a 3x3 copper patch at (5, 1),
# each with 20 ore per tile. Player spawns at (4, 6). Miner recipe requires
# 5 iron + 5 copper, so the agent must navigate between two ore types and
# select the correct recipe (index 0). 300 ticks.
# Tests: dual-resource mining + recipe selection.

_LEVEL_CRAFT_MINERS = BenchmarkLevel(
    name="craft_miners",
    description=(
        "9x9 map with iron and copper patches. Start with enough "
        "materials for two miners. Craft as many miners as possible. "
        "Tests mining two resource types and recipe selection for a "
        "multi-input recipe."
    ),
    level=(
        LevelBuilder(9, 9)
        .fill_rect(1, 1, 3, 3, BlockType.IRON, resources=20)
        .fill_rect(5, 1, 3, 3, BlockType.COPPER, resources=20)
        .build("basic_craft_miners")
    ),
    env_params=_params(9, 9, 300),
)
_LEVEL_CRAFT_MINERS.level.player_inventory = [
    (int(ItemType.IRON), 10),
    (int(ItemType.COPPER), 10),
]


# ---------------------------------------------------------------------------
# Level 5 — deploy_miner
# ---------------------------------------------------------------------------
# 7x7 dirt map. A 3x3 coal patch at (1, 1) with 50 ore per tile. Player
# starts with 1 miner and 10 coal in inventory. Place the miner on a coal
# tile, deposit coal into its fuel slot (slot 0), and let it run. A miner
# consumes 1 power/tick and mines 3 ore/tick. 10 coal = 100 power = 100
# ticks of mining, but the output slot caps at 64 items. 200 ticks.
# Tests: PLACE action + DEPOSIT into machine fuel slot.

_LEVEL_DEPLOY = BenchmarkLevel(
    name="deploy_miner",
    description=(
        "7x7 map with a coal patch. Start with a miner and coal. "
        "Place the miner on ore, fuel it, and let it produce. "
        "Tests machine placement and fuel deposit."
    ),
    level=(
        LevelBuilder(7, 7)
        .fill_rect(1, 1, 3, 3, BlockType.COAL, resources=50)
        .build("basic_deploy_miner")
    ),
    env_params=_params(7, 7, 200),
)
_LEVEL_DEPLOY.level.player_inventory = [
    (int(ItemType.MINER), 5),
    (int(ItemType.COAL), 10),
]


# ---------------------------------------------------------------------------
# Level 6 — mining_factory
# ---------------------------------------------------------------------------
# 11x11 dirt map. Iron 3x3 at (1, 1), copper 3x3 at (7, 1), coal 3x3 at
# (4, 5), all with 100 ore per tile. Player spawns at (5, 9) with nothing.
# The agent must mine iron + copper to craft a miner, place it on ore, mine
# coal to fuel it, and maximise total ore output. 500 ticks.
# Tests: full automation loop (the capstone skill).

_LEVEL_FACTORY = BenchmarkLevel(
    name="mining_factory",
    description=(
        "11x11 map with iron, copper, and coal patches. Start from "
        "nothing: mine, craft a miner, place it, fuel it, and "
        "maximise total ore output. Capstone level testing the "
        "full factory-building loop."
    ),
    level=(
        LevelBuilder(11, 11)
        .fill_rect(1, 1, 3, 3, BlockType.IRON, resources=100)
        .fill_rect(7, 1, 3, 3, BlockType.COPPER, resources=100)
        .fill_rect(4, 5, 3, 3, BlockType.COAL, resources=100)
        .build("basic_mining_factory")
    ),
    env_params=_params(11, 11, 500),
)


# ---------------------------------------------------------------------------
# Level 7 — place_and_fuel
# ---------------------------------------------------------------------------
# 7x7 dirt map with a 3x3 coal patch at (2, 2). Player spawns at centre
# (3, 3) — right on the coal. Start with 3 miners and 20 coal. Walk to
# the edge of the patch, place miners on adjacent coal tiles, fuel each.
# Each miner needs only 3 coal to fill its 64-item output slot (3 coal =
# 30 power, 30 * 3 ore/tick = 90, capped at 64). The dirt border gives
# room to navigate around placed machines. Hand-mine extra coal for the
# second and third miners.
# Tests: PLACE action + DEPOSIT into fuel slot + navigation around machines.

_LEVEL_PLACE_AND_FUEL = BenchmarkLevel(
    name="place_and_fuel",
    description=(
        "7x7 map with a coal patch. Start with 3 miners and 20 coal. "
        "Place miners on coal tiles and fuel them. Score measures "
        "total ore produced in miner output slots."
    ),
    level=(
        LevelBuilder(7, 7)
        .fill_rect(2, 2, 3, 3, BlockType.COAL, resources=50)
        .set_player_position(3, 1)
        .build("basic_place_and_fuel")
    ),
    env_params=_params(7, 7, 200),
)
_LEVEL_PLACE_AND_FUEL.level.player_inventory = [
    (int(ItemType.MINER), 3),
    (int(ItemType.COAL), 20),
]


# ---------------------------------------------------------------------------
# Level 8 — withdraw_ore
# ---------------------------------------------------------------------------
# 7x3 map. Three pre-placed miners at (1, 0), (3, 0), (5, 0) on coal
# tiles, each with pre-loaded output slots (10, 20, 30 ore). No fuel,
# not producing — just static storage. Player at (0, 1) walks along the
# dirt corridor in row 1, faces UP toward each miner, and withdraws.
# Miners block movement, so the agent cannot walk into row 0.
# Tests: WITHDRAW action + navigation to multiple machines.

_LEVEL_WITHDRAW = BenchmarkLevel(
    name="withdraw_ore",
    description=(
        "7x3 corridor with 3 pre-loaded miners. Walk to each miner "
        "and withdraw ore from its output slot. Score measures total "
        "ore collected in player inventory."
    ),
    level=(
        LevelBuilder(7, 3)
        .fill_rect(1, 0, 1, 1, BlockType.COAL, resources=50)
        .fill_rect(3, 0, 1, 1, BlockType.COAL, resources=50)
        .fill_rect(5, 0, 1, 1, BlockType.COAL, resources=50)
        .place_machine(1, 0, MachineType.MINER)
        .place_machine(3, 0, MachineType.MINER)
        .place_machine(5, 0, MachineType.MINER)
        .set_machine_inventory(1, 0, 1, int(ItemType.COAL), 10)
        .set_machine_inventory(3, 0, 1, int(ItemType.COAL), 20)
        .set_machine_inventory(5, 0, 1, int(ItemType.COAL), 30)
        .set_player_position(0, 1)
        .build("basic_withdraw_ore")
    ),
    env_params=_params(7, 3, 100),
)


BASIC_SKILLS_LEVELS: list[BenchmarkLevel] = [
    _LEVEL_MINE,
    _LEVEL_CRAFT,
    _LEVEL_FILL,
    _LEVEL_CRAFT_MINERS,
    _LEVEL_DEPLOY,
    _LEVEL_FACTORY,
    _LEVEL_PLACE_AND_FUEL,
    _LEVEL_WITHDRAW,
]
