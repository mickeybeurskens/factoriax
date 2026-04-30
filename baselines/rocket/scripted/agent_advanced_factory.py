"""Advanced-factory rocket agent — Phase A + Phase B (4 smelter cells).

Two high-level phases:

- **Phase A** drops a miner + pallet on each non-coal patch (iron,
  copper, tin, silicon) and a miner-only on the coal patch. Ore flows
  passively into the four patch pallets; the coal miner stalls because
  no buffer exists east of it yet — Phase B closes that gap by placing
  a belt at (10, 24) that the miner pushes onto.

- **Phase B** builds a smelter cell on each of the four ore patches,
  one at a time (iron → copper → tin → silicon). Each cell is a
  furnace + plate-extract arm + plate output pallet, fed by a coal
  trunk from the coal patch (one dedicated coal miner per cell).

The four output bus pallets at (10, 11), (25, 11), (25, 26), and
(17, 7) are left as bus tiles for future iterations to draw from
(rocket sub-assemblies, science packs, etc.). This iteration just
proves all four cells can be built and produce plates in parallel.

Pre-smelt strategy. ``ProduceInMachine`` picks the *nearest*
furnace/assembler each cycle. After the iron cell's furnace at (8, 11)
is placed, it is closer to the player's typical Phase B workspace
(near the bus pallets) than the pre-placed furnace at (15, 16); a
``ProduceInFurnace(COPPER_PLATE, ...)`` issued post-iron-cell would
deposit ore into the iron cell's furnace and corrupt iron production.
The fix: hand-mine and pre-smelt **every** plate the entire Phase B
chain needs in Phase A bootstrap, before any cell exists. Phase B
cells then only run ``ProduceInAssembler`` (assembler crafts target
the pre-placed assembler at (17, 16); there is no second assembler in
this iteration so no nearest-machine confusion). The user's directive
"collect from the factory instead of hand smelting once they are
available" is honored in spirit — the cells run continuously and
their plate-output bus pallets are left full for future phases.

Map layout::

                      silicon (14, 3-5)
                              |
                      [silicon trunk: col 7 north + row 8 east
                       across iron ore tiles]
                              |
       iron (7, 7-9)                copper (22-24, 7-9)
        |                           |
        [iron trunk col 10]         [copper trunk: col 11 + row 12]
        |                           |
                spawn (16, 16)
                              |
                              | [tin trunk row 26 east]
       coal (7-9, 22-24)            tin (22-24, 22-24)
        4 coal miners

Each cell follows the same shape: coal-buffer pallet south of the
furnace; plate-extract arm east of the furnace facing RIGHT; plate
output pallet east of the arm. Furnace faces RIGHT so its stand tile
(one west) is dirt rather than the ore-pallet that sits north.

Trunks are laid via :func:`place_belt_path`, which takes a list of
axis-aligned waypoints (start → corners → sink) and emits one
:class:`PlaceMachineAt` belt per intermediate tile, with the
direction inferred from the next tile in the path. The sink is the
coal-buffer pallet (placed separately), not a belt. Placement on ore
tiles is valid — the engine's
:func:`factoriax.placement.is_valid_placement_tile` only rejects
``WATER``/``OUT_OF_BOUNDS``/already-occupied tiles — so the silicon
trunk crosses iron ore tiles to take a more direct route.

Bootstrap budgets (Phase A pre-smelts everything):

================  =====  =====  ===  ===
Resource          mine   smelt  ...
================  =====  =====  ===  ===
IRON_ORE          20
COPPER_ORE        62
TIN_ORE           32
COAL              118    (1 per smelt)
IRON_PLATE        ...    20
COPPER_PLATE      ...    62
TIN_PLATE         ...    32
REFRACTORY        ...    4    (1 per cell furnace)
================  =====  =====  ===  ===

All assembler crafts (WIRE, MINER, PALLET, ARM, FURNACE, BELT) happen
across Phase A and the four Phase B sub-phases as each cell needs them.
"""

from __future__ import annotations

from factoriax.constants import Direction, ItemType, MachineType
from factoriax.state import EnvParams

from .agent import ScriptedAgent, _miner_has_output_predicate
from .goals import (
    Goal,
    MineOre,
    PlaceMachineAt,
    ProduceInAssembler,
    ProduceInFurnace,
    Wait,
    WaitUntil,
    WithdrawFromBusAt,
    build_smelter_cell_at,
    place_belt_path,
)
from .planner import Planner

# ---------------------------------------------------------------------------
# Plate-pallet bus tiles (output pallets of the four smelter cells)
# ---------------------------------------------------------------------------

_IRON_PLATE_BUS = (10, 11)
_COPPER_PLATE_BUS = (25, 11)
_TIN_PLATE_BUS = (25, 26)
_SILICON_PLATE_BUS = (17, 7)


# ---------------------------------------------------------------------------
# Phase A — bootstrap everything, then drop miners + pallets
# ---------------------------------------------------------------------------


def _phase_a_bootstrap_mine_and_smelt() -> list[Goal]:
    """Hand-mine + smelt every plate the entire run will need.

    Crafted up front so Phase B sub-phases never have to issue a
    FURNACE recipe — any post-iron-cell ``ProduceInFurnace`` would
    target the iron cell's furnace instead of the pre-placed one.
    """
    return [
        # Mining (in patch order — iron, copper, tin, coal — to keep
        # travel short).
        MineOre(ItemType.IRON_ORE, 20),
        MineOre(ItemType.COPPER_ORE, 62),
        MineOre(ItemType.TIN_ORE, 32),
        MineOre(ItemType.COAL, 118),
        # Smelt all the plates the entire run consumes. Only 1 furnace
        # exists at this point (the pre-placed one at (15, 16)) so
        # there's no nearest-machine confusion.
        ProduceInFurnace(ItemType.IRON_PLATE, 20),
        ProduceInFurnace(ItemType.COPPER_PLATE, 62),
        ProduceInFurnace(ItemType.TIN_PLATE, 32),
        ProduceInFurnace(ItemType.REFRACTORY, 4),
    ]


def _phase_a_craft_and_place() -> list[Goal]:
    """Craft Phase A's 10 WIRE + 5 MINER + 5 PALLET, then place them."""
    return [
        ProduceInAssembler(ItemType.WIRE, 10),
        ProduceInAssembler(ItemType.MINER, 5),
        ProduceInAssembler(ItemType.PALLET, 5),
        # Iron node
        PlaceMachineAt(MachineType.PALLET, (8, 10), int(Direction.DOWN)),
        PlaceMachineAt(MachineType.MINER, (8, 9), int(Direction.DOWN)),
        # Copper node
        PlaceMachineAt(MachineType.PALLET, (23, 10), int(Direction.DOWN)),
        PlaceMachineAt(MachineType.MINER, (23, 9), int(Direction.DOWN)),
        # Tin node
        PlaceMachineAt(MachineType.PALLET, (23, 25), int(Direction.DOWN)),
        PlaceMachineAt(MachineType.MINER, (23, 24), int(Direction.DOWN)),
        # Silicon node
        PlaceMachineAt(MachineType.PALLET, (15, 6), int(Direction.DOWN)),
        PlaceMachineAt(MachineType.MINER, (15, 5), int(Direction.DOWN)),
        # Coal — east-edge miner facing RIGHT, no pallet (Phase B
        # places a belt at (10, 24) which the miner pushes onto).
        PlaceMachineAt(MachineType.MINER, (9, 24), int(Direction.RIGHT)),
        WaitUntil(_miner_has_output_predicate(), max_ticks=30),
    ]


# ---------------------------------------------------------------------------
# Phase B helpers — one cell at a time. Each cell phase only crafts
# (assembler) and places; no smelting (would target the iron cell's
# furnace post-Phase-B-iron).
# ---------------------------------------------------------------------------


_MAP_SIZE: tuple[int, int] = (32, 32)


def _phase_b_iron() -> list[Goal]:
    """Iron smelter cell + 14-belt coal trunk (col 10 north, bend west).

    Cell anchored at furnace tile (8, 11): coal_buffer (8, 12) UP,
    plate-bus (10, 11) DOWN, arm (9, 11) RIGHT, furnace (8, 11) RIGHT.
    Trunk waypoints: (10, 24) coal-miner output → (10, 12) corner →
    (8, 12) sink (the cell's coal-buffer pallet).
    """
    return [
        ProduceInAssembler(ItemType.WIRE, 3),
        ProduceInAssembler(ItemType.PALLET, 2),
        ProduceInAssembler(ItemType.ARM, 1),
        ProduceInAssembler(ItemType.FURNACE, 1),
        ProduceInAssembler(ItemType.CONVEYOR_BELT, 14),
        *build_smelter_cell_at((8, 11), map_size=_MAP_SIZE),
        *place_belt_path([(10, 24), (10, 12), (8, 12)]),
    ]


def _phase_b_copper() -> list[Goal]:
    """Copper smelter cell + 28-belt trunk via row 25 / col 11 / row 12.

    Cell anchored at (23, 11): coal_buffer (23, 12), plate-bus (25, 11),
    arm (24, 11), furnace (23, 11). New coal miner (8, 24) DOWN pushes
    onto (8, 25). Trunk waypoints: (8, 25) → (11, 25) → (11, 12) →
    (23, 12) sink.

    Iron plates come from the iron cell's bus pallet (10, 11); copper
    + tin + refractory were pre-smelted in Phase A bootstrap.
    """
    return [
        WithdrawFromBusAt(_IRON_PLATE_BUS, ItemType.IRON_PLATE, 30),
        ProduceInAssembler(ItemType.WIRE, 4),
        ProduceInAssembler(ItemType.PALLET, 2),
        ProduceInAssembler(ItemType.ARM, 1),
        ProduceInAssembler(ItemType.FURNACE, 1),
        ProduceInAssembler(ItemType.MINER, 1),
        ProduceInAssembler(ItemType.CONVEYOR_BELT, 28),
        # New coal miner — stand (8, 23) is coal ore (walkable).
        PlaceMachineAt(MachineType.MINER, (8, 24), int(Direction.DOWN)),
        *build_smelter_cell_at((23, 11), map_size=_MAP_SIZE),
        *place_belt_path([(8, 25), (11, 25), (11, 12), (23, 12)]),
    ]


def _phase_b_tin() -> list[Goal]:
    """Tin smelter cell + 18-belt trunk via row 26 east + (22, 27) east.

    Cell anchored at (23, 26): coal_buffer (23, 27), plate-bus (25, 26),
    arm (24, 26), furnace (23, 26). Coal miner (7, 24) DOWN pushes
    south to (7, 25). Trunk waypoints: (7, 25) → (7, 26) → (22, 26) →
    (22, 27) → (23, 27) sink. The trunk detours south at col 22
    because (23, 26) is the tin furnace.

    Iron from iron cell, copper from copper cell, tin pre-smelted.
    """
    return [
        WithdrawFromBusAt(_IRON_PLATE_BUS, ItemType.IRON_PLATE, 20),
        WithdrawFromBusAt(_COPPER_PLATE_BUS, ItemType.COPPER_PLATE, 23),
        ProduceInAssembler(ItemType.WIRE, 4),
        ProduceInAssembler(ItemType.PALLET, 2),
        ProduceInAssembler(ItemType.ARM, 1),
        ProduceInAssembler(ItemType.FURNACE, 1),
        ProduceInAssembler(ItemType.MINER, 1),
        ProduceInAssembler(ItemType.CONVEYOR_BELT, 18),
        *build_smelter_cell_at((23, 26), map_size=_MAP_SIZE),
        # Belts placed BEFORE coal miner — belt (7, 25) DOWN's stand
        # tile (7, 24) is the miner's eventual position. Place belts
        # while the miner tile is still walkable coal ore.
        *place_belt_path([(7, 25), (7, 26), (22, 26), (22, 27), (23, 27)]),
        PlaceMachineAt(MachineType.MINER, (7, 24), int(Direction.DOWN)),
    ]


def _phase_b_silicon() -> list[Goal]:
    """Silicon smelter cell + 21-belt trunk via row 8 across iron ore.

    Cell anchored at (15, 7): coal_buffer (15, 8), plate-bus (17, 7),
    arm (16, 7), furnace (15, 7). Coal miner (7, 22) UP pushes coal
    into (7, 21). Trunk waypoints: (7, 21) → (7, 8) → (14, 8) →
    (15, 8) sink. The mid-segment crosses iron ore tiles — placement
    on ore is valid (the engine's
    :func:`factoriax.placement.is_valid_placement_tile` only rejects
    ``WATER``/``OUT_OF_BOUNDS``/already-occupied tiles), so cutting
    straight across row 8 above the iron cell saves 6 belts vs. the
    earlier detour through col 6.

    Iron from iron cell, copper from copper cell, tin from tin cell.
    Belt budget: 21 (down from the earlier 27-belt detour).
    """
    return [
        WithdrawFromBusAt(_IRON_PLATE_BUS, ItemType.IRON_PLATE, 23),
        WithdrawFromBusAt(_COPPER_PLATE_BUS, ItemType.COPPER_PLATE, 26),
        WithdrawFromBusAt(_TIN_PLATE_BUS, ItemType.TIN_PLATE, 6),
        ProduceInAssembler(ItemType.WIRE, 4),
        ProduceInAssembler(ItemType.PALLET, 2),
        ProduceInAssembler(ItemType.ARM, 1),
        ProduceInAssembler(ItemType.FURNACE, 1),
        ProduceInAssembler(ItemType.MINER, 1),
        ProduceInAssembler(ItemType.CONVEYOR_BELT, 21),
        *build_smelter_cell_at((15, 7), map_size=_MAP_SIZE),
        # Belts BEFORE coal miner — (7, 21) UP's stand tile (7, 22) is
        # the miner's position. Place trunk while (7, 22) is walkable
        # coal ore, then place the miner last.
        *place_belt_path([(7, 21), (7, 8), (14, 8), (15, 8)]),
        PlaceMachineAt(MachineType.MINER, (7, 22), int(Direction.UP)),
    ]


# ---------------------------------------------------------------------------
# Top-level goal sequence
# ---------------------------------------------------------------------------


def build_advanced_factory_goals() -> list[Goal]:
    """Phase A pre-smelts everything; Phase B builds 4 smelter cells.

    Inter-phase Waits give each cell time to make plates before the
    next phase's WithdrawFromBusAt drains them. The waits are sized
    by trunk length × 1 tick/belt + a couple of plate cycles.
    """
    return [
        *_phase_a_bootstrap_mine_and_smelt(),
        *_phase_a_craft_and_place(),
        *_phase_b_iron(),
        # Iron trunk (14 belts) + plate accumulation: ~80 ticks.
        Wait(80),
        *_phase_b_copper(),
        # Copper trunk (28 belts) + plate accumulation: ~120 ticks.
        Wait(120),
        *_phase_b_tin(),
        # Tin trunk (18 belts) + plate accumulation: ~80 ticks.
        Wait(80),
        *_phase_b_silicon(),
        # Final wait: silicon cell is the last to come online; coal
        # traversal of its 27-belt trunk + a few recipe cycles needs
        # ~150 ticks before plates appear in (17, 7).
        Wait(300),
    ]


def make_advanced_factory_rocket_agent(env_params: EnvParams) -> ScriptedAgent:
    """advanced_factory: scripted agent with Phase A + Phase B (4 cells)."""
    return ScriptedAgent(env_params, Planner(build_advanced_factory_goals()))
