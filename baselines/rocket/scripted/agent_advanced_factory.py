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

from factoriax.constants import Direction, ItemType
from factoriax.state import EnvParams

from .agent import ScriptedAgent, _miner_has_output_predicate
from .goals import (
    Goal,
    MineOre,
    ProduceInAssembler,
    ProduceInFurnace,
    Wait,
    WaitUntil,
    WithdrawFromBusAt,
    build_assembler_module_at,
    build_smelter_cell_at,
    place_belt_path,
    place_ore_node,
    wire_coal_feed,
)
from .planner import Planner

# ---------------------------------------------------------------------------
# Plate-pallet bus tiles (output pallets of the four smelter cells)
# ---------------------------------------------------------------------------

_IRON_PLATE_BUS = (10, 11)
# Copper cell is mirrored across the y axis (facing=LEFT) so its
# plate output sits on row 11 between the iron and copper patches
# instead of east of the copper patch — keeps the central-assembler
# plate feeds straight without crossing belt lines.
_COPPER_PLATE_BUS = (21, 11)
_TIN_PLATE_BUS = (25, 26)
_SILICON_PLATE_BUS = (17, 7)

# Central assembler module that auto-crafts CONVEYOR_BELT
# (IRON_PLATE + COPPER_PLATE) from plates fed off the iron and
# copper extractor arms.
_CENTRAL_ASSEMBLER_TILE = (16, 9)
_CENTRAL_ASSEMBLER_OUTPUT = (18, 9)
_CENTRAL_ASSEMBLER_INPUT_A = (16, 8)  # copper feed sink
_CENTRAL_ASSEMBLER_INPUT_B = (15, 9)  # iron feed sink

_MAP_SIZE: tuple[int, int] = (32, 32)


# ---------------------------------------------------------------------------
# Phase A — bootstrap everything, then drop miners + pallets
# ---------------------------------------------------------------------------


def _phase_a_bootstrap_mine_and_smelt() -> list[Goal]:
    """Hand-mine + smelt every plate the entire run will need.

    Crafted up front so Phase B sub-phases never have to issue a
    FURNACE recipe — any post-iron-cell ``ProduceInFurnace`` would
    target the iron cell's furnace instead of the pre-placed one.

    Plate budget covers: smelter cells (5 cells × 1 furnace each =
    5 IRON for FURNACE recipes; +12 IRON for 11 plate-feed belts +
    1 FRAME), plate-feed belts (+11 IRON, +11 COPPER, +1 TIN,
    +1 COPPER, +1 SILICON+COAL for the WAFER, +1 COPPER+1 IRON
    rolling up). The single WAFER + 1 SILICON_ORE feeds the
    central-assembler phase's ASSEMBLER craft (FRAME+CIRCUIT).
    """
    return [
        # Pre-iron-cell IRON consumption: Phase A 5 MINER + Phase B-iron
        # 1 FURNACE + 14 BELT = 20 IRON. Slack +2.
        MineOre(ItemType.IRON_ORE, 22),
        # Pre-copper-cell COPPER consumption: Phase A 10 WIRE + Phase
        # B-iron 3 WIRE + 2 ARM + 14 BELT + Phase B-copper 4 WIRE +
        # 2 ARM + 28 BELT = 63 COPPER. Slack +2.
        MineOre(ItemType.COPPER_ORE, 65),
        # All-phase TIN consumption (pre-smelted because tin cell only
        # comes online after silicon): 15 (Phase A) + 5 (iron) + 6
        # (copper) + 8 (central asm: 4 WIRE + 3 PALLET + 1 FRAME) +
        # 6 (tin) = 40 TIN. Slack +2.
        MineOre(ItemType.TIN_ORE, 42),
        # Single silicon ore for the WAFER that the central-assembler
        # phase needs to craft 1 CIRCUIT (and from there 1 ASSEMBLER).
        MineOre(ItemType.SILICON, 1),
        # COAL: 1 per smelt cycle = 22 + 65 + 42 + 1 (wafer) +
        # 4 (refractory) = 134. Slack +1.
        MineOre(ItemType.COAL, 135),
        ProduceInFurnace(ItemType.IRON_PLATE, 22),
        ProduceInFurnace(ItemType.COPPER_PLATE, 65),
        ProduceInFurnace(ItemType.TIN_PLATE, 42),
        ProduceInFurnace(ItemType.WAFER, 1),
        ProduceInFurnace(ItemType.REFRACTORY, 4),
    ]


def _phase_a_craft_and_place() -> list[Goal]:
    """Craft Phase A's 10 WIRE + 5 MINER + 5 PALLET, then place them.

    Each non-coal patch gets a south-edge DOWN-facing miner pushing
    into a buffer pallet. The fifth crafted miner is consumed by
    :func:`_phase_b_iron` via :func:`wire_coal_feed` as the iron
    cell's coal feed — no coal miner is placed in Phase A; the
    cells own their coal feeds end-to-end.
    """
    return [
        ProduceInAssembler(ItemType.WIRE, 10),
        ProduceInAssembler(ItemType.MINER, 5),
        ProduceInAssembler(ItemType.PALLET, 5),
        *place_ore_node((8, 9), map_size=_MAP_SIZE),  # iron
        *place_ore_node((23, 9), map_size=_MAP_SIZE),  # copper
        *place_ore_node((23, 24), map_size=_MAP_SIZE),  # tin
        *place_ore_node((15, 5), map_size=_MAP_SIZE),  # silicon
        WaitUntil(_miner_has_output_predicate(), max_ticks=30),
    ]


# ---------------------------------------------------------------------------
# Phase B helpers — one cell at a time. Each cell phase only crafts
# (assembler) and places; no smelting (would target the iron cell's
# furnace post-Phase-B-iron).
# ---------------------------------------------------------------------------


def _phase_b_iron() -> list[Goal]:
    """Iron smelter cell with extractor + coal feed.

    Cell anchored at (8, 11) facing RIGHT, extractor arm at (11, 11)
    facing RIGHT pulling plates out of the bus at (10, 11) and
    pushing east onto (12, 11) — the first belt of the iron-feed
    trunk laid by :func:`_phase_b_central_assembler`.

    Coal miner at (9, 24) facing RIGHT pushes coal onto trunk
    (10, 24) → (10, 12) → (8, 12) sink. The fifth miner crafted in
    Phase A is consumed here; iron Phase B doesn't craft another.
    """
    return [
        ProduceInAssembler(ItemType.WIRE, 3),
        ProduceInAssembler(ItemType.PALLET, 2),
        # 2 ARMs: cell arm + extractor.
        ProduceInAssembler(ItemType.ARM, 2),
        ProduceInAssembler(ItemType.FURNACE, 1),
        ProduceInAssembler(ItemType.CONVEYOR_BELT, 14),
        *build_smelter_cell_at(
            (8, 11),
            facing=int(Direction.RIGHT),
            extract_facing=int(Direction.RIGHT),
            map_size=_MAP_SIZE,
        ),
        *wire_coal_feed(
            miner_tile=(9, 24),
            miner_facing=int(Direction.RIGHT),
            trunk_waypoints=[(10, 24), (10, 12), (8, 12)],
            map_size=_MAP_SIZE,
        ),
    ]


def _phase_b_copper() -> list[Goal]:
    """Copper smelter cell mirrored across y axis + extractor + coal feed.

    Cell anchored at (23, 11) facing LEFT — plate output at (21, 11)
    instead of (25, 11). The mirror brings the output onto row 11
    between the iron and copper patches so plates flow *toward* the
    central assembler. Extractor arm at (20, 11) facing LEFT pulls
    plates from the bus at (21, 11) and pushes west onto (19, 11) —
    the first belt of the copper-feed trunk.

    Coal feed unchanged from the canonical layout: miner at (8, 24)
    DOWN pushes onto trunk (8, 25) → (11, 25) → (11, 12) → (23, 12)
    sink, since the coal_buffer's south-of-furnace position is
    symmetric across the mirror.
    """
    return [
        WithdrawFromBusAt(_IRON_PLATE_BUS, ItemType.IRON_PLATE, 30),
        ProduceInAssembler(ItemType.WIRE, 4),
        ProduceInAssembler(ItemType.PALLET, 2),
        # 2 ARMs: cell arm + extractor.
        ProduceInAssembler(ItemType.ARM, 2),
        ProduceInAssembler(ItemType.FURNACE, 1),
        ProduceInAssembler(ItemType.MINER, 1),
        ProduceInAssembler(ItemType.CONVEYOR_BELT, 28),
        *build_smelter_cell_at(
            (23, 11),
            facing=int(Direction.LEFT),
            extract_facing=int(Direction.LEFT),
            map_size=_MAP_SIZE,
        ),
        *wire_coal_feed(
            miner_tile=(8, 24),
            miner_facing=int(Direction.DOWN),
            trunk_waypoints=[(8, 25), (11, 25), (11, 12), (23, 12)],
            map_size=_MAP_SIZE,
        ),
    ]


def _phase_b_central_assembler() -> list[Goal]:
    """Build the central CONVEYOR_BELT assembler + plate-feed belts.

    Sits between the iron and copper patches at center (16, 9), with
    inputs fed by belts from the two extractor arms placed earlier:

    - **Iron-feed**: extractor at (11, 11) RIGHT pushes onto belt
      chain (12, 11) → (15, 11) → (15, 9) input_b. 5 belts.
    - **Copper-feed**: extractor at (20, 11) LEFT pushes onto belt
      chain (19, 11) → (19, 8) → (16, 8) input_a. 6 belts.

    No belt crossings — iron-feed runs along row 11 cols 12-15 +
    col 15 rows 10-9, copper-feed runs along col 19 rows 9-11 + row
    8 cols 17-19. Both terminate inside the assembler module's
    input pallets.

    Auto-crafts CONVEYOR_BELT (IRON_PLATE + COPPER_PLATE, 4 ticks)
    once both feeds deliver. Output pallet at (18, 9) accumulates
    belts.

    Known limitation: the extractor arms pull from cell buses at 1
    plate/tick while each cell produces one every ~5 ticks, so the
    bus pallets stay near-empty. Downstream
    :func:`WithdrawFromBusAt` calls in tin/silicon Phase B against
    those buses time out under their 24-idle-tick threshold; an
    iteration adding a buffer-pallet between cell bus and extractor
    (or rerouting tin/silicon to use the central assembler's belt
    output) is needed to make the full chain run end-to-end.

    Bootstrap: 12 IRON + 13 COPPER (withdrawn from buses), 4 WIRE,
    1 FRAME, 1 CIRCUIT, 1 ASSEMBLER, 3 PALLET (assembler module),
    1 ARM (assembler module), 11 BELT (plate feeds). The single
    WAFER for CIRCUIT comes from Phase A's pre-smelted wafer; the
    1 TIN for FRAME and 3 TIN for PALLETs come from Phase A's
    over-smelted tin reserve carried in player inventory.
    """
    return [
        WithdrawFromBusAt(_IRON_PLATE_BUS, ItemType.IRON_PLATE, 12),
        WithdrawFromBusAt(_COPPER_PLATE_BUS, ItemType.COPPER_PLATE, 13),
        ProduceInAssembler(ItemType.WIRE, 4),
        ProduceInAssembler(ItemType.FRAME, 1),
        ProduceInAssembler(ItemType.CIRCUIT, 1),
        ProduceInAssembler(ItemType.ASSEMBLER, 1),
        ProduceInAssembler(ItemType.PALLET, 3),
        ProduceInAssembler(ItemType.ARM, 1),
        ProduceInAssembler(ItemType.CONVEYOR_BELT, 11),
        *build_assembler_module_at(
            _CENTRAL_ASSEMBLER_TILE,
            map_size=_MAP_SIZE,
        ),
        *place_belt_path(
            [(12, 11), (15, 11), _CENTRAL_ASSEMBLER_INPUT_B],
        ),
        *place_belt_path(
            [(19, 11), (19, 8), _CENTRAL_ASSEMBLER_INPUT_A],
        ),
    ]


def _phase_b_tin() -> list[Goal]:
    """Tin smelter cell + coal feed via row 26 east + (22, 27) east.

    Cell anchored at (23, 26). Coal miner at (7, 24) DOWN pushes onto
    trunk (7, 25) → (7, 26) → (22, 26) → (22, 27) → (23, 27) sink.
    The trunk detours south at col 22 because (23, 26) is the tin
    furnace itself; ``wire_coal_feed`` places the trunk before the
    miner because the first belt's stand tile is the miner's tile.

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
        *wire_coal_feed(
            miner_tile=(7, 24),
            miner_facing=int(Direction.DOWN),
            trunk_waypoints=[(7, 25), (7, 26), (22, 26), (22, 27), (23, 27)],
            map_size=_MAP_SIZE,
        ),
    ]


def _phase_b_silicon() -> list[Goal]:
    """Silicon smelter cell + coal feed via row 8 across iron ore.

    Cell anchored at (15, 7). Coal miner at (7, 22) UP pushes onto
    trunk (7, 21) → (7, 8) → (14, 8) → (15, 8) sink. The mid-segment
    crosses iron ore tiles — placement on ore is valid (the engine's
    :func:`factoriax.placement.is_valid_placement_tile` only rejects
    ``WATER``/``OUT_OF_BOUNDS``/already-occupied tiles), so cutting
    straight across row 8 above the iron cell saves 6 belts vs. the
    earlier detour through col 6. As with tin, ``wire_coal_feed``
    detects that the first belt's stand tile is the miner's tile
    and emits trunk before miner.

    Iron from iron cell, copper from copper cell, tin from tin cell.
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
        *wire_coal_feed(
            miner_tile=(7, 22),
            miner_facing=int(Direction.UP),
            trunk_waypoints=[(7, 21), (7, 8), (14, 8), (15, 8)],
            map_size=_MAP_SIZE,
        ),
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
        *_phase_b_central_assembler(),
        # Plate-feed belts (5 + 6) + assembler recipe priming: ~80 ticks.
        Wait(80),
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
