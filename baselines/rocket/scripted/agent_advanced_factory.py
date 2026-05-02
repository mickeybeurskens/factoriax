"""Advanced-factory rocket agent.

Pipeline shape: Phase 0 bootstrap (hand-mine + smelt + craft at
the pre-placed furnace + assembler) followed by Phase 1
(iron / copper / tin / silicon automated smelter cells), Phase 2
(WIRE assembler module + tin SPLITTER feeding it), and Phase 3
(FRAME assembler with an iron extractor + iron / tin routes that
bypass the pre-placed F+A drain zone). See the "Phase 3.FRAME"
geometry block below for why iron routes around col 19.

1. Hand-mines a starter inventory sized for *every* placement
   Phase 1 will make, smelts each plate, then crafts every machine
   the cells + coal trunks + supporting infrastructure consume.
2. For each ore (iron, copper) drops one ore-MINER + 2 belts
   feeding the smelter ore_pallet, places the smelter cell
   (furnace + arm + plate-bus PALLET + coal-buffer PALLET), then a
   1-miner + 6-belt coal trunk on its dedicated row feeding the
   coal-buffer.
3. After each cell, verifies the placements landed and waits for
   the miners to spin up.

Phase 0 quantities are recipe-driven: targets feed
:func:`bill_of_materials` and :func:`production_schedule`, so a
recipe rebalance auto-resizes mining / smelting / crafting without
hand edits.

Map layout the agent assumes (v2; see
:func:`factoriax.benchmarks.rocket.build_rocket_level`)::

      0 1 2 3 4 5 6 7 8 9 ...
    9 # · · I I · · . O . . .       (M=miner, .=belt, F=furnace,
   10 # · · I I · · F a P . .        a=arm, P=pallet, O=ore_pallet,
   11 M . . . . . . . P . . .        #=coal column, I=iron patch,
   12 # · · U U · · . O . . .        U=copper patch).
   13 # · · U U · · F a P . .
   14 M . . . . . . . P . . .

Each cell's coal miner sits *on* the coal column at (0, k) facing
RIGHT — its natural stand tile (-1, k) is off-map, so the agent
uses ``PlaceMachineFromBackAt``: stand on the belt at (1, k)
facing LEFT, place (miner inherits LEFT), then ROTATE_RIGHT.
"""

from __future__ import annotations

import dataclasses

from factoriax.constants import Direction, ItemType, MachineType
from factoriax.recipes import BASE_RECIPE_BOOK, RecipeBook
from factoriax.state import EnvParams

from .agent import ScriptedAgent, _miner_has_output_predicate
from .goals import (
    Goal,
    MineOre,
    PlaceMachineAt,
    PlaceMachineFromBackAt,
    ProduceInAssembler,
    ProduceInFurnace,
    VerifyLayout,
    Wait,
    WaitUntil,
    WithdrawFromBusAt,
    assembler_module_inventory,
    build_assembler_module_at,
    build_smelter_cell_at,
    smelter_cell_inventory,
)
from .layout import expected_layout_from_goals
from .planner import Planner
from .recipe_planning import (
    bill_of_materials,
    book_without_recipes_for,
    production_schedule,
    sum_inventories,
)

# Pre-placed machinery from the rocket benchmark level — included in
# every stage's ``VerifyLayout`` expected map so the helper doesn't
# flag them as STRAY.
_PRE_PLACED_LAYOUT: dict[tuple[int, int], tuple[int, int]] = {
    (15, 16): (int(MachineType.FURNACE), int(Direction.DOWN)),
    (17, 16): (int(MachineType.ASSEMBLER), int(Direction.DOWN)),
}

# ---------------------------------------------------------------------------
# Geometry — per-cell smelter spec
# ---------------------------------------------------------------------------

_MAP_SIZE: tuple[int, int] = (32, 32)


@dataclasses.dataclass(frozen=True)
class _SmelterCellSpec:
    """All tiles + facing a single Phase 1 smelter cell touches.

    The shape is identical for iron / copper / tin / silicon: a 1x2
    ore-feed belt run on the patch's top row pushing east into a
    DOWN-facing ore_pallet, a smelter cell anchored at the furnace
    tile (RIGHT-facing), and a 1-miner + 6-belt coal trunk on the
    next dirt row south of the cell.
    """

    label: str  # "iron" / "copper" / ...
    plate_bus_tile: tuple[int, int]
    furnace_tile: tuple[int, int]
    ore_pallet_tile: tuple[int, int]
    ore_miner_tile: tuple[int, int]
    ore_belt_tiles: tuple[tuple[int, int], ...]
    coal_miner_tile: tuple[int, int]
    coal_belt_tiles: tuple[tuple[int, int], ...]

    def cell_inventory(self) -> dict[int, int]:
        """Items the cell consumes when placed (cell + feeders)."""
        return sum_inventories(
            smelter_cell_inventory(),
            {
                int(ItemType.MINER): 2,  # 1 ore + 1 coal
                # +1 belt for the ore feeder (placed at ore_pallet_tile
                # facing DOWN — pushes ore south into the furnace via
                # the directional Phase 0 pull). The miner's east-push
                # lands the ore on this belt.
                int(ItemType.CONVEYOR_BELT): (
                    len(self.ore_belt_tiles) + len(self.coal_belt_tiles) + 1
                ),
            },
        )


_IRON_CELL = _SmelterCellSpec(
    label="iron",
    plate_bus_tile=(9, 10),
    furnace_tile=(7, 10),
    ore_pallet_tile=(7, 9),
    ore_miner_tile=(4, 9),
    ore_belt_tiles=((5, 9), (6, 9)),
    coal_miner_tile=(0, 11),
    coal_belt_tiles=((1, 11), (2, 11), (3, 11), (4, 11), (5, 11), (6, 11)),
)

# Copper sits one ore-patch slot south. Cell anchor on row 13 mirrors
# iron's row 10. Coal trunk on row 14 (the dirt gap between copper
# rows 12-13 and tin rows 15-16).
_COPPER_CELL = _SmelterCellSpec(
    label="copper",
    plate_bus_tile=(9, 13),
    furnace_tile=(7, 13),
    ore_pallet_tile=(7, 12),
    ore_miner_tile=(4, 12),
    ore_belt_tiles=((5, 12), (6, 12)),
    coal_miner_tile=(0, 14),
    coal_belt_tiles=((1, 14), (2, 14), (3, 14), (4, 14), (5, 14), (6, 14)),
)

# Tin patch on rows 15-16. Cell anchor (7, 16); coal trunk on row 17.
_TIN_CELL = _SmelterCellSpec(
    label="tin",
    plate_bus_tile=(9, 16),
    furnace_tile=(7, 16),
    ore_pallet_tile=(7, 15),
    ore_miner_tile=(4, 15),
    ore_belt_tiles=((5, 15), (6, 15)),
    coal_miner_tile=(0, 17),
    coal_belt_tiles=((1, 17), (2, 17), (3, 17), (4, 17), (5, 17), (6, 17)),
)

# Silicon patch on rows 18-19. Cell anchor (7, 19); coal trunk on
# row 20. Furnace recipe SILICON + COAL → WAFER fits the same cell
# shape unchanged; the cell is recipe-agnostic.
_SILICON_CELL = _SmelterCellSpec(
    label="silicon",
    plate_bus_tile=(9, 19),
    furnace_tile=(7, 19),
    ore_pallet_tile=(7, 18),
    ore_miner_tile=(4, 18),
    ore_belt_tiles=((5, 18), (6, 18)),
    coal_miner_tile=(0, 20),
    coal_belt_tiles=((1, 20), (2, 20), (3, 20), (4, 20), (5, 20), (6, 20)),
)

_PHASE_1_CELLS: tuple[_SmelterCellSpec, ...] = (
    _IRON_CELL,
    _COPPER_CELL,
    _TIN_CELL,
    _SILICON_CELL,
)

# Convenience aliases kept stable for external references (tests).
_IRON_PLATE_BUS_TILE: tuple[int, int] = _IRON_CELL.plate_bus_tile
_COPPER_PLATE_BUS_TILE: tuple[int, int] = _COPPER_CELL.plate_bus_tile
_TIN_PLATE_BUS_TILE: tuple[int, int] = _TIN_CELL.plate_bus_tile
_SILICON_PLATE_BUS_TILE: tuple[int, int] = _SILICON_CELL.plate_bus_tile


def _phase_1_targets() -> dict[int, int]:
    """Return ``{ItemType: count}`` for every Phase 1 placement."""
    return sum_inventories(*[c.cell_inventory() for c in _PHASE_1_CELLS])


# ---------------------------------------------------------------------------
# Geometry — Phase 2.WIRE (first inter-smelter route)
# ---------------------------------------------------------------------------
#
# WIRE recipe: 1 COPPER_PLATE + 1 TIN_PLATE -> 1 WIRE in an assembler.
# The cell pulls plates straight off the copper and tin smelter
# plate-buses with extractor arms, then routes them east along
# rows 13 (copper) and 16 (tin) to a single 2-input assembler
# module centered at (13, 14).
#
# The extractor arms sit *east* of each plate-bus (col 10) facing
# RIGHT so they pull from the bus and push onto the first east-
# bound belt. Their natural stand tile (col 9) is the bus itself
# (non-walkable PALLET), so they go in via ``PlaceMachineFromBackAt``
# from the east side, same trick as the coal miners on x=0.

_WIRE_ASSEMBLER_TILE: tuple[int, int] = (13, 14)
_WIRE_INPUT_A_TILE: tuple[int, int] = (13, 13)  # COPPER_PLATE (north)
_WIRE_INPUT_B_TILE: tuple[int, int] = (12, 14)  # TIN_PLATE   (west)
_WIRE_OUTPUT_TILE: tuple[int, int] = (15, 14)  # WIRE pallet  (east)
# arm at (14, 14) is internal to build_assembler_module_at.

# Copper extractor arm + route into a horizontal splitter that fans
# plates DOWN to the WIRE input_a feeder at (13, 13) and UP to the
# CIRCUIT-bound corridor on row 11 (consumed by Phase 3.CIRCUIT).
# Routing the copper UP off the extractor row into row 12 lets us
# place the splitter where its DOWN output lands directly on WIRE's
# input_a feeder without any extra belt — and the UP output corridor
# on row 11 is fully clear east to col 18.
_COPPER_EXTRACT_ARM_TILE: tuple[int, int] = (10, 13)
_COPPER_PRE_SPLITTER_BELTS: tuple[tuple[tuple[int, int], int], ...] = (
    ((11, 13), int(Direction.UP)),  # extractor pushes east; belt routes UP
    ((11, 12), int(Direction.RIGHT)),
    ((12, 12), int(Direction.RIGHT)),
)
_COPPER_SPLITTER_TILE: tuple[int, int] = (13, 12)

# Tin extractor arm pushes east into a SPLITTER, which fans plates
# UP (-> WIRE) and DOWN (-> future FRAME route).
_TIN_EXTRACT_ARM_TILE: tuple[int, int] = (10, 16)
_TIN_SPLITTER_TILE: tuple[int, int] = (11, 16)
# Splitter UP output (11, 15) -> belt RIGHT -> (12, 15) UP ->
# WIRE input_b at (12, 14).
_TIN_TO_WIRE_BELTS: tuple[tuple[tuple[int, int], int], ...] = (
    ((11, 15), int(Direction.RIGHT)),
    ((12, 15), int(Direction.UP)),
)


# ---------------------------------------------------------------------------
# Geometry — Phase 3.FRAME (iron + tin -> frame)
# ---------------------------------------------------------------------------
#
# FRAME recipe: 1 IRON_PLATE + 1 TIN_PLATE -> 1 FRAME, in an
# assembler.
#
# The pre-placed FURNACE (15, 16) and ASSEMBLER (17, 16) both
# auto-pull from every adjacent buffer regardless of recipe match
# (verified empirically: the pre-placed assembler accumulated
# 1150 IRON_PLATE in input slot 0 with slot 1 empty, then stalled).
# Their combined drain zone is the union of their 4-neighbours::
#
#     (14, 16) (15, 15) (15, 17) (16, 16)   -- F neighbours
#     (16, 16) (17, 15) (17, 17) (18, 16)   -- A neighbours
#
# i.e. an L-shape spanning cols 14-18 across rows 15-17. Any plate
# belt or pallet on those tiles loses its plate to F or A.
#
# Workaround: the first fully-clear N-S column east of the L is
# col 19. Iron extracts at (10, 10), runs row 10 east to col 19,
# drops south down col 19 into FRAME's input_a at (19, 21). Tin
# reuses the existing SPLITTER's DOWN output at (11, 17), runs
# south down col 11 then east on row 22 into FRAME's input_b at
# (18, 22).
#
# The tin trunk's first belt at (11, 17) needs ``PlaceMachineFrom-
# BackAt`` because its natural stand tile (11, 16) is the tin
# splitter (non-walkable). Every other belt's stand tile is dirt
# or a previously-placed walkable belt at place time.

_FRAME_ASSEMBLER_TILE: tuple[int, int] = (19, 22)
_FRAME_INPUT_A_TILE: tuple[int, int] = (19, 21)  # IRON_PLATE (north)
_FRAME_INPUT_B_TILE: tuple[int, int] = (18, 22)  # TIN_PLATE  (west)
_FRAME_OUTPUT_TILE: tuple[int, int] = (21, 22)
_IRON_EXTRACT_ARM_TILE: tuple[int, int] = (10, 10)

# Iron route belts in placement order (sink-first). Every belt's
# stand tile (target - unit(facing)) is dirt or a prior walkable
# belt. The bend at (19, 10) faces DOWN so it accepts the east
# push from (18, 10) RIGHT and pushes south down col 19. Two tiles
# on col 19 are *not* in this list — Phase 3.MOTOR places a CROSSING
# at (19, 12) and Phase 3.SENSOR places a CROSSING at (19, 20). Both
# crossings carry iron DOWN through their vertical lane unchanged;
# the horizontal lanes carry wire / CIRCUIT respectively.
_IRON_TO_FRAME_BELTS: tuple[tuple[tuple[int, int], int], ...] = (
    ((19, 19), int(Direction.DOWN)),
    ((19, 18), int(Direction.DOWN)),
    ((19, 17), int(Direction.DOWN)),
    ((19, 16), int(Direction.DOWN)),
    ((19, 15), int(Direction.DOWN)),
    ((19, 14), int(Direction.DOWN)),
    ((19, 13), int(Direction.DOWN)),
    ((19, 11), int(Direction.DOWN)),
    ((19, 10), int(Direction.DOWN)),
    ((18, 10), int(Direction.RIGHT)),
    ((17, 10), int(Direction.RIGHT)),
    ((16, 10), int(Direction.RIGHT)),
    ((15, 10), int(Direction.RIGHT)),
    ((14, 10), int(Direction.RIGHT)),
    ((13, 10), int(Direction.RIGHT)),
    ((12, 10), int(Direction.RIGHT)),
    ((11, 10), int(Direction.RIGHT)),
)

# Tin -> FRAME belts, sink-first. The trunk's source tile at
# (11, 17) is *not* placed by FRAME — Phase 3.CIRCUIT lands a
# CROSSING there instead so the wafer route can cross the tin
# trunk on row 17 (vertical lane = tin DOWN, horizontal lane =
# wafer RIGHT). The CROSSING substitutes for the BELT-from-back
# the FRAME phase used to emit; tin still flows into (11, 18).
_TIN_TO_FRAME_BELTS: tuple[tuple[tuple[int, int], int], ...] = (
    ((17, 22), int(Direction.RIGHT)),
    ((16, 22), int(Direction.RIGHT)),
    ((15, 22), int(Direction.RIGHT)),
    ((14, 22), int(Direction.RIGHT)),
    ((13, 22), int(Direction.RIGHT)),
    ((12, 22), int(Direction.RIGHT)),
    ((11, 22), int(Direction.RIGHT)),
    ((11, 21), int(Direction.DOWN)),
    ((11, 20), int(Direction.DOWN)),
    ((11, 19), int(Direction.DOWN)),
    ((11, 18), int(Direction.DOWN)),
)


# ---------------------------------------------------------------------------
# Geometry — Phase 3.CIRCUIT (copper + wafer -> circuit)
# ---------------------------------------------------------------------------
#
# CIRCUIT recipe: 1 COPPER_PLATE + 1 WAFER -> 1 CIRCUIT, in an
# assembler.
#
# Copper feed: tap off the WIRE-bound copper splitter at (13, 12).
# Its UP output lands on (13, 11); the CIRCUIT phase routes that
# east along row 11 (the dirt corridor between the iron / copper
# patches and iron-to-FRAME's row-10 east-bound trunk) to col 16,
# then south down col 16 (which is one column west of the F+A
# drain zone — col 16 row 16 sits between the pre-placed FURNACE
# at (15, 16) and ASSEMBLER at (17, 16) but is safe because the
# belt there faces DOWN, not LEFT/RIGHT toward either combiner,
# so the directional Phase 0 pull doesn't trigger). The route
# terminates at the CIRCUIT module's input_a feeder belt at
# (16, 17).
#
# Wafer feed: extract from the silicon plate-bus at (9, 19) by
# placing an arm at (9, 18) facing UP via ``PlaceMachineFromBackAt``
# (its natural stand tile is the bus PALLET, non-walkable). The
# arm pushes UP onto (9, 17); the route then runs east along row 17
# (above the silicon and tin coal trunks on row 17 col 1-7), passes
# through a CROSSING at (11, 17) that shares the tile with the tin
# -> FRAME trunk's vertical lane, and bends DOWN at col 14 onto
# row 18 into the CIRCUIT module's input_b feeder belt at (15, 18).
#
# CIRCUIT module sits at (16, 18) facing DOWN — east of the WIRE
# module and west of the iron-to-FRAME col-19 trunk. Output PALLET
# at (18, 18) is checked for CIRCUITs by the smoke test.

_CIRCUIT_ASSEMBLER_TILE: tuple[int, int] = (16, 18)
_CIRCUIT_INPUT_A_TILE: tuple[int, int] = (16, 17)  # COPPER_PLATE (north)
_CIRCUIT_INPUT_B_TILE: tuple[int, int] = (15, 18)  # WAFER       (west)
_CIRCUIT_OUTPUT_TILE: tuple[int, int] = (18, 18)
_WAFER_EXTRACT_ARM_TILE: tuple[int, int] = (9, 18)

# Copper splitter UP -> CIRCUIT. Belts in placement order (sink-first
# toward (13, 11) where the splitter dumps copper): col 16 south-to-
# north, then row 11 east-to-west. (16, 12) is *not* in this list —
# Phase 3.MOTOR places a CROSSING there so its WIRE -> MOTOR route
# can cross col 16 at row 12 (vertical lane carries copper DOWN
# unchanged; horizontal lane carries wire RIGHT).
_COPPER_TO_CIRCUIT_BELTS: tuple[tuple[tuple[int, int], int], ...] = (
    ((16, 16), int(Direction.DOWN)),
    ((16, 15), int(Direction.DOWN)),
    ((16, 14), int(Direction.DOWN)),
    ((16, 13), int(Direction.DOWN)),
    ((16, 11), int(Direction.DOWN)),
    ((15, 11), int(Direction.RIGHT)),
    ((14, 11), int(Direction.RIGHT)),
    ((13, 11), int(Direction.RIGHT)),
)

# Wafer extractor UP -> row 17 east -> CROSSING -> bend DOWN to
# (15, 18). Sink-first placement order keeps every stand tile dirt
# or a previously-placed walkable belt:
# - (14, 18) RIGHT     (final approach into input_b from the west)
# - (14, 17) DOWN      (bend off row 17)
# - (13, 17), (12, 17) RIGHT (post-crossing eastward)
# - CROSSING (11, 17)  (separately, after both side belts so its
#                       dir=1 stand-tile (12, 17) is already a belt)
# - (10, 17), (9, 17)  RIGHT (pre-crossing east-flow off the wafer
#                       extractor)
_WAFER_CROSSING_TILE: tuple[int, int] = (11, 17)
_WAFER_PRE_CROSSING_BELTS: tuple[tuple[tuple[int, int], int], ...] = (
    ((10, 17), int(Direction.RIGHT)),
    ((9, 17), int(Direction.RIGHT)),
)
_WAFER_POST_CROSSING_BELTS: tuple[tuple[tuple[int, int], int], ...] = (
    ((14, 18), int(Direction.RIGHT)),
    ((14, 17), int(Direction.DOWN)),
    ((13, 17), int(Direction.RIGHT)),
    ((12, 17), int(Direction.RIGHT)),
)


# ---------------------------------------------------------------------------
# Geometry — Phase 3.MOTOR (frame + wire -> motor)
# ---------------------------------------------------------------------------
#
# MOTOR recipe: 1 FRAME + 1 WIRE -> 1 MOTOR, in an assembler.
#
# MOTOR module sits east of the FRAME cell at (24, 22) so its
# input_b feeder belt (23, 22) catches a single arm-push directly
# off the FRAME output PALLET at (21, 22) — no FRAME-route belts
# are needed, just one extractor arm at (22, 22) facing RIGHT
# (from-back, since the natural west stand tile is the PALLET).
#
# WIRE comes off the WIRE output PALLET at (15, 14) via an extractor
# arm at (15, 13) facing UP (from-back, since the natural south
# stand tile is the PALLET itself). The arm pushes WIRE UP onto
# (15, 12); the route then runs east along row 12 through two
# CROSSINGs:
#
# * CROSSING at (16, 12) dir=1 — vertical lane carries CIRCUIT-bound
#   copper DOWN unchanged (substitutes for the BELT the CIRCUIT phase
#   used to emit there); horizontal lane carries wire RIGHT.
# * CROSSING at (19, 12) dir=1 — vertical lane carries iron DOWN
#   unchanged (substitutes for the BELT the FRAME phase used to emit
#   there); horizontal lane carries wire RIGHT.
#
# After (19, 12) the route continues east on row 12 to col 24 then
# DOWN col 24 to the MOTOR input_a feeder at (24, 21).
#
# Pushing wire UP off the PALLET (instead of east, the natural
# extractor direction used by the iron / copper / tin extractors)
# avoids a 3-way collision: the WIRE pallet's east neighbour (16, 14)
# is already the CIRCUIT copper trunk's south-bound belt, and the
# (16, 12) CROSSING reuses col 16's vertical copper lane to ferry
# the wire eastward without a fourth CIRCUIT detour.

_MOTOR_ASSEMBLER_TILE: tuple[int, int] = (24, 22)
_MOTOR_INPUT_A_TILE: tuple[int, int] = (24, 21)  # WIRE  (north)
_MOTOR_INPUT_B_TILE: tuple[int, int] = (23, 22)  # FRAME (west)
_MOTOR_OUTPUT_TILE: tuple[int, int] = (26, 22)
_WIRE_EXTRACT_ARM_TILE: tuple[int, int] = (15, 13)
_FRAME_EXTRACT_ARM_TILE: tuple[int, int] = (22, 22)
_WIRE_COPPER_CROSSING_TILE: tuple[int, int] = (16, 12)
_WIRE_IRON_CROSSING_TILE: tuple[int, int] = (19, 12)

# WIRE -> MOTOR belts in placement order (sink-first toward the
# extractor at (15, 13) UP). The two CROSSINGs at (16, 12) and
# (19, 12) are placed separately, between the corresponding flank
# belts, so each crossing's stand tile (target - unit(LEFT) =
# target + (1, 0)) is already a walkable belt. Two further tiles
# on this trunk are *not* in this list — they are placed by
# Phase 3.SENSOR:
#
# * (24, 12) is a SPLITTER (horizontal RIGHT) that fans wire DOWN
#   to MOTOR (continuing the existing col-24 trunk) and UP to a new
#   SENSOR-bound corridor on row 11.
# * (24, 20) is a CROSSING dir=1 that lets CIRCUIT-to-SENSOR cross
#   the col-24 wire trunk (vertical lane carries wire DOWN; horizontal
#   lane carries CIRCUIT RIGHT).
_WIRE_TO_MOTOR_BELTS: tuple[tuple[tuple[int, int], int], ...] = (
    # Col 24 sink-first: south-end belt (24, 20) feeds the input_a
    # feeder at (24, 21); each belt's stand tile is the next belt
    # north (or dirt at the row 12 end). (24, 12) and (24, 20) are
    # owned by Phase 3.SENSOR (SPLITTER and CROSSING respectively).
    ((24, 19), int(Direction.DOWN)),
    ((24, 18), int(Direction.DOWN)),
    ((24, 17), int(Direction.DOWN)),
    ((24, 16), int(Direction.DOWN)),
    ((24, 15), int(Direction.DOWN)),
    ((24, 14), int(Direction.DOWN)),
    ((24, 13), int(Direction.DOWN)),
    # Row 12 east-flow into col 24 (sink-first toward the iron
    # CROSSING).
    ((23, 12), int(Direction.RIGHT)),
    ((22, 12), int(Direction.RIGHT)),
    ((21, 12), int(Direction.RIGHT)),
    ((20, 12), int(Direction.RIGHT)),
)
# Belts placed *after* the iron CROSSING at (19, 12) so the next
# CROSSING (16, 12) sees its (17, 12) stand tile as a fresh belt.
_WIRE_TO_MOTOR_MID_BELTS: tuple[tuple[tuple[int, int], int], ...] = (
    ((18, 12), int(Direction.RIGHT)),
    ((17, 12), int(Direction.RIGHT)),
)
# Final belt placed after the copper CROSSING at (16, 12); the WIRE
# extractor arm at (15, 13) UP stands on (15, 12).
_WIRE_TO_MOTOR_TAIL_BELTS: tuple[tuple[tuple[int, int], int], ...] = (
    ((15, 12), int(Direction.RIGHT)),
)


# ---------------------------------------------------------------------------
# Geometry — Phase 3.SENSOR (circuit + wire -> sensor)
# ---------------------------------------------------------------------------
#
# SENSOR recipe: 1 CIRCUIT + 1 WIRE -> 1 SENSOR, in an assembler.
#
# SENSOR module sits at (28, 18) — east of FRAME's col-19 trunk and
# south-east of MOTOR. WIRE comes off a SPLITTER at (24, 12) that
# replaces the wire belt on the WIRE -> MOTOR trunk; the splitter's
# DOWN output continues the existing MOTOR feed unchanged, while
# its UP output drives a new corridor along row 11 east, then south
# down col 28 into the SENSOR input_a feeder at (28, 17).
#
# CIRCUIT is extracted from the CIRCUIT output PALLET at (18, 18)
# by an arm at (18, 19) facing DOWN (from-back, since the natural
# stand tile is the PALLET). The arm pushes CIRCUIT onto (18, 20);
# the route runs east along row 20 through two CROSSINGs:
#
# * CROSSING at (19, 20) dir=1 — vertical lane carries iron DOWN
#   unchanged (substitutes for the BELT the FRAME phase used to
#   emit there); horizontal lane carries CIRCUIT RIGHT.
# * CROSSING at (24, 20) dir=1 — vertical lane carries WIRE DOWN
#   to MOTOR (substitutes for the BELT the MOTOR phase used to
#   emit there); horizontal lane carries CIRCUIT RIGHT.
#
# After (24, 20) the route continues east on row 20 to col 27, then
# bends UP col 27 to land on the SENSOR input_b feeder at (27, 18).

_SENSOR_ASSEMBLER_TILE: tuple[int, int] = (28, 18)
_SENSOR_INPUT_A_TILE: tuple[int, int] = (28, 17)  # WIRE    (north)
_SENSOR_INPUT_B_TILE: tuple[int, int] = (27, 18)  # CIRCUIT (west)
_SENSOR_OUTPUT_TILE: tuple[int, int] = (30, 18)
_CIRCUIT_EXTRACT_ARM_TILE: tuple[int, int] = (18, 19)
_WIRE_SPLITTER_TILE: tuple[int, int] = (24, 12)
_CIRCUIT_IRON_CROSSING_TILE: tuple[int, int] = (19, 20)
_CIRCUIT_WIRE_CROSSING_TILE: tuple[int, int] = (24, 20)

# WIRE splitter UP -> SENSOR. Belts in placement order (sink-first
# from (28, 16) back to the splitter at (24, 12)):
_WIRE_TO_SENSOR_BELTS: tuple[tuple[tuple[int, int], int], ...] = (
    ((28, 16), int(Direction.DOWN)),
    ((28, 15), int(Direction.DOWN)),
    ((28, 14), int(Direction.DOWN)),
    ((28, 13), int(Direction.DOWN)),
    ((28, 12), int(Direction.DOWN)),
    ((28, 11), int(Direction.DOWN)),
    ((27, 11), int(Direction.RIGHT)),
    ((26, 11), int(Direction.RIGHT)),
    ((25, 11), int(Direction.RIGHT)),
    ((24, 11), int(Direction.RIGHT)),
)

# CIRCUIT extractor DOWN -> row 20 east -> col 27 UP -> SENSOR
# input_b. Belts split into three batches around the two CROSSINGs
# so each crossing's stand tile (target - unit(LEFT) = target +
# (1, 0)) is already a walkable belt at placement time.
_CIRCUIT_TO_SENSOR_BELTS: tuple[tuple[tuple[int, int], int], ...] = (
    # Col 27 UP into input_b feeder at (27, 18). Sink-first:
    ((27, 19), int(Direction.UP)),
    ((27, 20), int(Direction.UP)),
    # Row 20 west-to-east approach to col 27 — sink-first toward
    # the wire CROSSING at (24, 20).
    ((26, 20), int(Direction.RIGHT)),
    ((25, 20), int(Direction.RIGHT)),
)
_CIRCUIT_TO_SENSOR_MID_BELTS: tuple[tuple[tuple[int, int], int], ...] = (
    # Belts between the wire CROSSING at (24, 20) and the iron
    # CROSSING at (19, 20). Sink-first toward the iron CROSSING.
    ((23, 20), int(Direction.RIGHT)),
    ((22, 20), int(Direction.RIGHT)),
    ((21, 20), int(Direction.RIGHT)),
    ((20, 20), int(Direction.RIGHT)),
)
_CIRCUIT_TO_SENSOR_TAIL_BELTS: tuple[tuple[tuple[int, int], int], ...] = (
    # Final belt between the iron CROSSING and the extractor arm.
    ((18, 20), int(Direction.RIGHT)),
)


def _phase_2_wire_targets() -> dict[int, int]:
    """Return ``{ItemType: count}`` for the WIRE cell + plate routes.

    Counts: 1 ASSEMBLER + 1 PALLET + 1 ARM + 2 BELT (assembler module
    via :func:`assembler_module_inventory`) + 2 ARM (copper + tin
    extractors) + 2 SPLITTER (copper splitter at (13, 12) fanning to
    WIRE+CIRCUIT, tin splitter at (11, 16) fanning to WIRE+FRAME) +
    N CONVEYOR_BELT (pre-splitter copper belts + tin route belts).
    The CIRCUIT-bound copper belts past the splitter are owned by
    Phase 3.CIRCUIT.
    """
    return sum_inventories(
        assembler_module_inventory(input_b=True),
        {
            int(ItemType.ARM): 2,
            int(ItemType.SPLITTER): 2,
            int(ItemType.CONVEYOR_BELT): (
                len(_COPPER_PRE_SPLITTER_BELTS) + len(_TIN_TO_WIRE_BELTS)
            ),
        },
    )


def _phase_3_frame_targets() -> dict[int, int]:
    """Return ``{ItemType: count}`` for the FRAME cell + plate routes.

    Counts: 1 ASSEMBLER + 1 PALLET + 1 ARM + 2 BELT (assembler module
    via :func:`assembler_module_inventory`) + 1 ARM (iron extractor) +
    N CONVEYOR_BELT (iron route + tin route). The tin extractor +
    splitter were placed in Phase 2; the tin trunk head at (11, 17)
    is placed by Phase 3.CIRCUIT as a CROSSING (so wafer can cross
    the tin trunk on its way east to the CIRCUIT cell).
    """
    return sum_inventories(
        assembler_module_inventory(input_b=True),
        {
            int(ItemType.ARM): 1,
            int(ItemType.CONVEYOR_BELT): (
                len(_IRON_TO_FRAME_BELTS) + len(_TIN_TO_FRAME_BELTS)
            ),
        },
    )


def _phase_3_circuit_targets() -> dict[int, int]:
    """Return ``{ItemType: count}`` for the CIRCUIT cell + plate routes.

    Counts: 1 ASSEMBLER + 1 PALLET + 1 ARM + 2 BELT (assembler module
    via :func:`assembler_module_inventory`) + 1 ARM (wafer extractor)
    + 1 CROSSING (tin / wafer cross at (11, 17), substituting for the
    BELT the FRAME phase used to emit there) + N CONVEYOR_BELT
    (copper-to-CIRCUIT branch off the splitter + wafer route).
    """
    return sum_inventories(
        assembler_module_inventory(input_b=True),
        {
            int(ItemType.ARM): 1,
            int(ItemType.CROSSING): 1,
            int(ItemType.CONVEYOR_BELT): (
                len(_COPPER_TO_CIRCUIT_BELTS)
                + len(_WAFER_PRE_CROSSING_BELTS)
                + len(_WAFER_POST_CROSSING_BELTS)
            ),
        },
    )


def _phase_3_motor_targets() -> dict[int, int]:
    """Return ``{ItemType: count}`` for the MOTOR cell + plate routes.

    Counts: 1 ASSEMBLER + 1 PALLET + 1 ARM + 2 BELT (assembler module
    via :func:`assembler_module_inventory`) + 2 ARM (wire + frame
    extractors) + 2 CROSSING (wire / copper cross at (16, 12) and
    wire / iron cross at (19, 12), substituting for the BELTs the
    CIRCUIT and FRAME phases used to emit there) + N CONVEYOR_BELT
    (wire route on row 12 + col 24 south). The FRAME extractor arm
    pushes directly onto the MOTOR input_b feeder belt at (23, 22),
    so no FRAME-route belts are needed.
    """
    return sum_inventories(
        assembler_module_inventory(input_b=True),
        {
            int(ItemType.ARM): 2,
            int(ItemType.CROSSING): 2,
            int(ItemType.CONVEYOR_BELT): (
                len(_WIRE_TO_MOTOR_BELTS)
                + len(_WIRE_TO_MOTOR_MID_BELTS)
                + len(_WIRE_TO_MOTOR_TAIL_BELTS)
            ),
        },
    )


def _phase_3_sensor_targets() -> dict[int, int]:
    """Return ``{ItemType: count}`` for the SENSOR cell + plate routes.

    Counts: 1 ASSEMBLER + 1 PALLET + 1 ARM + 2 BELT (assembler module
    via :func:`assembler_module_inventory`) + 1 ARM (CIRCUIT extractor;
    WIRE comes off the splitter, not a dedicated arm) + 1 SPLITTER
    (wire fan-out at (24, 12) — substitutes for the BELT the MOTOR
    phase used to emit there) + 2 CROSSING ((19, 20) iron / CIRCUIT
    cross, substituting for the BELT the FRAME phase used to emit
    there; (24, 20) wire / CIRCUIT cross, substituting for the BELT
    the MOTOR phase used to emit there) + N CONVEYOR_BELT (wire +
    CIRCUIT routes).
    """
    return sum_inventories(
        assembler_module_inventory(input_b=True),
        {
            int(ItemType.ARM): 1,
            int(ItemType.SPLITTER): 1,
            int(ItemType.CROSSING): 2,
            int(ItemType.CONVEYOR_BELT): (
                len(_WIRE_TO_SENSOR_BELTS)
                + len(_CIRCUIT_TO_SENSOR_BELTS)
                + len(_CIRCUIT_TO_SENSOR_MID_BELTS)
                + len(_CIRCUIT_TO_SENSOR_TAIL_BELTS)
            ),
        },
    )


# Plate-leaf items: in Phase 0b the agent withdraws these from the
# Phase 1 cell plate-buses instead of hand-smelting them, so the
# truncated recipe book treats them as raw leaves for the BOM walk.
_PLATE_LEAVES: frozenset[int] = frozenset(
    {
        int(ItemType.IRON_PLATE),
        int(ItemType.COPPER_PLATE),
        int(ItemType.TIN_PLATE),
        int(ItemType.WAFER),
    }
)

# Per-plate-leaf cell pallet — withdraw from these in Phase 0b.
_PLATE_PALLETS: dict[int, tuple[int, int]] = {
    int(ItemType.IRON_PLATE): _IRON_PLATE_BUS_TILE,
    int(ItemType.COPPER_PLATE): _COPPER_PLATE_BUS_TILE,
    int(ItemType.TIN_PLATE): _TIN_PLATE_BUS_TILE,
    int(ItemType.WAFER): _SILICON_PLATE_BUS_TILE,
}

# Inventory-cap-aware chunk size for Phase 0b crafts. The player can
# hold up to 99 of each item; we withdraw at most this many plates of
# any one type before crafting, keeping headroom under the cap so a
# stray bus-pallet over-pull never stalls the goal. 25 also keeps each
# WithdrawFromBusAt's wait short — a cell smelts ~1 plate every 2
# ticks, so 25 plates land in ~50 ticks.
_PHASE_0B_CHUNK: int = 25


# ---------------------------------------------------------------------------
# Bootstrap slack
# ---------------------------------------------------------------------------

# Per-leaf slack added on top of bill_of_materials to absorb the
# deposit-then-wait race in :class:`ProduceInMachine` (engine auto-
# pull may consume an ore / coal unit between two of the player's
# deposits). Tuned by hand from v1 runs.
_DEFAULT_SLACK: dict[int, int] = {
    int(ItemType.IRON_ORE): 2,
    int(ItemType.COPPER_ORE): 2,
    int(ItemType.TIN_ORE): 2,
    int(ItemType.SILICON): 2,
    int(ItemType.LIMESTONE): 2,
    int(ItemType.COAL): 5,
}


# ---------------------------------------------------------------------------
# Phase 0a — hand-bootstrap items needed to build Phase 1 cells
# ---------------------------------------------------------------------------


def _phase_0a(
    crafted_targets: dict[int, int],
    book: RecipeBook,
    slack: dict[int, int],
) -> list[Goal]:
    """Hand-mine + smelt + craft enough to cover ``crafted_targets``.

    ``crafted_targets`` should be sized for Phase 1 placements only
    (cells + ore-feeders + coal trunks). Phase 2 / Phase 3 placements
    are sourced from the live cell plate-buses by :func:`_phase_0b`,
    avoiding the brute-force hand-mining and hand-smelting of every
    plate the agent ever needs.
    """
    schedule = production_schedule(crafted_targets, book)
    bom = bill_of_materials(crafted_targets, book)
    for item, qty in slack.items():
        bom[int(item)] = bom.get(int(item), 0) + int(qty)

    mine_goals: list[Goal] = [MineOre(item, qty) for item, qty in sorted(bom.items())]

    smelt_goals: list[Goal] = []
    craft_goals: list[Goal] = []
    for output_item, qty, machine_type in schedule:
        if machine_type == int(MachineType.FURNACE):
            smelt_goals.append(ProduceInFurnace(output_item, qty, book=book))
        elif machine_type == int(MachineType.ASSEMBLER):
            craft_goals.append(ProduceInAssembler(output_item, qty, book=book))
        # Other machine types are not produced via Phase 0a (e.g. recipes
        # gated to a specific cell). None of the rocket recipes use them
        # at this stage.

    return [*mine_goals, *smelt_goals, *craft_goals]


# ---------------------------------------------------------------------------
# Phase 0b — bootstrap part 2: withdraw plates from cells, craft Phase 2/3
# ---------------------------------------------------------------------------


def _phase_0b(
    crafted_targets: dict[int, int],
    book: RecipeBook,
    slack: dict[int, int],
) -> list[Goal]:
    """Withdraw plates from cell pallets, mine raw, craft Phase 2/3 items.

    Run *after* the Phase 1 cells are placed and producing. Treats
    plates and wafer as leaves (truncating their smelt recipes from
    the book), so the BOM walk only requests raw resources the cells
    don't produce — namely COAL for the SPLITTER + CROSSING crafts
    that consume coal directly in the assembler.

    For each scheduled craft, the agent:

    1. Withdraws the per-cycle plate input from the corresponding cell
       plate-bus pallet (chunked to :data:`_PHASE_0B_CHUNK` cycles per
       batch so the player's inventory stays under cap).
    2. Runs ``ProduceInMachine`` for the chunk. The machine is the
       pre-placed assembler at (17, 16) for assembler recipes (or the
       pre-placed furnace at (15, 16) for any furnace recipe — none
       in the rocket pipeline today).

    Slack only applies to raw leaves; plates have effectively unlimited
    supply from the cells, so no per-plate slack is needed.
    """
    truncated = book_without_recipes_for(book, set(_PLATE_LEAVES))
    bom = bill_of_materials(crafted_targets, truncated)
    schedule = production_schedule(crafted_targets, truncated)
    for item, qty in slack.items():
        if int(item) in _PLATE_LEAVES:
            continue
        bom[int(item)] = bom.get(int(item), 0) + int(qty)

    goals: list[Goal] = []
    for item, qty in sorted(bom.items()):
        if item in _PLATE_LEAVES or qty <= 0:
            continue
        goals.append(MineOre(item, qty))

    book_lookup = {r.output: r for r in book.recipes}
    for output_item, qty, machine_type in schedule:
        recipe = book_lookup[output_item]
        for chunk_start in range(0, qty, _PHASE_0B_CHUNK):
            chunk_qty = min(_PHASE_0B_CHUNK, qty - chunk_start)
            for input_item, per_craft in recipe.inputs:
                input_id = int(input_item)
                if input_id not in _PLATE_LEAVES:
                    continue
                need = chunk_qty * int(per_craft)
                # WithdrawFromBusAt is absolute ("until held >= count");
                # after the previous chunk's craft consumed the inputs
                # the player holds 0 of this plate, so the goal pulls
                # `need` plates from the cell pallet. max_idle_attempts
                # is generous because the cell may need a few hundred
                # ticks to refill after a large pull.
                goals.append(
                    WithdrawFromBusAt(
                        _PLATE_PALLETS[input_id],
                        input_id,
                        need,
                        max_idle_attempts=400,
                    )
                )
            if int(machine_type) == int(MachineType.FURNACE):
                goals.append(ProduceInFurnace(output_item, chunk_qty, book=book))
            elif int(machine_type) == int(MachineType.ASSEMBLER):
                goals.append(ProduceInAssembler(output_item, chunk_qty, book=book))

    return goals


# ---------------------------------------------------------------------------
# Phase 1 — per-cell smelter + ore feed + coal trunk
# ---------------------------------------------------------------------------


def _phase_1_cell_goals(spec: _SmelterCellSpec) -> list[Goal]:
    """Build the placement goals for a single Phase 1 smelter cell.

    Placement order. Belts are always placed *before* the machines
    that push onto them; otherwise the player would have to stand
    on a non-walkable miner / pallet tile to place the next belt
    east of it.

    1. Ore feed belts ferrying ore from the patch east to the
       ore-feeder belt. Belt 0's stand tile sits on the ore patch
       itself — walkable.
    2. Ore-feeder belt facing DOWN via ``PlaceMachineFromBackAt``.
       This is what the furnace's directional Phase 0 pull reads
       from — it must be a belt facing the furnace, not a pallet.
       The natural (north) stand tile would either be dirt or the
       previous cell's coal-feeder belt (non-walkable); from-back
       uniformly stands the player on the dirt south of the
       feeder, places facing UP, then ROTATEs to DOWN.
    3. Ore miner facing RIGHT pushing east onto the first belt.
       Stand tile is the patch's third column — walkable ore.
    4. Smelter cell at the furnace tile facing RIGHT.
       ``build_smelter_cell_at`` internally orders coal-feeder
       belt -> plate_bus pallet -> arm -> furnace so every stand
       tile is walkable at place time.
    5. Six coal belts feeding the cell's coal-feeder belt. Placed
       before the coal miner so the trunk's first tile has a belt
       to receive the miner's eastward push.
    6. Coal miner on the coal column at x=0 via
       ``PlaceMachineFromBackAt``. Natural stand tile (-1, k) is
       off-map; the from-back variant stands on the belt at
       (1, k) facing LEFT, places (miner inherits LEFT), then
       ROTATEs to RIGHT.

    The caller appends a WaitUntil + VerifyLayout gate to confirm
    the placements landed before the next cell starts (or before
    the trailing Wait that lets plates accumulate).
    """
    goals: list[Goal] = []

    for tile in spec.ore_belt_tiles:
        goals.append(
            PlaceMachineAt(MachineType.CONVEYOR_BELT, tile, int(Direction.RIGHT))
        )
    # Ore feeder is now a belt facing DOWN (south, into the furnace's
    # asm_in slot 0 via Phase 0's directional pull) instead of a
    # pallet — combiners no longer auto-pull from neighbouring
    # buffers, only from facing belts. The miner's east-push lands
    # ore on this belt; the belt holds one tick, then the furnace
    # pulls it south.
    goals.append(
        PlaceMachineFromBackAt(
            MachineType.CONVEYOR_BELT, spec.ore_pallet_tile, int(Direction.DOWN)
        )
    )
    goals.append(
        PlaceMachineAt(MachineType.MINER, spec.ore_miner_tile, int(Direction.RIGHT))
    )
    goals.extend(
        build_smelter_cell_at(
            spec.furnace_tile,
            facing=int(Direction.RIGHT),
            map_size=_MAP_SIZE,
        )
    )
    for tile in spec.coal_belt_tiles:
        goals.append(
            PlaceMachineAt(MachineType.CONVEYOR_BELT, tile, int(Direction.RIGHT))
        )
    goals.append(
        PlaceMachineFromBackAt(
            MachineType.MINER, spec.coal_miner_tile, int(Direction.RIGHT)
        )
    )
    return goals


# ---------------------------------------------------------------------------
# Phase 2.WIRE — assembler module + copper/tin plate routing
# ---------------------------------------------------------------------------


def _phase_2_wire_goals() -> list[Goal]:
    """Place the WIRE assembler + plate routes + both inter-cell splitters.

    Both routes go through a SPLITTER so the same single extractor
    can feed two downstream cells:

    * **copper** splitter at (13, 12) horizontal RIGHT — DOWN output
      lands on the WIRE input_a feeder (13, 13); UP output is consumed
      by Phase 3.CIRCUIT (off the (13, 11) corridor).
    * **tin** splitter at (11, 16) horizontal RIGHT — UP output runs
      east to WIRE input_b; DOWN output drops south into the FRAME
      tin trunk (Phase 3.FRAME).

    Placement order (each step's stand tile is dirt or a previously
    placed walkable belt):

    1. Assembler module at ``_WIRE_ASSEMBLER_TILE``.
    2. Pre-splitter copper belts in source-first order so the splitter
       at (13, 12) has a walkable belt at (12, 12) under it: (11, 13)
       UP, (11, 12) RIGHT, (12, 12) RIGHT.
    3. Copper SPLITTER at (13, 12) RIGHT.
    4. Copper extractor arm at (10, 13) via from-back (stand on the
       (11, 13) UP belt placed in step 2).
    5. Tin -> WIRE route belts (sink-first).
    6. Tin extractor arm at (10, 16) RIGHT via from-back — placed
       *before* the splitter so the from-back stand tile (11, 16)
       is still dirt.
    7. Tin SPLITTER at (11, 16) RIGHT via from-back.
    """
    goals: list[Goal] = []

    goals.extend(
        build_assembler_module_at(
            _WIRE_ASSEMBLER_TILE,
            input_b=True,
            map_size=_MAP_SIZE,
        )
    )

    for tile, facing in _COPPER_PRE_SPLITTER_BELTS:
        goals.append(PlaceMachineAt(MachineType.CONVEYOR_BELT, tile, facing))
    goals.append(
        PlaceMachineAt(
            MachineType.SPLITTER, _COPPER_SPLITTER_TILE, int(Direction.RIGHT)
        )
    )
    goals.append(
        PlaceMachineFromBackAt(
            MachineType.ARM, _COPPER_EXTRACT_ARM_TILE, int(Direction.RIGHT)
        )
    )

    # Tin -> WIRE belts. Place from sink (12, 15) UP back toward the
    # splitter (11, 15) RIGHT so each stand tile stays dirt or a
    # prior walkable belt.
    for tile, facing in reversed(_TIN_TO_WIRE_BELTS):
        goals.append(PlaceMachineAt(MachineType.CONVEYOR_BELT, tile, facing))

    # Tin extractor arm at (10, 16) RIGHT — placed *before* the
    # splitter so the from-back stand tile (11, 16) is still dirt.
    # If the splitter went first, both (10, 16) and (11, 16) would
    # be non-walkable for any subsequent placement here.
    goals.append(
        PlaceMachineFromBackAt(
            MachineType.ARM, _TIN_EXTRACT_ARM_TILE, int(Direction.RIGHT)
        )
    )

    # Tin splitter — both natural stand tiles ((10, 16) is the arm,
    # (12, 16) is dirt) — go from the back: stand at (12, 16)
    # facing LEFT, place splitter inheriting LEFT (still horizontal-
    # facing, same I/O), then ROTATE_RIGHT to settle at facing=RIGHT.
    goals.append(
        PlaceMachineFromBackAt(
            MachineType.SPLITTER, _TIN_SPLITTER_TILE, int(Direction.RIGHT)
        )
    )
    return goals


# ---------------------------------------------------------------------------
# Phase 3.FRAME — assembler + iron extractor + iron and tin routes
# ---------------------------------------------------------------------------


def _phase_3_frame_goals() -> list[Goal]:
    """Place the FRAME assembler module, iron extractor, and routes.

    Iron is extracted from the iron plate-bus at (9, 10) by an
    arm at (10, 10) facing RIGHT (from-back, since the natural
    stand tile is the plate-bus PALLET). Belts run east along
    row 10 to col 19 (the first fully-clear N-S column east of
    the F+A drain zone), then south down col 19 into FRAME's
    input_a at (19, 21).

    Tin reuses the splitter placed in Phase 2: its DOWN output
    drops onto (11, 17). Belts run south down col 11, bend east
    on row 22, and feed FRAME's input_b at (18, 22). The tin
    trunk head at (11, 17) is *not* placed here — Phase 3.CIRCUIT
    lands a CROSSING at that tile (the wafer route shares it),
    and the CROSSING's vertical lane carries tin DOWN unchanged.

    Placement order so every stand tile is walkable:

    1. FRAME assembler module at ``_FRAME_ASSEMBLER_TILE``.
    2. Iron route belts (sink-first), then iron extractor arm.
    3. Tin route belts (sink-first). Phase 3.CIRCUIT will place the
       CROSSING at (11, 17) afterward to complete the tin trunk's
       splitter-to-trunk join.
    """
    goals: list[Goal] = []

    goals.extend(
        build_assembler_module_at(
            _FRAME_ASSEMBLER_TILE,
            input_b=True,
            map_size=_MAP_SIZE,
        )
    )

    for tile, facing in _IRON_TO_FRAME_BELTS:
        goals.append(PlaceMachineAt(MachineType.CONVEYOR_BELT, tile, facing))
    goals.append(
        PlaceMachineFromBackAt(
            MachineType.ARM, _IRON_EXTRACT_ARM_TILE, int(Direction.RIGHT)
        )
    )

    for tile, facing in _TIN_TO_FRAME_BELTS:
        goals.append(PlaceMachineAt(MachineType.CONVEYOR_BELT, tile, facing))
    return goals


# ---------------------------------------------------------------------------
# Phase 3.CIRCUIT — assembler + wafer extractor + copper / wafer routes
# ---------------------------------------------------------------------------


def _phase_3_circuit_goals() -> list[Goal]:
    """Place the CIRCUIT assembler module + copper-tap + wafer route.

    Copper comes off the WIRE-bound splitter at (13, 12)'s UP output:
    a 9-belt route (col 16 south + row 11 west) drops it into the
    CIRCUIT module's input_a feeder at (16, 17). Wafer comes off the
    silicon plate-bus via a from-back arm at (9, 18) facing UP, runs
    east on row 17 through a CROSSING at (11, 17), bends DOWN at
    col 14, and lands on the input_b feeder at (15, 18).

    The CROSSING dir=1 (vertical N->S, horizontal W->E) replaces the
    BELT the previous design placed at (11, 17) via from-back. The
    vertical lane carries tin DOWN from the splitter at (11, 16)
    into the FRAME trunk at (11, 18); the horizontal lane carries
    wafer RIGHT from (10, 17) into (12, 17).

    Placement order (each step's stand tile is dirt or a previously
    placed walkable belt; CROSSINGs and SPLITTERs are not walkable):

    1. CIRCUIT module at ``_CIRCUIT_ASSEMBLER_TILE``.
    2. Copper -> CIRCUIT belts sink-first (col 16 from row 16 up to
       row 11, then row 11 from col 15 west to col 13).
    3. Wafer post-crossing belts sink-first ((14, 18), (14, 17),
       (13, 17), (12, 17)).
    4. CROSSING at (11, 17). Stand = (12, 17) belt (placed in step 3).
    5. Wafer pre-crossing belts ((10, 17), (9, 17)).
    6. Wafer extractor arm at (9, 18) UP via from-back.
    """
    goals: list[Goal] = []

    goals.extend(
        build_assembler_module_at(
            _CIRCUIT_ASSEMBLER_TILE,
            input_b=True,
            map_size=_MAP_SIZE,
        )
    )

    for tile, facing in _COPPER_TO_CIRCUIT_BELTS:
        goals.append(PlaceMachineAt(MachineType.CONVEYOR_BELT, tile, facing))

    for tile, facing in _WAFER_POST_CROSSING_BELTS:
        goals.append(PlaceMachineAt(MachineType.CONVEYOR_BELT, tile, facing))

    # CROSSING dir=1: vertical N->S (tin) + horizontal W->E (wafer).
    # PlaceMachineAt navigates to target - unit(facing); facing=1
    # (LEFT) means stand = (12, 17), which is the wafer post-crossing
    # belt placed above.
    goals.append(PlaceMachineAt(MachineType.CROSSING, _WAFER_CROSSING_TILE, 1))

    for tile, facing in _WAFER_PRE_CROSSING_BELTS:
        goals.append(PlaceMachineAt(MachineType.CONVEYOR_BELT, tile, facing))

    goals.append(
        PlaceMachineFromBackAt(
            MachineType.ARM, _WAFER_EXTRACT_ARM_TILE, int(Direction.UP)
        )
    )
    return goals


# ---------------------------------------------------------------------------
# Phase 3.MOTOR — assembler + frame extractor + wire route
# ---------------------------------------------------------------------------


def _phase_3_motor_goals() -> list[Goal]:
    """Place the MOTOR assembler module + WIRE route + extractor arms.

    FRAME comes off (21, 22) via a single arm at (22, 22) facing
    RIGHT (from-back) that pushes directly onto the MOTOR input_b
    feeder belt at (23, 22) — no FRAME-route belts needed.

    WIRE comes off the WIRE PALLET at (15, 14) via an arm at (15, 13)
    facing UP (from-back; the natural south stand tile is the PALLET).
    The arm pushes WIRE UP onto (15, 12); the route runs east on
    row 12 through two CROSSINGs at (16, 12) and (19, 12), continues
    to col 24, then south down col 24 to the MOTOR input_a feeder at
    (24, 21).

    Placement order (each step's stand tile is dirt or a previously
    placed walkable belt; CROSSINGs are not walkable):

    1. MOTOR module at ``_MOTOR_ASSEMBLER_TILE``.
    2. Sink-first row 12 + col 24 belts up to (20, 12).
    3. CROSSING at (19, 12) dir=1. Stand = (20, 12) belt.
    4. Mid belts at (18, 12) and (17, 12).
    5. CROSSING at (16, 12) dir=1. Stand = (17, 12) belt.
    6. Tail belt (15, 12).
    7. WIRE extractor arm at (15, 13) UP via from-back. Stand =
       (15, 12) belt (placed in step 6).
    8. FRAME extractor arm at (22, 22) RIGHT via from-back. Stand =
       (23, 22) = MOTOR input_b feeder belt.
    """
    goals: list[Goal] = []

    goals.extend(
        build_assembler_module_at(
            _MOTOR_ASSEMBLER_TILE,
            input_b=True,
            map_size=_MAP_SIZE,
        )
    )

    for tile, facing in _WIRE_TO_MOTOR_BELTS:
        goals.append(PlaceMachineAt(MachineType.CONVEYOR_BELT, tile, facing))

    # CROSSING dir=1: vertical N->S (iron) + horizontal W->E (wire).
    # Stand = (20, 12) belt placed above.
    goals.append(PlaceMachineAt(MachineType.CROSSING, _WIRE_IRON_CROSSING_TILE, 1))

    for tile, facing in _WIRE_TO_MOTOR_MID_BELTS:
        goals.append(PlaceMachineAt(MachineType.CONVEYOR_BELT, tile, facing))

    # CROSSING dir=1: vertical N->S (CIRCUIT-bound copper) +
    # horizontal W->E (wire). Stand = (17, 12) belt placed above.
    goals.append(PlaceMachineAt(MachineType.CROSSING, _WIRE_COPPER_CROSSING_TILE, 1))

    for tile, facing in _WIRE_TO_MOTOR_TAIL_BELTS:
        goals.append(PlaceMachineAt(MachineType.CONVEYOR_BELT, tile, facing))

    # WIRE extractor's natural stand tile (15, 14) is the WIRE output
    # PALLET (non-walkable), so it goes via from-back: stand on the
    # (15, 12) belt placed above, face DOWN, place inheriting DOWN,
    # then ROTATE to UP. After placement the arm pulls wire from
    # (15, 14) PALLET (south, behind) and pushes onto (15, 12) belt
    # (north, front).
    goals.append(
        PlaceMachineFromBackAt(
            MachineType.ARM, _WIRE_EXTRACT_ARM_TILE, int(Direction.UP)
        )
    )
    goals.append(
        PlaceMachineFromBackAt(
            MachineType.ARM, _FRAME_EXTRACT_ARM_TILE, int(Direction.RIGHT)
        )
    )
    return goals


# ---------------------------------------------------------------------------
# Phase 3.SENSOR — assembler + CIRCUIT extractor + WIRE splitter
# ---------------------------------------------------------------------------


def _phase_3_sensor_goals() -> list[Goal]:
    """Place the SENSOR module + WIRE splitter + CIRCUIT extractor.

    WIRE comes off a SPLITTER at (24, 12) horizontal RIGHT that
    replaces the wire belt on the WIRE -> MOTOR trunk. The splitter's
    DOWN output continues feeding MOTOR; its UP output feeds a new
    corridor along row 11 east, then south down col 28 into the
    SENSOR input_a feeder at (28, 17).

    CIRCUIT comes off the CIRCUIT output PALLET at (18, 18) via an
    arm at (18, 19) facing DOWN (from-back; the natural stand tile
    is the PALLET). The arm pushes CIRCUIT onto (18, 20); the route
    runs east on row 20 through two CROSSINGs at (19, 20) (iron) and
    (24, 20) (wire), continues to col 27, then bends UP into the
    SENSOR input_b feeder at (27, 18).

    Placement order (each step's stand tile is dirt or a previously
    placed walkable belt; CROSSINGs and SPLITTERs are not walkable):

    1. SENSOR module at ``_SENSOR_ASSEMBLER_TILE``.
    2. WIRE -> SENSOR belts sink-first (col 28 from row 16 up to
       row 11, then row 11 from col 27 west to col 24).
    3. WIRE SPLITTER at (24, 12). Stand = (23, 12) wire-trunk belt
       (placed by Phase 3.MOTOR).
    4. CIRCUIT -> SENSOR sink belts (col 27 UP, row 20 east-end
       sink-first to (25, 20)).
    5. CROSSING (24, 20) dir=1. Stand = (25, 20) belt.
    6. CIRCUIT -> SENSOR mid belts ((23, 20)..(20, 20) RIGHT).
    7. CROSSING (19, 20) dir=1. Stand = (20, 20) belt.
    8. CIRCUIT -> SENSOR tail belt ((18, 20) RIGHT).
    9. CIRCUIT extractor arm at (18, 19) DOWN via from-back. Stand
       = (18, 20) belt.
    """
    goals: list[Goal] = []

    goals.extend(
        build_assembler_module_at(
            _SENSOR_ASSEMBLER_TILE,
            input_b=True,
            map_size=_MAP_SIZE,
        )
    )

    for tile, facing in _WIRE_TO_SENSOR_BELTS:
        goals.append(PlaceMachineAt(MachineType.CONVEYOR_BELT, tile, facing))

    # WIRE SPLITTER — horizontal facing RIGHT outputs UP and DOWN.
    # DOWN feeds the existing MOTOR-bound col-24 trunk; UP feeds
    # the SENSOR-bound row-11 corridor placed above.
    goals.append(
        PlaceMachineAt(MachineType.SPLITTER, _WIRE_SPLITTER_TILE, int(Direction.RIGHT))
    )

    for tile, facing in _CIRCUIT_TO_SENSOR_BELTS:
        goals.append(PlaceMachineAt(MachineType.CONVEYOR_BELT, tile, facing))

    # CROSSING dir=1: vertical N->S (wire) + horizontal W->E (CIRCUIT).
    goals.append(PlaceMachineAt(MachineType.CROSSING, _CIRCUIT_WIRE_CROSSING_TILE, 1))

    for tile, facing in _CIRCUIT_TO_SENSOR_MID_BELTS:
        goals.append(PlaceMachineAt(MachineType.CONVEYOR_BELT, tile, facing))

    # CROSSING dir=1: vertical N->S (iron) + horizontal W->E (CIRCUIT).
    goals.append(PlaceMachineAt(MachineType.CROSSING, _CIRCUIT_IRON_CROSSING_TILE, 1))

    for tile, facing in _CIRCUIT_TO_SENSOR_TAIL_BELTS:
        goals.append(PlaceMachineAt(MachineType.CONVEYOR_BELT, tile, facing))

    # CIRCUIT extractor arm at (18, 19) facing DOWN — natural stand
    # tile (18, 18) is the CIRCUIT output PALLET, so go from-back:
    # stand on the (18, 20) belt placed in step 8, face UP, place
    # inheriting UP, then ROTATE to DOWN.
    goals.append(
        PlaceMachineFromBackAt(
            MachineType.ARM, _CIRCUIT_EXTRACT_ARM_TILE, int(Direction.DOWN)
        )
    )
    return goals


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_advanced_factory_goals(
    book: RecipeBook = BASE_RECIPE_BOOK,
    slack: dict[int, int] | None = None,
) -> list[Goal]:
    """Build the flat goal list for the advanced-factory rocket agent.

    Phase 0 hand-bootstraps every item later phases consume; Phase 1
    drops one automated smelter cell per ore (iron / copper / tin /
    silicon); Phase 2 places the WIRE assembler with both inter-cell
    splitters (copper at (13, 12), tin at (11, 16)); Phase 3.FRAME
    places the FRAME assembler with an iron extractor and routes
    that bypass the F+A drain zone; Phase 3.CIRCUIT places the
    CIRCUIT assembler with a wafer extractor, plus a CROSSING that
    lets the wafer route share row 17 with the tin -> FRAME trunk.
    Phase 0's mine / smelt / craft quantities are derived from
    later-phase consumption via a BOM walk, so a recipe rebalance
    flows through automatically.

    Args:
        book: :class:`~factoriax.recipes.RecipeBook` whose recipes
            drive the BOM and production schedule. Defaults to the
            shipped :data:`~factoriax.recipes.BASE_RECIPE_BOOK`;
            pass a tuned book (e.g.
            ``BASE_RECIPE_BOOK.with_balance(...)``) to make the
            agent's mine / smelt / craft quantities track the
            balance overlay.
        slack: Per-leaf-resource slack added on top of the BOM
            count. Defaults to :data:`_DEFAULT_SLACK`. Pass an
            empty dict to mine exactly the BOM amount (useful for
            tests that want to fail fast on a starvation bug).

    Returns:
        Flat list of :class:`~baselines.rocket.scripted.goals.Goal`
        instances ready for the planner.
    """
    if slack is None:
        slack = _DEFAULT_SLACK

    phase_0a_targets = _phase_1_targets()
    phase_0b_targets = sum_inventories(
        _phase_2_wire_targets(),
        _phase_3_frame_targets(),
        _phase_3_circuit_targets(),
        _phase_3_motor_targets(),
        _phase_3_sensor_targets(),
    )
    goals: list[Goal] = list(_phase_0a(phase_0a_targets, book, slack))

    for spec in _PHASE_1_CELLS:
        goals.extend(_phase_1_cell_goals(spec))
        # WaitUntil after each cell so its miners spin up before
        # VerifyLayout reads the map; new placements fire the
        # output-predicate within a few ticks.
        goals.append(WaitUntil(_miner_has_output_predicate(), max_ticks=60))
        expected = {**_PRE_PLACED_LAYOUT, **expected_layout_from_goals(goals)}
        goals.append(VerifyLayout(expected, label=f"phase 1.{spec.label}"))

    # Settle wait so each cell's plate-bus has at least one plate
    # before Phase 0b's first WithdrawFromBusAt fires. Each cell
    # smelts at 1 plate per ~2 ticks once ore + coal are flowing,
    # so 80 ticks comfortably covers spin-up and the first plate
    # landing on the bus.
    goals.append(Wait(80))
    goals.extend(_phase_0b(phase_0b_targets, book, slack))

    goals.extend(_phase_2_wire_goals())
    expected = {**_PRE_PLACED_LAYOUT, **expected_layout_from_goals(goals)}
    goals.append(VerifyLayout(expected, label="phase 2.wire"))

    goals.extend(_phase_3_frame_goals())
    expected = {**_PRE_PLACED_LAYOUT, **expected_layout_from_goals(goals)}
    goals.append(VerifyLayout(expected, label="phase 3.frame"))

    goals.extend(_phase_3_circuit_goals())
    expected = {**_PRE_PLACED_LAYOUT, **expected_layout_from_goals(goals)}
    goals.append(VerifyLayout(expected, label="phase 3.circuit"))

    goals.extend(_phase_3_motor_goals())
    expected = {**_PRE_PLACED_LAYOUT, **expected_layout_from_goals(goals)}
    goals.append(VerifyLayout(expected, label="phase 3.motor"))

    goals.extend(_phase_3_sensor_goals())
    expected = {**_PRE_PLACED_LAYOUT, **expected_layout_from_goals(goals)}
    goals.append(VerifyLayout(expected, label="phase 3.sensor"))

    # Settle wait — let the chain reach steady state so MOTOR and
    # SENSOR pallets fill before the player tries to withdraw. ~70
    # belts plus four assembler cycles take a few hundred ticks for
    # the first item of each kind to land.
    goals.append(Wait(800))

    # End-of-run achievement pickups. MOTOR and SENSOR are produced
    # only on-belt (no Phase 0 hand-craft path), so the player has
    # never held one. WithdrawFromBusAt drains a single unit from
    # each live output pallet, which is enough to unlock the
    # craft_motor / craft_sensor achievements (the benchmark checks
    # player inventory, not pallet contents).
    goals.append(
        WithdrawFromBusAt(
            _MOTOR_OUTPUT_TILE,
            ItemType.MOTOR,
            count=1,
            max_idle_attempts=400,
        )
    )
    goals.append(
        WithdrawFromBusAt(
            _SENSOR_OUTPUT_TILE,
            ItemType.SENSOR,
            count=1,
            max_idle_attempts=400,
        )
    )

    # Trailing wait. Lets late achievements (e.g. accumulating output
    # pallet counts) finish firing before episode end.
    goals.append(Wait(1500))
    return goals


def make_advanced_factory_rocket_agent(
    env_params: EnvParams,
    book: RecipeBook = BASE_RECIPE_BOOK,
    slack: dict[int, int] | None = None,
) -> ScriptedAgent:
    """Construct the advanced-factory scripted agent.

    Args:
        env_params: Environment parameters (passed to the planner).
        book: :class:`~factoriax.recipes.RecipeBook` driving the
            bootstrap quantities. Pass a tuned book to explore
            balance changes.
        slack: Per-leaf-resource slack on top of the BOM. See
            :func:`build_advanced_factory_goals`.
    """
    return ScriptedAgent(
        env_params,
        Planner(build_advanced_factory_goals(book=book, slack=slack)),
    )
