"""Advanced-factory rocket agent — v2 incremental rebuild.

Currently at **M5** of the v2 plan: Phase 0 bootstrap + Phase 1
(iron *and* copper automated smelters). Iron and copper share the
same cell shape; M6+ adds tin / silicon / wafer cells and the
inter-cell crafting chain.

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
    assembler_module_inventory,
    build_assembler_module_at,
    build_smelter_cell_at,
    smelter_cell_inventory,
)
from .layout import expected_layout_from_goals
from .planner import Planner
from .recipe_planning import (
    bill_of_materials,
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
                int(ItemType.CONVEYOR_BELT): (
                    len(self.ore_belt_tiles) + len(self.coal_belt_tiles)
                ),
                int(ItemType.PALLET): 1,  # ore_pallet
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

# Copper extractor arm + east route into input_a.
_COPPER_EXTRACT_ARM_TILE: tuple[int, int] = (10, 13)
_COPPER_TO_WIRE_BELT_TILES: tuple[tuple[int, int], ...] = ((11, 13), (12, 13))

# Tin extractor arm + east route then north turn into input_b.
_TIN_EXTRACT_ARM_TILE: tuple[int, int] = (10, 16)
_TIN_TO_WIRE_BELT_TILES: tuple[tuple[int, int], ...] = (
    (11, 16),  # RIGHT
    (12, 16),  # UP — turns north
    (12, 15),  # UP — pushes north into input_b at (12, 14)
)


def _phase_2_wire_targets() -> dict[int, int]:
    """Return ``{ItemType: count}`` for the WIRE cell + plate routes.

    Counts: 1 ASSEMBLER + 3 PALLET + 1 ARM (assembler module) +
    2 ARM (copper + tin extractors) + 5 CONVEYOR_BELT (route belts).
    """
    return sum_inventories(
        assembler_module_inventory(input_b=True),
        {
            int(ItemType.ARM): 2,
            int(ItemType.CONVEYOR_BELT): (
                len(_COPPER_TO_WIRE_BELT_TILES) + len(_TIN_TO_WIRE_BELT_TILES)
            ),
        },
    )


# Achievement-preservation targets: items the agent smelts / crafts
# even though no Phase 1 placement consumes them, so the rocket
# benchmark unlocks fire (e.g. ``smelt_wafer`` requires holding a
# wafer at some point — Phase 1.iron doesn't need wafer otherwise).
# M5 will retire this set as wafer / etc. become genuine
# infrastructure inputs.
_ACHIEVEMENT_KEEPSAKES: dict[int, int] = {
    int(ItemType.WAFER): 2,
}


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
# Phase 0 — bootstrap (hand-mine + smelt + craft at pre-placed F+A)
# ---------------------------------------------------------------------------


def _phase_0(
    crafted_targets: dict[int, int],
    book: RecipeBook,
    slack: dict[int, int],
) -> list[Goal]:
    """Hand-mine + smelt + craft enough to cover ``crafted_targets``.

    ``crafted_targets`` is the union of every recipe-output item the
    agent will *consume* in subsequent phases (placements +
    crafts that feed into other crafts). The schedule helper expands
    that into per-recipe production counts; the BOM walks back to
    leaves so we know how much raw to mine.
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
        # Other machine types are not produced via Phase 0 (e.g. recipes
        # gated to a specific cell). None of the rocket recipes use them
        # at this stage.

    return [*mine_goals, *smelt_goals, *craft_goals]


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
       ore_pallet. Belt 0's stand tile sits on the ore patch
       itself — walkable.
    2. Ore_pallet facing DOWN via ``PlaceMachineFromBackAt``.
       The natural (north) stand tile would either be dirt
       (iron — fine on its own) or the previous cell's
       coal_buffer (copper, tin, silicon — non-walkable). Using
       from-back uniformly stands the player on the dirt south of
       the pallet, places facing UP, then ROTATEs to DOWN.
    3. Ore miner facing RIGHT pushing east onto the first belt.
       Stand tile is the patch's third column — walkable ore.
    4. Smelter cell at the furnace tile facing RIGHT.
       ``build_smelter_cell_at`` internally orders coal_buffer ->
       plate_bus -> arm -> furnace so every stand tile is walkable
       at place time.
    5. Six coal belts feeding the cell's coal_buffer. Placed
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
    goals.append(
        PlaceMachineFromBackAt(
            MachineType.PALLET, spec.ore_pallet_tile, int(Direction.DOWN)
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
    """Place the WIRE assembler + the two plate routes feeding it.

    Placement order (each step's stand tile is dirt or a previously
    placed walkable belt):

    1. Assembler module at ``_WIRE_ASSEMBLER_TILE``. The module
       helper places output -> input_b -> arm -> assembler ->
       input_a, in that order (output last among the input pallets
       so each stand tile stays walkable).
    2. Copper route belts west-to-east into the input_a tile.
       Order ``[(12, 13), (11, 13)]`` because (11, 13)'s stand tile
       (10, 13) must be dirt at place time — the extractor arm
       lands there only afterward.
    3. Copper extractor arm at (10, 13) RIGHT via from-back. Its
       natural stand tile (9, 13) is the copper plate-bus PALLET
       (non-walkable); the from-back variant stands on the belt
       at (11, 13), places facing LEFT, then ROTATEs to RIGHT.
    4. Tin route belts. Order ``[(12, 15), (12, 16), (11, 16)]``
       so each new belt's stand tile stays dirt.
    5. Tin extractor arm at (10, 16) RIGHT via from-back, same
       trick as copper.
    """
    goals: list[Goal] = []

    goals.extend(
        build_assembler_module_at(
            _WIRE_ASSEMBLER_TILE,
            input_b=True,
            map_size=_MAP_SIZE,
        )
    )

    for tile in reversed(_COPPER_TO_WIRE_BELT_TILES):
        goals.append(
            PlaceMachineAt(MachineType.CONVEYOR_BELT, tile, int(Direction.RIGHT))
        )
    goals.append(
        PlaceMachineFromBackAt(
            MachineType.ARM, _COPPER_EXTRACT_ARM_TILE, int(Direction.RIGHT)
        )
    )

    # Tin route belts. Tiles in order [(11, 16), (12, 16), (12, 15)]
    # with facings [RIGHT, UP, UP]. Each placement's stand tile is
    # dirt at the moment of placement: (10, 16) for (11, 16) RIGHT,
    # (12, 17) for (12, 16) UP, (12, 16) [now a belt] for (12, 15) UP.
    tin_belt_facings: tuple[int, ...] = (
        int(Direction.RIGHT),
        int(Direction.UP),
        int(Direction.UP),
    )
    for tile, facing in zip(_TIN_TO_WIRE_BELT_TILES, tin_belt_facings):
        goals.append(PlaceMachineAt(MachineType.CONVEYOR_BELT, tile, facing))
    goals.append(
        PlaceMachineFromBackAt(
            MachineType.ARM, _TIN_EXTRACT_ARM_TILE, int(Direction.RIGHT)
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

    Currently M5: Phase 0 bootstrap + Phase 1 (iron + copper
    automated smelter cells). Phase 0 is sized for exactly the
    items Phase 1 places, so a recipe rebalance flows through to
    mining / smelting / crafting counts automatically.

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

    crafted_targets = sum_inventories(
        _phase_1_targets(),
        _phase_2_wire_targets(),
        _ACHIEVEMENT_KEEPSAKES,
    )
    goals: list[Goal] = list(_phase_0(crafted_targets, book, slack))

    for spec in _PHASE_1_CELLS:
        goals.extend(_phase_1_cell_goals(spec))
        # WaitUntil after each cell so its miners spin up before
        # VerifyLayout reads the map; new placements fire the
        # output-predicate within a few ticks.
        goals.append(WaitUntil(_miner_has_output_predicate(), max_ticks=60))
        expected = {**_PRE_PLACED_LAYOUT, **expected_layout_from_goals(goals)}
        goals.append(VerifyLayout(expected, label=f"phase 1.{spec.label}"))

    goals.extend(_phase_2_wire_goals())
    expected = {**_PRE_PLACED_LAYOUT, **expected_layout_from_goals(goals)}
    goals.append(VerifyLayout(expected, label="phase 2.wire"))

    # Trailing wait. Each cell needs ~50 ticks to make a plate
    # (smelt + arm-pull-out + push-into-bus); 1500 ticks lets every
    # bus pallet collect multiple plates by end of episode and the
    # WIRE assembler enough cycles to fill its output pallet.
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
