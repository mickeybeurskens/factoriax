"""Advanced-factory rocket agent — v2 incremental rebuild.

Currently at **M4** of the v2 plan: Phase 0 bootstrap + Phase 1.iron
(first automated smelter). The agent

1. Hand-mines a starter inventory sized for *every* placement Phase
   1.iron will make, smelts each plate, then crafts every machine
   the cell + coal trunk + supporting infrastructure consumes.
2. Drops one MINER on the iron patch, threads ore east into the
   smelter cell's ore_pallet, then places the smelter (furnace,
   arm, plate-bus PALLET, coal-buffer PALLET) and a 1-miner + 7-
   belt coal trunk along row 11 feeding the coal-buffer.
3. Verifies the placements landed, waits for plates to accumulate.

Phase 0 quantities are recipe-driven: targets feed
:func:`bill_of_materials` and :func:`production_schedule`, so a
recipe rebalance auto-resizes mining / smelting / crafting without
hand edits.

M5+ extend Phase 1 to copper / tin / silicon by tacking more
identical-shaped cells onto the same Phase 0 + Phase 1 prefix.

Map layout the agent assumes (v2; see
:func:`factoriax.benchmarks.rocket.build_rocket_level`)::

      0 1 2 3 4 5 6 7 8 9 ...
    0 #
    .                              spawn at (16, 16),
    9 # · · I I · · . O . . .      pre-placed F at (15,16),
   10 # · · I I · · F a P . .      pre-placed A at (17,16)
   11 M . . . . . . . P . . .
    .
   12 # · · U U · · . . . . .       (M=miner, .=belt, F=furnace,
   13 # · · U U                      a=arm, P=pallet, O=ore_pallet,
   14                                #=coal column, I=iron patch)

The coal column is a single tile wide (x=0). The iron coal miner
sits *on* the column at (0, 11) facing RIGHT. Its natural stand
tile (-1, 11) is off-map; ``PlaceMachineFromBackAt`` works around
that by placing from the front (player at (1, 11) facing LEFT)
and rotating the miner with ``ROTATE_RIGHT`` afterward.
"""

from __future__ import annotations

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
# Geometry — Phase 1.iron tile coordinates
# ---------------------------------------------------------------------------

_MAP_SIZE: tuple[int, int] = (32, 32)

# Iron smelter cell anchor (the furnace tile). build_smelter_cell_at
# derives ore_pallet (north), coal_pallet (south), arm (east), and
# plate-bus (east+1) from this.
_IRON_FURNACE_TILE: tuple[int, int] = (7, 10)
_IRON_ORE_PALLET_TILE: tuple[int, int] = (7, 9)  # north of furnace
_IRON_PLATE_BUS_TILE: tuple[int, int] = (9, 10)  # east of arm

# Iron miner sits on the bottom-right tile of the 2x2 iron patch
# at (3..4, 9..10) and pushes east. Two belts carry the ore to the
# ore_pallet north of the furnace.
_IRON_MINER_TILE: tuple[int, int] = (4, 9)
_IRON_ORE_BELT_TILES: tuple[tuple[int, int], ...] = ((5, 9), (6, 9))

# Iron coal trunk: one coal miner on the coal column, seven belts
# running east along row 11 into the smelter cell's coal_buffer
# pallet at (7, 11). Row 11 is the dirt strip between the iron
# patch (rows 9-10) and the copper patch (rows 12-13).
#
# Miner sits at (0, 11) — on the coal column itself — facing RIGHT.
# Its natural stand tile (-1, 11) is off-map, so the agent uses
# ``PlaceMachineFromBackAt``: stand at (1, 11) facing LEFT, place
# (miner inherits LEFT), then ROTATE_RIGHT to flip facing.
_IRON_COAL_MINER_TILE: tuple[int, int] = (0, 11)
_IRON_COAL_BELT_TILES: tuple[tuple[int, int], ...] = (
    (1, 11),
    (2, 11),
    (3, 11),
    (4, 11),
    (5, 11),
    (6, 11),
)

# ---------------------------------------------------------------------------
# Phase-1 placement inventory — items needed for Phase 1.iron alone
# ---------------------------------------------------------------------------

# Iron-cell + coal-trunk machinery (does not include the ore_pallet
# nor the smelter-cell pieces; those come from
# smelter_cell_inventory() + 1 explicit PALLET below).
_PHASE_1_IRON_INFRASTRUCTURE: dict[int, int] = {
    int(ItemType.MINER): 2,  # 1 ore + 1 coal
    int(ItemType.CONVEYOR_BELT): len(_IRON_ORE_BELT_TILES) + len(_IRON_COAL_BELT_TILES),
    int(ItemType.PALLET): 1,  # ore_pallet at (7, 9)
}


def _phase_1_iron_targets() -> dict[int, int]:
    """Return ``{ItemType: count}`` of items Phase 1.iron places.

    Sums the cell footprint (2 PALLET + 1 ARM + 1 FURNACE) with the
    feeder infrastructure (2 MINER + 8 BELT + 1 ore PALLET).
    """
    return sum_inventories(
        smelter_cell_inventory(),
        _PHASE_1_IRON_INFRASTRUCTURE,
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
    int(ItemType.LIMESTONE): 1,
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
# Phase 1.iron — one smelter cell + ore feed + coal trunk
# ---------------------------------------------------------------------------


def _phase_1_iron() -> list[Goal]:
    """Place the iron smelter cell, ore feed belts, and coal trunk.

    Placement order. Belts are always placed *before* the
    machines that push onto them, otherwise the player would have
    to stand on a non-walkable miner / pallet tile to place the
    next belt east of it.

    1. Two ore belts on row 9 ((5, 9), (6, 9)) ferrying ore from
       the iron patch east to the ore_pallet. Belt (5, 9)'s stand
       tile is (4, 9) — iron ore tile, walkable.
    2. Ore_pallet at (7, 9) facing DOWN. Stand tile (7, 8) is
       dirt.
    3. Iron miner on patch (4, 9) facing RIGHT pushing east onto
       the belt at (5, 9). Stand tile (3, 9) is iron ore.
    4. Smelter cell at (7, 10) facing RIGHT. ``build_smelter_cell_at``
       internally orders coal_buffer -> plate_bus -> arm -> furnace
       so every stand tile is walkable at place time.
    5. Six coal belts on row 11 ((1, 11)..(6, 11)) feeding the
       coal_buffer at (7, 11). Placed before the coal miner so
       (1, 11) has a belt to receive the miner's eastward push.
    6. Coal miner at (0, 11) RIGHT via ``PlaceMachineFromBackAt``.
       Natural stand tile (-1, 11) is off-map; the from-back
       variant stands on the belt at (1, 11) facing LEFT, places
       (miner inherits LEFT), then ROTATEs to RIGHT.

    A WaitUntil + VerifyLayout gate confirms automation_mining
    fires and every Phase 1.iron placement is on the map before
    the trailing settle-Wait counts plates.
    """
    goals: list[Goal] = []

    # 1. Iron ore feed belts (placed before the miner so the miner's
    # stand tile isn't blocked by a previously placed neighbour).
    for tile in _IRON_ORE_BELT_TILES:
        goals.append(
            PlaceMachineAt(MachineType.CONVEYOR_BELT, tile, int(Direction.RIGHT))
        )

    # 2. Ore_pallet (faces DOWN; receives push from the last belt at
    # (6, 9) RIGHT and pushes south into the furnace at (7, 10)).
    goals.append(
        PlaceMachineAt(MachineType.PALLET, _IRON_ORE_PALLET_TILE, int(Direction.DOWN))
    )

    # 3. Iron miner pushing east onto the belt at (5, 9).
    goals.append(
        PlaceMachineAt(MachineType.MINER, _IRON_MINER_TILE, int(Direction.RIGHT))
    )

    # 4. Smelter cell — places coal_buffer (7, 11) UP, plate_bus
    # (9, 10) DOWN, arm (8, 10) RIGHT, furnace (7, 10) RIGHT.
    goals.extend(
        build_smelter_cell_at(
            _IRON_FURNACE_TILE,
            facing=int(Direction.RIGHT),
            map_size=_MAP_SIZE,
        )
    )

    # 5. Coal trunk belts on row 11.
    for tile in _IRON_COAL_BELT_TILES:
        goals.append(
            PlaceMachineAt(MachineType.CONVEYOR_BELT, tile, int(Direction.RIGHT))
        )

    # 6. Coal miner at (0, 11) RIGHT via place-from-back.
    goals.append(
        PlaceMachineFromBackAt(
            MachineType.MINER, _IRON_COAL_MINER_TILE, int(Direction.RIGHT)
        )
    )

    # 7. Stage gates.
    goals.append(WaitUntil(_miner_has_output_predicate(), max_ticks=60))
    expected = {**_PRE_PLACED_LAYOUT, **expected_layout_from_goals(goals)}
    goals.append(VerifyLayout(expected, label="phase 1.iron"))
    return goals


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_advanced_factory_goals(
    book: RecipeBook = BASE_RECIPE_BOOK,
    slack: dict[int, int] | None = None,
) -> list[Goal]:
    """Build the flat goal list for the advanced-factory rocket agent.

    Currently M4: Phase 0 bootstrap + Phase 1.iron (one automated
    smelter cell). Phase 0 is sized for exactly the items Phase 1
    places, so a recipe rebalance flows through to mining /
    smelting / crafting counts automatically.

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
        _phase_1_iron_targets(),
        _ACHIEVEMENT_KEEPSAKES,
    )
    return [
        *_phase_0(crafted_targets, book, slack),
        *_phase_1_iron(),
        # Trailing wait. The cell needs ~50 ticks to make a plate
        # (smelt + arm-pull-out + push-into-bus); 1500 ticks ensures
        # the bus pallet has multiple plates by end of episode.
        Wait(1500),
    ]


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
