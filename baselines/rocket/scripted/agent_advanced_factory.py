"""Advanced-factory rocket agent — iterative bootstrap (4 cells).

Two-phase iterative bootstrap. Phase 1 sizes mining + pre-smelting
for *just* the iron + copper cells, builds those, then Phase 2
runs while the iron + copper cells are already producing plates:

- **Phase 1** drops a miner + pallet on every non-coal patch (iron,
  copper, tin, silicon — even though tin/silicon won't be used until
  Phase 2, dropping their miners now means the patches accumulate
  ore for free). It then places the iron + copper coal miners +
  trunks and finally builds the iron + copper splitter cells. Phase
  1's mine + smelt totals are sized for *only* what's needed to
  build those two cells + their trunks + the four ore-nodes.

- **Phase 2** builds the tin + silicon cells. By the time Phase 2
  starts, the iron + copper cells have been producing plates for
  ~60+ ticks, so Phase 2 *withdraws* IRON_PLATE / COPPER_PLATE from
  the running cells' manual stashes via ``WithdrawFromBusAt``
  instead of mining + smelting more ore. Only TIN_PLATE +
  REFRACTORY get pre-smelted at the idle pre-placed furnace at
  (15, 16) (using ``ProduceInFurnaceAt`` so the deposit isn't
  routed to a flowing iron / copper cell furnace by the closest-
  machine heuristic).

Each cell has the T1 output design (``output_split=True``):

  * a SPLITTER replaces the old plate-bus PALLET at the same tile;
  * a *manual_stash* PALLET sits north of the splitter — the agent
    withdraws plates from here for hand-crafting;
  * an *automation belt* (south of the splitter) carries the
    splitter's other half-stream into a *sink* PALLET. The sink is
    a drain so the splitter's both-or-nothing fire condition is
    met whenever the manual stash has space.

The copper cell uses ``automation_belt=False`` because the
splitter's south output drops directly onto a sink PALLET at
(21, 12). The copper coal trunk is rerouted through row 13 (one
row south of the original row-12 segment) so it never crosses
(21, 12) — no CROSSING is needed for the four-cell layout.

Recipe-driven bootstrap. The mine / smelt / craft *quantities* are
no longer hand-typed; they're computed from the active
:class:`~factoriax.recipes.RecipeBook` via
:mod:`baselines.rocket.scripted.recipe_planning`. The geometry —
tile coordinates, cell facings, belt routes — is still hardcoded
(it's a spatial decision, not a balance one). To explore balance
changes:

>>> from factoriax.recipes import BASE_RECIPE_BOOK, RecipeBalance, RecipeOverride
>>> book = BASE_RECIPE_BOOK.with_balance(RecipeBalance(overrides=(
...     (int(ItemType.IRON_PLATE), RecipeOverride(input_counts=(2, 1))),
... )))
>>> agent = make_advanced_factory_rocket_agent(env_params, book=book)

The agent's MineOre, ProduceInFurnace, and ProduceInAssembler goals
will all reflect the doubled iron-ore demand.

Map layout (32×32, spawn (16, 16); pre-placed FURNACE (15, 16),
ASSEMBLER (17, 16))::

                      silicon (14-16, 3-5)
                              |
                   silicon cell @ (15, 7) RIGHT
                              ^
        [silicon coal trunk: col 7 north then row 8 east]
                              ^
       iron (7-9, 7-9)              copper (22-24, 7-9)
        |                           |
       iron cell @ (8, 11) RIGHT   copper cell @ (23, 11) LEFT
        |                           |
        [iron coal: col 10 +        [copper coal: col 11 + row 13
         westward bend at row 14]    east to (23, 13) then UP to
                                     (23, 12) coal_buffer; sink
                                     PALLET at (21, 12) catches
                                     the splitter S output]
        |                           |
                spawn (16, 16)
                              |
        [tin coal: col 7 south + row 27 east]
                              |
       coal (7-9, 22-24)            tin (22-24, 22-24)
        4 coal miners (south        tin cell @ (23, 26) RIGHT
        edge + (7, 22) UP for
        silicon's long northward
        trunk)

Pre-smelt strategy. ``ProduceInMachine`` picks the *nearest*
furnace / assembler each cycle, so once a cell furnace is placed,
any further ``ProduceInFurnace`` would land ore in the wrong
furnace. Phase 1 pre-smelts before any cell exists, so its
``ProduceInFurnace`` calls all route to the pre-placed (15, 16)
furnace. Phase 2 (running while the iron + copper cells exist)
uses :class:`ProduceInFurnaceAt` pinned to (15, 16) for its
TIN_PLATE / REFRACTORY smelts so deposits never get redirected to
the cell furnaces.
"""

from __future__ import annotations

from factoriax.constants import Direction, ItemType, MachineType
from factoriax.recipes import BASE_RECIPE_BOOK, RecipeBook
from factoriax.state import EnvParams

from .agent import ScriptedAgent, _miner_has_output_predicate
from .goals import (
    BeltPath,
    Goal,
    MineOre,
    PlaceMachineAt,
    ProduceInAssembler,
    ProduceInFurnace,
    ProduceInFurnaceAt,
    Wait,
    WaitUntil,
    WithdrawFromBusAt,
    belt_network_inventory,
    build_smelter_cell_at,
    ore_node_inventory,
    place_belt_network,
    place_ore_node,
    smelter_cell_inventory,
)
from .planner import Planner
from .recipe_planning import (
    bill_of_materials,
    book_without_recipes_for,
    production_schedule,
    scale_inventory,
    sum_inventories,
)

# ---------------------------------------------------------------------------
# Geometry — tile coordinates, facings, belt routes
# ---------------------------------------------------------------------------

_MAP_SIZE: tuple[int, int] = (32, 32)

# Manual-stash bus tiles (one per cell).
_IRON_MANUAL_STASH = (10, 10)
_COPPER_MANUAL_STASH = (21, 10)
_TIN_MANUAL_STASH = (25, 25)
_SILICON_MANUAL_STASH = (17, 6)

# Sink PALLETs at the automation lane's downstream tile. Iron / tin /
# silicon flow splitter -> automation_belt -> sink. Copper has
# ``automation_belt=False`` so its splitter's south output pushes
# *directly* into a sink at (21, 12); the copper coal trunk is
# rerouted through row 13 to avoid that tile.
_IRON_AUTOMATION_SINK = (10, 13)
_COPPER_AUTOMATION_SINK = (21, 12)
_TIN_AUTOMATION_SINK = (25, 28)
_SILICON_AUTOMATION_SINK = (17, 9)

# Coal trunk waypoints. Belt counts feed straight into
# ``belt_network_inventory`` so a waypoint edit propagates to the
# bootstrap craft list automatically.
_IRON_COAL_WAYPOINTS = [(10, 24), (10, 14), (8, 14), (8, 12)]
_COPPER_COAL_WAYPOINTS = [(8, 25), (11, 25), (11, 13), (23, 13), (23, 12)]
_TIN_COAL_WAYPOINTS = [(7, 25), (7, 27), (22, 27), (23, 27)]
_SILICON_COAL_WAYPOINTS = [(7, 21), (7, 8), (15, 8)]


def _belt_paths() -> list[BeltPath]:
    """The four coal trunks as labelled BeltPath records."""
    return [
        BeltPath(_IRON_COAL_WAYPOINTS, label="iron coal"),
        BeltPath(_COPPER_COAL_WAYPOINTS, label="copper coal"),
        BeltPath(_TIN_COAL_WAYPOINTS, label="tin coal"),
        BeltPath(_SILICON_COAL_WAYPOINTS, label="silicon coal"),
    ]


# ---------------------------------------------------------------------------
# Per-phase craft targets — derived from the inventory helpers
# ---------------------------------------------------------------------------

# Phase 1 places one ore-node per non-coal patch (4 patches).
_PHASE_1_ORE_NODE_TARGETS: dict[int, int] = scale_inventory(
    ore_node_inventory(with_pallet=True), 4
)


def _phase_1_belt_paths() -> list[BeltPath]:
    return [
        BeltPath(_IRON_COAL_WAYPOINTS, label="iron coal"),
        BeltPath(_COPPER_COAL_WAYPOINTS, label="copper coal"),
    ]


def _phase_2_belt_paths() -> list[BeltPath]:
    return [
        BeltPath(_TIN_COAL_WAYPOINTS, label="tin coal"),
        BeltPath(_SILICON_COAL_WAYPOINTS, label="silicon coal"),
    ]


def _phase_1_belt_network_targets() -> dict[int, int]:
    """Items the Phase 1 belt-network crafts: 2 coal miners, 2 sink
    pallets, and the BELT count from iron + copper trunks."""
    return sum_inventories(
        belt_network_inventory(_phase_1_belt_paths()),
        scale_inventory(ore_node_inventory(with_pallet=False), 2),
        {int(ItemType.PALLET): 2},
    )


def _phase_2_belt_network_targets() -> dict[int, int]:
    """Items the Phase 2 belt-network crafts: 2 coal miners, 2 sink
    pallets, and the BELT count from tin + silicon trunks."""
    return sum_inventories(
        belt_network_inventory(_phase_2_belt_paths()),
        scale_inventory(ore_node_inventory(with_pallet=False), 2),
        {int(ItemType.PALLET): 2},
    )


def _cell_targets_with_belt() -> dict[int, int]:
    """Iron / tin / silicon cells: full output_split with automation belt."""
    return smelter_cell_inventory(output_split=True, automation_belt=True)


def _cell_targets_no_belt() -> dict[int, int]:
    """Copper cell: output_split, no automation belt (splitter south
    output drops directly into the sink pallet at (21, 12))."""
    return smelter_cell_inventory(output_split=True, automation_belt=False)


def _phase_1_targets() -> dict[int, int]:
    """Items Phase 1 builds: 4 ore-nodes, iron + copper cells + their
    coal trunks. Phase 1's mining + pre-smelting is sized for these.
    """
    return sum_inventories(
        _PHASE_1_ORE_NODE_TARGETS,
        _phase_1_belt_network_targets(),
        _cell_targets_with_belt(),  # iron cell
        _cell_targets_no_belt(),  # copper cell
    )


def _phase_2_targets() -> dict[int, int]:
    """Items Phase 2 builds: tin + silicon cells + their coal trunks.

    Phase 2 sources IRON_PLATE / COPPER_PLATE from the running iron +
    copper manual stashes (via WithdrawFromBusAt), not by smelting,
    so the BOM is computed against a book where the IRON_PLATE and
    COPPER_PLATE recipes are dropped. Only TIN_PLATE and REFRACTORY
    get pre-smelted (at the idle pre-placed furnace at (15, 16)).
    """
    return sum_inventories(
        _phase_2_belt_network_targets(),
        scale_inventory(_cell_targets_with_belt(), 2),  # tin + silicon cells
    )


_RUNNING_CELLS_SUPPLY: set[int] = {
    int(ItemType.IRON_PLATE),
    int(ItemType.COPPER_PLATE),
}


# ---------------------------------------------------------------------------
# Default leaf-resource slack
# ---------------------------------------------------------------------------

# Slack absorbs the ProduceInFurnace cycle's deposit-then-wait
# ordering quirks where the engine's auto-pull may consume a coal /
# ore unit between two of the player's deposits. Tuned by hand from
# end-to-end runs; expose as a kwarg on
# :func:`build_advanced_factory_goals` so balance experiments can
# raise it for tuned books that need more headroom.
_DEFAULT_SLACK: dict[int, int] = {
    int(ItemType.IRON_ORE): 2,
    int(ItemType.COPPER_ORE): 2,
    int(ItemType.TIN_ORE): 2,
    int(ItemType.LIMESTONE): 1,
    int(ItemType.COAL): 5,
}


# ---------------------------------------------------------------------------
# Helper goal-builders — schedule-driven mine / smelt / craft
# ---------------------------------------------------------------------------


def _mine_goals_for(
    targets: dict[int, int],
    book: RecipeBook,
    slack: dict[int, int],
) -> list[Goal]:
    """One :class:`MineOre` per leaf in the BOM, sorted for stability."""
    bom = bill_of_materials(targets, book)
    for item, qty in slack.items():
        bom[int(item)] = bom.get(int(item), 0) + int(qty)
    return [MineOre(item, qty) for item, qty in sorted(bom.items())]


def _smelt_goals_for(
    targets: dict[int, int],
    book: RecipeBook,
) -> list[Goal]:
    """:class:`ProduceInFurnace` per FURNACE entry in the schedule.

    Topologically ordered so e.g. REFRACTORY's LIMESTONE smelt runs
    after any plate dependency it has (here it has none — REFRACTORY
    is leaf-fed by LIMESTONE + COAL).
    """
    schedule = production_schedule(targets, book)
    return [
        ProduceInFurnace(item, qty, book=book)
        for item, qty, machine in schedule
        if machine == int(MachineType.FURNACE)
    ]


def _craft_goals_for(
    targets: dict[int, int],
    book: RecipeBook,
) -> list[Goal]:
    """:class:`ProduceInAssembler` per ASSEMBLER entry in the schedule.

    Filters out FURNACE entries because every plate is pre-smelted
    in Phase A; the ``schedule`` for a per-phase target dict still
    enumerates plate intermediates (since plates *are* recipe inputs
    of the ASSEMBLER recipes), but the agent doesn't re-smelt them
    per phase.
    """
    schedule = production_schedule(targets, book)
    return [
        ProduceInAssembler(item, qty, book=book)
        for item, qty, machine in schedule
        if machine == int(MachineType.ASSEMBLER)
    ]


# ---------------------------------------------------------------------------
# Phase 1 — pre-cells: bootstrap iron + copper cells + their coal trunks
# ---------------------------------------------------------------------------

# Pre-placed furnace tile (rocket benchmark scenario).
_PRE_PLACED_FURNACE = (15, 16)


def _phase_1(book: RecipeBook, slack: dict[int, int]) -> list[Goal]:
    """Bootstrap mining + smelting sized only for iron + copper cells.

    Mines every leaf resource Phase 1 will consume, pre-smelts the
    plates needed for iron + copper cells + their two trunks + the
    four ore-node placements, then crafts *every Phase 1 item in a
    single batch* so the craft schedule shares intermediate cycles
    (BELT, SPLITTER) across cells. Splitting the crafts into per-cell
    calls — as a previous version did — caused integer-ceiling
    over-spend (e.g. 2 SPLITTER cycles instead of 1, 6 BELT cycles
    instead of 5), which by the copper cell's craft step left
    inventory ~1 plate short of each type. The copper cell's ARM,
    FURNACE, and SPLITTER cycles then FAILed, the cell built without
    its smelter, and the entire downstream Phase 2 cascade starved
    on missing COPPER_PLATE.

    Tin + silicon mining and smelting is deferred to Phase 2 so the
    iron + copper cells can produce plates passively while Phase 2
    runs.
    """
    targets = _phase_1_targets()
    return [
        # Phase 1 mining — every leaf the BOM needs.
        *_mine_goals_for(targets, book, slack),
        # Phase 1 pre-smelt — plates the iron + copper cell crafts
        # will consume. Routes to (15, 16) since no cells exist yet.
        *_smelt_goals_for(targets, book),
        # Phase 1 craft — one batch, summed target. Same schedule the
        # pre-smelt sized for, so plates balance exactly.
        *_craft_goals_for(targets, book),
        # Place 4 patch miners + 4 buffer pallets.
        *place_ore_node((8, 9), map_size=_MAP_SIZE),  # iron
        *place_ore_node((23, 9), map_size=_MAP_SIZE),  # copper
        *place_ore_node((23, 24), map_size=_MAP_SIZE),  # tin
        *place_ore_node((15, 5), map_size=_MAP_SIZE),  # silicon
        WaitUntil(_miner_has_output_predicate(), max_ticks=30),
        # Iron + copper coal trunks.
        *place_belt_network(
            _phase_1_belt_paths(),
            map_size=_MAP_SIZE,
            start_near=(16, 16),
        ),
        # Iron + copper coal miners.
        PlaceMachineAt(MachineType.MINER, (9, 24), int(Direction.RIGHT)),
        PlaceMachineAt(MachineType.MINER, (8, 24), int(Direction.DOWN)),
        # Iron + copper sink pallets.
        PlaceMachineAt(MachineType.PALLET, _IRON_AUTOMATION_SINK, int(Direction.DOWN)),
        PlaceMachineAt(
            MachineType.PALLET, _COPPER_AUTOMATION_SINK, int(Direction.DOWN)
        ),
        # Build iron + copper cells.
        *build_smelter_cell_at(
            (8, 11),
            facing=int(Direction.RIGHT),
            output_split=True,
            map_size=_MAP_SIZE,
        ),
        Wait(60),
        *build_smelter_cell_at(
            (23, 11),
            facing=int(Direction.LEFT),
            output_split=True,
            automation_belt=False,
            map_size=_MAP_SIZE,
        ),
    ]


# ---------------------------------------------------------------------------
# Phase 2 — post-cells: tin + silicon built using running cells' output
# ---------------------------------------------------------------------------


def _phase_2(book: RecipeBook, slack: dict[int, int]) -> list[Goal]:
    """Build tin + silicon cells while iron + copper cells run.

    IRON_PLATE and COPPER_PLATE for tin/silicon cell construction
    come from the iron + copper manual stashes via WithdrawFromBusAt
    — no smelting needed for those plates. TIN_PLATE and any extra
    REFRACTORY get pre-smelted at the idle pre-placed furnace at
    (15, 16) using ProduceInFurnaceAt, so the deposit can't be
    misrouted to a cell furnace whose input pallet is full of
    flowing ore.

    Mining excludes IRON_ORE and COPPER_ORE entirely — the running
    cells supply those plates. Only TIN_ORE, LIMESTONE, and the
    extra COAL needed for tin + silicon construction get mined.
    """
    targets = _phase_2_targets()

    # Running iron + copper cells supply IRON_PLATE / COPPER_PLATE.
    # Drop their recipes from the book so the BOM treats those plates
    # as leaves and the production schedule omits them.
    trimmed_book = book_without_recipes_for(book, _RUNNING_CELLS_SUPPLY)
    bom = bill_of_materials(targets, trimmed_book)

    # Strip the plate "leaves" — those become withdrawals, not mines.
    iron_plate_qty = bom.pop(int(ItemType.IRON_PLATE), 0)
    copper_plate_qty = bom.pop(int(ItemType.COPPER_PLATE), 0)

    # Phase 2 slack: same per-leaf amounts but only on leaves that are
    # actually mined this phase. Skip iron / copper ore entirely.
    phase_2_slack = {
        item: qty
        for item, qty in slack.items()
        if item in bom or item in (int(ItemType.LIMESTONE), int(ItemType.COAL))
    }
    for item, qty in phase_2_slack.items():
        bom[int(item)] = bom.get(int(item), 0) + int(qty)

    mine_goals: list[Goal] = [MineOre(item, qty) for item, qty in sorted(bom.items())]

    # Pre-smelt at the pre-placed (15, 16) furnace — TIN_PLATE +
    # REFRACTORY only (the trimmed book has no IRON_PLATE /
    # COPPER_PLATE recipes left).
    smelt_goals: list[Goal] = [
        ProduceInFurnaceAt(_PRE_PLACED_FURNACE, item, qty, book=book)
        for item, qty, machine in production_schedule(targets, trimmed_book)
        if machine == int(MachineType.FURNACE)
    ]

    # Withdraw plates from the running cells' manual stashes. The cells
    # have been firing since Phase 1 ended; by the time Phase 2 mining
    # finishes, the stashes hold many plates (~1 plate per ~5 ticks
    # per cell, vs ~50+ ticks of Phase 2 mining).
    withdraw_goals: list[Goal] = []
    if iron_plate_qty > 0:
        withdraw_goals.append(
            WithdrawFromBusAt(_IRON_MANUAL_STASH, ItemType.IRON_PLATE, iron_plate_qty)
        )
    if copper_plate_qty > 0:
        withdraw_goals.append(
            WithdrawFromBusAt(
                _COPPER_MANUAL_STASH, ItemType.COPPER_PLATE, copper_plate_qty
            )
        )

    return [
        *mine_goals,
        *smelt_goals,
        *withdraw_goals,
        # Tin + silicon coal trunks.
        *_craft_goals_for(_phase_2_belt_network_targets(), book),
        *place_belt_network(
            _phase_2_belt_paths(),
            map_size=_MAP_SIZE,
            start_near=(16, 16),
        ),
        # Tin + silicon coal miners.
        PlaceMachineAt(MachineType.MINER, (7, 24), int(Direction.DOWN)),
        PlaceMachineAt(MachineType.MINER, (7, 22), int(Direction.UP)),
        # Tin + silicon sink pallets.
        PlaceMachineAt(MachineType.PALLET, _TIN_AUTOMATION_SINK, int(Direction.DOWN)),
        PlaceMachineAt(
            MachineType.PALLET, _SILICON_AUTOMATION_SINK, int(Direction.DOWN)
        ),
        # Build tin + silicon cells. Their craft goals craft against
        # the pre-placed assembler at (17, 16); no cell has an
        # assembler so routing is unambiguous.
        *_craft_goals_for(_cell_targets_with_belt(), book),
        *build_smelter_cell_at(
            (23, 26),
            facing=int(Direction.RIGHT),
            output_split=True,
            map_size=_MAP_SIZE,
        ),
        *_craft_goals_for(_cell_targets_with_belt(), book),
        *build_smelter_cell_at(
            (15, 7),
            facing=int(Direction.RIGHT),
            output_split=True,
            map_size=_MAP_SIZE,
        ),
    ]


# ---------------------------------------------------------------------------
# Top-level goal sequence
# ---------------------------------------------------------------------------


def build_advanced_factory_goals(
    book: RecipeBook = BASE_RECIPE_BOOK,
    slack: dict[int, int] | None = None,
) -> list[Goal]:
    """Build the flat goal list for the advanced-factory rocket agent.

    Phase 1 mines + pre-smelts only what iron + copper cells (and
    their coal trunks) need, then places those two cells. Phase 2
    runs while the iron + copper cells are passively producing
    plates: it withdraws plates from those manual stashes (instead
    of mining + smelting more iron / copper ore) and uses them to
    build the tin + silicon cells. A final wait lets the manual
    stashes accumulate measurable plates before episode end.

    Args:
        book: :class:`~factoriax.recipes.RecipeBook` whose recipes
            drive the BOM and production schedule. Defaults to the
            shipped :data:`~factoriax.recipes.BASE_RECIPE_BOOK`;
            pass a tuned book (e.g.
            ``BASE_RECIPE_BOOK.with_balance(...)``) to make the
            agent's mine / smelt / craft quantities track the
            balance overlay.
        slack: Per-leaf-resource slack added on top of the BOM
            count. Defaults to :data:`_DEFAULT_SLACK` (2 per ore,
            1 LIMESTONE, 5 COAL — absorbs the
            :class:`~baselines.rocket.scripted.goals.ProduceInMachine`
            cycle's deposit-then-wait ordering quirks). Set to an
            empty dict to mine exactly the BOM amount.

    Returns:
        Flat list of :class:`~baselines.rocket.scripted.goals.Goal`
        instances ready for the planner.
    """
    if slack is None:
        slack = _DEFAULT_SLACK
    return [
        *_phase_1(book, slack),
        *_phase_2(book, slack),
        # Final wait: cells ramp up their splitter fire rate (~1
        # plate per ~5 ticks per cell) so the manual stashes
        # accumulate measurable plates before episode end.
        Wait(400),
    ]


def make_advanced_factory_rocket_agent(
    env_params: EnvParams,
    book: RecipeBook = BASE_RECIPE_BOOK,
    slack: dict[int, int] | None = None,
) -> ScriptedAgent:
    """advanced_factory: scripted agent with Phase A + Phase B (4 cells).

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
