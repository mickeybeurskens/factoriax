"""Advanced-factory rocket agent — Phase A + full Phase B (4 cells).

Two high-level phases:

- **Phase A** drops a miner + pallet on every non-coal patch (iron,
  copper, tin, silicon) and pre-smelts every plate the rest of the
  run will consume. The four coal-feed miners are deferred to Phase
  B because their push tiles are the south end of each coal trunk,
  which doesn't exist yet.

- **Phase B** builds a full splitter smelter cell on each of the
  four ore patches and lays the four coal trunks via a single
  :func:`place_belt_network` call. Each cell has the new T1 output
  design (``output_split=True``):

  * a SPLITTER replaces the old plate-bus PALLET at the same tile;
  * a *manual_stash* PALLET sits north of the splitter — the agent
    withdraws plates from here for hand-crafting;
  * an *automation belt* (south of the splitter) carries the
    splitter's other half-stream into a *sink* PALLET that Phase B
    places at the automation lane's downstream tile. The sink is
    just a drain so the splitter's both-or-nothing fire condition
    is met whenever the manual stash has space.

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
furnace/assembler each cycle, so once a cell's furnace is placed
post-Phase-A, any further ``ProduceInFurnace`` would land ore in
the wrong furnace. Phase A pre-smelts every plate the entire run
needs (computed from the production schedule) before any cell
exists. Phase B sub-phases only run ``ProduceInAssembler`` against
the pre-placed assembler at (17, 16).
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

# Phase A places one ore-node per non-coal patch (4 patches).
_PHASE_A_TARGETS: dict[int, int] = scale_inventory(
    ore_node_inventory(with_pallet=True), 4
)


def _belt_network_targets() -> dict[int, int]:
    """Items the belt-network phase crafts: 4 coal miners, 4 sink
    pallets, and the BELT count derived from the four trunks."""
    return sum_inventories(
        belt_network_inventory(_belt_paths()),
        scale_inventory(ore_node_inventory(with_pallet=False), 4),
        {int(ItemType.PALLET): 4},
    )


def _cell_targets_with_belt() -> dict[int, int]:
    """Iron / tin / silicon cells: full output_split with automation belt."""
    return smelter_cell_inventory(output_split=True, automation_belt=True)


def _cell_targets_no_belt() -> dict[int, int]:
    """Copper cell: output_split, no automation belt (splitter south
    output drops directly into the sink pallet at (21, 12))."""
    return smelter_cell_inventory(output_split=True, automation_belt=False)


def _bootstrap_targets() -> dict[int, int]:
    """Total items the bootstrap will craft.

    Sum of every per-phase craft target. Used by
    :func:`bill_of_materials` to size the MineOre goals and by
    :func:`production_schedule` to size the Phase A pre-smelt and
    every per-phase craft list.

    Edit the per-phase target helpers (or the inventory helpers in
    :mod:`baselines.rocket.scripted.goals`) to scale the bootstrap
    automatically — the BOM walk picks up the new totals.
    """
    return sum_inventories(
        _PHASE_A_TARGETS,
        _belt_network_targets(),
        scale_inventory(_cell_targets_with_belt(), 3),  # iron, tin, silicon
        _cell_targets_no_belt(),  # copper
    )


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
# Phase A — bootstrap (mine + pre-smelt + craft + place ore nodes)
# ---------------------------------------------------------------------------


def _phase_a(book: RecipeBook, slack: dict[int, int]) -> list[Goal]:
    """Mine all leaf resources, pre-smelt all plates, craft + place
    the 4 patch miners and their buffer pallets.

    The 4 coal-feed miners are deferred to the belt-network phase —
    their push tiles are the south end of each coal trunk, which
    doesn't exist yet.
    """
    targets = _bootstrap_targets()
    return [
        *_mine_goals_for(targets, book, slack),
        *_smelt_goals_for(targets, book),
        *_craft_goals_for(_PHASE_A_TARGETS, book),
        *place_ore_node((8, 9), map_size=_MAP_SIZE),  # iron
        *place_ore_node((23, 9), map_size=_MAP_SIZE),  # copper
        *place_ore_node((23, 24), map_size=_MAP_SIZE),  # tin
        *place_ore_node((15, 5), map_size=_MAP_SIZE),  # silicon
        WaitUntil(_miner_has_output_predicate(), max_ticks=30),
    ]


# ---------------------------------------------------------------------------
# Phase B — coal-trunk network + four splitter cells
# ---------------------------------------------------------------------------


def _phase_b_belt_network(book: RecipeBook) -> list[Goal]:
    """Lay the four coal trunks + place the four coal miners + sinks.

    Trunks (sink = each cell's coal_buffer pallet, placed by the cell
    helper afterwards):

    * **Iron coal** (14 belts): col 10 north with a westward bend at
      row 14 to sidestep iron's automation belt at (10, 12).
    * **Copper coal** (28 belts): col 11 + row 13 east instead of row
      12 so it sidesteps (21, 12), the copper splitter's S-output
      sink. The trunk's last belt at (23, 13) UP feeds the copper
      coal_buffer at (23, 12) from the south.
    * **Tin coal** (18 belts): drops south from (7, 24) DOWN coal
      miner, runs row 27 east. No collisions with iron/copper.
    * **Silicon coal** (21 belts): (7, 22) UP coal miner pushes north
      into (7, 21); trunk runs col 7 north past iron-patch tiles
      (7, 7..9) (belts on ore are valid placements) then row 8 east
      into the silicon coal_buffer at (15, 8).

    Trunks go down *before* the four coal miners so the silicon
    miner at (7, 22) UP — whose stand tile is (7, 23), still coal —
    doesn't block the silicon trunk's first belt at (7, 21) UP, whose
    stand tile is (7, 22). After the trunk is placed, the miner
    lands on its coal tile and pushes north into the existing belt.
    """
    return [
        *_craft_goals_for(_belt_network_targets(), book),
        *place_belt_network(
            _belt_paths(),
            map_size=_MAP_SIZE,
            # Seed the proximity walk near spawn so the first trunk
            # placed is whichever has its first ready belt closest
            # to (16, 16) — keeps the agent from crossing the whole
            # map to start at the lex-smallest tile.
            start_near=(16, 16),
        ),
        # Then the four coal miners. Each pushes into the trunk's
        # first belt (iron east into (10, 24) UP; copper south into
        # (8, 25) RIGHT; tin south into (7, 25) DOWN; silicon north
        # into (7, 21) UP).
        PlaceMachineAt(MachineType.MINER, (9, 24), int(Direction.RIGHT)),
        PlaceMachineAt(MachineType.MINER, (8, 24), int(Direction.DOWN)),
        PlaceMachineAt(MachineType.MINER, (7, 24), int(Direction.DOWN)),
        PlaceMachineAt(MachineType.MINER, (7, 22), int(Direction.UP)),
        # Sink pallets — drain each splitter's S output so the atomic-
        # fire condition is met whenever the manual stash has space.
        PlaceMachineAt(MachineType.PALLET, _IRON_AUTOMATION_SINK, int(Direction.DOWN)),
        PlaceMachineAt(
            MachineType.PALLET, _COPPER_AUTOMATION_SINK, int(Direction.DOWN)
        ),
        PlaceMachineAt(MachineType.PALLET, _TIN_AUTOMATION_SINK, int(Direction.DOWN)),
        PlaceMachineAt(
            MachineType.PALLET, _SILICON_AUTOMATION_SINK, int(Direction.DOWN)
        ),
    ]


def _phase_b_iron(book: RecipeBook) -> list[Goal]:
    """Iron splitter cell at furnace (8, 11) RIGHT."""
    return [
        *_craft_goals_for(_cell_targets_with_belt(), book),
        *build_smelter_cell_at(
            (8, 11),
            facing=int(Direction.RIGHT),
            output_split=True,
            map_size=_MAP_SIZE,
        ),
    ]


def _phase_b_copper(book: RecipeBook) -> list[Goal]:
    """Copper splitter cell mirrored across y axis at furnace (23, 11) LEFT.

    ``automation_belt=False`` because the splitter's south output
    drops *directly* onto a sink PALLET at (21, 12) — placed by
    :func:`_phase_b_belt_network`. The copper coal trunk is rerouted
    through row 13 so it never crosses (21, 12), so no CROSSING is
    needed for this layout.
    """
    return [
        WithdrawFromBusAt(_IRON_MANUAL_STASH, ItemType.IRON_PLATE, 6),
        *_craft_goals_for(_cell_targets_no_belt(), book),
        *build_smelter_cell_at(
            (23, 11),
            facing=int(Direction.LEFT),
            output_split=True,
            automation_belt=False,
            map_size=_MAP_SIZE,
        ),
    ]


def _phase_b_tin(book: RecipeBook) -> list[Goal]:
    """Tin splitter cell at furnace (23, 26) RIGHT.

    Layout::

           tin_pallet (23, 25)  ← Phase A
              furnace(23, 26) RIGHT
                arm  (24, 26) RIGHT
              splitter(25, 26) RIGHT
        coal_buffer (23, 27) UP   ← tin coal trunk sink
        manual_stash(25, 25) DOWN
        automation_belt(25, 27) DOWN
                sink (25, 28) DOWN ← placed by belt-network phase
    """
    return [
        WithdrawFromBusAt(_IRON_MANUAL_STASH, ItemType.IRON_PLATE, 4),
        WithdrawFromBusAt(_COPPER_MANUAL_STASH, ItemType.COPPER_PLATE, 4),
        *_craft_goals_for(_cell_targets_with_belt(), book),
        *build_smelter_cell_at(
            (23, 26),
            facing=int(Direction.RIGHT),
            output_split=True,
            map_size=_MAP_SIZE,
        ),
    ]


def _phase_b_silicon(book: RecipeBook) -> list[Goal]:
    """Silicon splitter cell at furnace (15, 7) RIGHT.

    Layout::

           silicon_pallet (15, 6)  ← Phase A
              furnace(15, 7) RIGHT
                arm  (16, 7) RIGHT
              splitter(17, 7) RIGHT
        coal_buffer (15, 8) UP   ← silicon coal trunk sink
        manual_stash(17, 6) DOWN
        automation_belt(17, 8) DOWN
                sink (17, 9) DOWN ← placed by belt-network phase

    Silicon ore + COAL → WAFER (the cell auto-pulls coal from the
    coal_buffer just like every other cell; the recipe gate switches
    on the ore type in the north-pallet).
    """
    return [
        WithdrawFromBusAt(_IRON_MANUAL_STASH, ItemType.IRON_PLATE, 4),
        WithdrawFromBusAt(_COPPER_MANUAL_STASH, ItemType.COPPER_PLATE, 4),
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

    Phase A pre-smelts every plate the entire run will consume.
    Phase B builds the coal-trunk belt network *first* (so coal
    flows the moment each cell's coal_buffer drops in), then the
    four splitter cells, then a final wait for production to fill
    the manual stashes.

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
        *_phase_a(book, slack),
        *_phase_b_belt_network(book),
        *_phase_b_iron(book),
        Wait(60),
        *_phase_b_copper(book),
        *_phase_b_tin(book),
        *_phase_b_silicon(book),
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
