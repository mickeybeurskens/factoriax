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
needs (95 IRON_PLATE, 118 COPPER_PLATE, 50 TIN_PLATE, 4 REFRACTORY)
before any cell exists. Phase B sub-phases only run
``ProduceInAssembler`` against the pre-placed assembler at
(17, 16).
"""

from __future__ import annotations

from factoriax.constants import Direction, ItemType, MachineType
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
    build_smelter_cell_at,
    place_belt_network,
    place_ore_node,
)
from .planner import Planner

# ---------------------------------------------------------------------------
# Manual-stash bus tiles (one per cell, replaces the old _PLATE_BUS)
# ---------------------------------------------------------------------------

_IRON_MANUAL_STASH = (10, 10)
_COPPER_MANUAL_STASH = (21, 10)
_TIN_MANUAL_STASH = (25, 25)
_SILICON_MANUAL_STASH = (17, 6)

# Sink PALLETs placed by Phase B at the automation lane's downstream
# tile. Iron / tin / silicon flow splitter -> automation_belt -> sink.
# Copper has ``automation_belt=False`` so its splitter's south output
# pushes *directly* into a sink at (21, 12); the copper coal trunk is
# rerouted through row 13 to avoid that tile.
_IRON_AUTOMATION_SINK = (10, 13)
_COPPER_AUTOMATION_SINK = (21, 12)
_TIN_AUTOMATION_SINK = (25, 28)
_SILICON_AUTOMATION_SINK = (17, 9)

_MAP_SIZE: tuple[int, int] = (32, 32)


# ---------------------------------------------------------------------------
# Phase A — bootstrap everything, then drop miners + pallets
# ---------------------------------------------------------------------------


def _phase_a_bootstrap_mine_and_smelt() -> list[Goal]:
    """Hand-mine + smelt every plate the four cells will consume.

    The grand total is the sum across:

    * Phase A crafts: 10 WIRE, 5 MINER, 5 PALLET (10 COPPER, 5 IRON,
      15 TIN at plate level after the WIRE intermediate is unrolled).
    * Four splitter cells (output_split=True): each cell consumes
      2 IRON + 5 COPPER + 6 TIN + 2 COAL + 1 LIMESTONE for its
      ARM/PALLET×2/FURNACE/SPLITTER/BELT (the copper cell drops the
      automation BELT and instead lets the network's row-13 reroute
      finish the coal line, so it consumes 1 IRON + 4 COPPER + 6 TIN
      + 2 COAL + 1 LIMESTONE).
    * The 81-belt network (4 trunks; copper coal trunk is one belt
      longer than the row-12 variant because it bends through row
      13 to sidestep the splitter S column): 81 IRON + 81 COPPER
      for the belts, +3 MINERs for the coal-feed miners (3 IRON +
      3 COPPER + 3 TIN through WIRE), +4 sink PALLETs (4 TIN + 4
      COPPER + 4 TIN through WIRE).

    Totals (with +2 ore slack each, +1 LIMESTONE slack, +5 COAL
    slack to absorb the ProduceInFurnace cycle's 1-coal-per-smelt
    cost): 97 IRON_ORE / 120 COPPER_ORE / 52 TIN_ORE / 5 LIMESTONE /
    282 COAL. ProduceInFurnace targets: 95 / 118 / 50 / 4.
    """
    return [
        MineOre(ItemType.IRON_ORE, 97),
        MineOre(ItemType.COPPER_ORE, 120),
        MineOre(ItemType.TIN_ORE, 52),
        MineOre(ItemType.LIMESTONE, 5),
        MineOre(ItemType.COAL, 282),
        ProduceInFurnace(ItemType.IRON_PLATE, 95),
        ProduceInFurnace(ItemType.COPPER_PLATE, 118),
        ProduceInFurnace(ItemType.TIN_PLATE, 50),
        ProduceInFurnace(ItemType.REFRACTORY, 4),
    ]


def _phase_a_craft_and_place() -> list[Goal]:
    """Craft Phase A's 10 WIRE + 5 MINER + 5 PALLET, then place them.

    Each non-coal patch gets a south-edge DOWN-facing miner pushing
    into a buffer pallet. The fifth crafted miner is consumed by the
    iron coal feed in Phase B (Phase B's ``_phase_b_belt_network``
    crafts +3 more for the other three coal miners).
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
# Phase B — combined coal-trunk network + four splitter cells
# ---------------------------------------------------------------------------


def _phase_b_belt_network() -> list[Goal]:
    """Lay the four coal trunks + place the four coal miners + sinks.

    Trunks (sink = each cell's coal_buffer pallet, placed by the cell
    helper afterwards):

    * **Iron coal**: (10, 24) → (10, 14) → (8, 14) → (8, 12). The
      westward bend at row 14 sidesteps (10, 12), which is iron's
      splitter automation belt. 14 belts.
    * **Copper coal**: (8, 25) → (11, 25) → (11, 13) → (23, 13) →
      (23, 12). 28 belts. Bends down through col 11 to row 13
      instead of row 12 so it sidesteps (21, 12), which is the
      copper splitter's S-output sink — no CROSSING needed. The
      trunk's last belt at (23, 13) UP feeds the copper coal_buffer
      at (23, 12) from the south.
    * **Tin coal**: (7, 25) → (7, 27) → (22, 27) → (23, 27). 18
      belts. Drops south from the (7, 24) DOWN coal miner, runs row
      27 east. No collisions with iron/copper trunks.
    * **Silicon coal**: (7, 21) → (7, 8) → (15, 8). 21 belts. The
      (7, 22) UP coal miner pushes north into (7, 21); the trunk
      runs col 7 north past iron-patch tiles (7, 7..9) (belts on ore
      are valid placements) then row 8 east into the silicon
      coal_buffer at (15, 8).

    Trunks are laid *before* the four coal miners so the silicon
    miner at (7, 22) UP — whose stand tile is (7, 23), still coal —
    doesn't block the silicon trunk's first belt at (7, 21) UP, whose
    stand tile is (7, 22). After the trunk is placed, the miner
    lands on its coal tile and pushes north into the existing belt.

    Crafts: 81 BELT (14 + 28 + 18 + 21), 3 MINER (4 coal miners
    total; the iron one consumes Phase A's slack), 4 sink PALLET,
    plus 7 WIRE intermediate (3 for MINERs, 4 for sink PALLETs).
    """
    iron_coal_waypoints = [(10, 24), (10, 14), (8, 14), (8, 12)]
    copper_coal_waypoints = [(8, 25), (11, 25), (11, 13), (23, 13), (23, 12)]
    tin_coal_waypoints = [(7, 25), (7, 27), (22, 27), (23, 27)]
    silicon_coal_waypoints = [(7, 21), (7, 8), (15, 8)]

    return [
        # Three additional cell coal MINERs (the iron coal miner uses
        # Phase A's +1 slack so only 3 fresh MINERs are crafted here).
        ProduceInAssembler(ItemType.WIRE, 7),  # 3 for MINER + 4 for PALLET
        ProduceInAssembler(ItemType.MINER, 3),
        # 14 + 28 + 18 + 21 = 81 BELTs.
        ProduceInAssembler(ItemType.CONVEYOR_BELT, 81),
        # Four sink pallets (one per cell).
        ProduceInAssembler(ItemType.PALLET, 4),
        # Lay the trunks first so the silicon miner's eventual location
        # at (7, 22) doesn't block the silicon trunk's first belt at
        # (7, 21) UP whose stand tile is (7, 22).
        *place_belt_network(
            [
                BeltPath(iron_coal_waypoints, label="iron coal"),
                BeltPath(copper_coal_waypoints, label="copper coal"),
                BeltPath(tin_coal_waypoints, label="tin coal"),
                BeltPath(silicon_coal_waypoints, label="silicon coal"),
            ],
            map_size=_MAP_SIZE,
            # Seed the proximity-aware walk near the spawn so the
            # first trunk placed is whichever has its first ready
            # belt closest to (16, 16) — keeps the agent from
            # crossing the whole map to start at the lex-smallest
            # tile when one trunk is already nearby.
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
        PlaceMachineAt(
            MachineType.PALLET,
            _IRON_AUTOMATION_SINK,
            int(Direction.DOWN),
        ),
        PlaceMachineAt(
            MachineType.PALLET,
            _COPPER_AUTOMATION_SINK,
            int(Direction.DOWN),
        ),
        PlaceMachineAt(
            MachineType.PALLET,
            _TIN_AUTOMATION_SINK,
            int(Direction.DOWN),
        ),
        PlaceMachineAt(
            MachineType.PALLET,
            _SILICON_AUTOMATION_SINK,
            int(Direction.DOWN),
        ),
    ]


def _phase_b_iron() -> list[Goal]:
    """Iron splitter cell at furnace (8, 11) RIGHT."""
    return [
        ProduceInAssembler(ItemType.WIRE, 3),
        ProduceInAssembler(ItemType.PALLET, 2),  # coal_buffer + manual_stash
        ProduceInAssembler(ItemType.ARM, 1),
        ProduceInAssembler(ItemType.SPLITTER, 1),
        ProduceInAssembler(ItemType.FURNACE, 1),
        ProduceInAssembler(ItemType.CONVEYOR_BELT, 1),  # automation belt
        *build_smelter_cell_at(
            (8, 11),
            facing=int(Direction.RIGHT),
            output_split=True,
            map_size=_MAP_SIZE,
        ),
    ]


def _phase_b_copper() -> list[Goal]:
    """Copper splitter cell mirrored across y axis at furnace (23, 11) LEFT.

    ``automation_belt=False`` because the splitter's south output
    drops *directly* onto a sink PALLET at (21, 12) — placed by
    :func:`_phase_b_belt_network`. The copper coal trunk is rerouted
    through row 13 so it never crosses (21, 12), so no CROSSING is
    needed for this layout.
    """
    return [
        WithdrawFromBusAt(_IRON_MANUAL_STASH, ItemType.IRON_PLATE, 6),
        ProduceInAssembler(ItemType.WIRE, 3),
        ProduceInAssembler(ItemType.PALLET, 2),
        ProduceInAssembler(ItemType.ARM, 1),
        ProduceInAssembler(ItemType.SPLITTER, 1),
        ProduceInAssembler(ItemType.FURNACE, 1),
        *build_smelter_cell_at(
            (23, 11),
            facing=int(Direction.LEFT),
            output_split=True,
            automation_belt=False,
            map_size=_MAP_SIZE,
        ),
    ]


def _phase_b_tin() -> list[Goal]:
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
        ProduceInAssembler(ItemType.WIRE, 3),
        ProduceInAssembler(ItemType.PALLET, 2),
        ProduceInAssembler(ItemType.ARM, 1),
        ProduceInAssembler(ItemType.SPLITTER, 1),
        ProduceInAssembler(ItemType.FURNACE, 1),
        ProduceInAssembler(ItemType.CONVEYOR_BELT, 1),
        *build_smelter_cell_at(
            (23, 26),
            facing=int(Direction.RIGHT),
            output_split=True,
            map_size=_MAP_SIZE,
        ),
    ]


def _phase_b_silicon() -> list[Goal]:
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
        ProduceInAssembler(ItemType.WIRE, 3),
        ProduceInAssembler(ItemType.PALLET, 2),
        ProduceInAssembler(ItemType.ARM, 1),
        ProduceInAssembler(ItemType.SPLITTER, 1),
        ProduceInAssembler(ItemType.FURNACE, 1),
        ProduceInAssembler(ItemType.CONVEYOR_BELT, 1),
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


def build_advanced_factory_goals() -> list[Goal]:
    """Phase A pre-smelts everything. Phase B builds the coal-trunk
    belt network *first* (so coal flows as soon as each cell's
    coal_buffer drops in), then the four splitter cells, then a
    final wait for production to fill the manual stashes.
    """
    return [
        *_phase_a_bootstrap_mine_and_smelt(),
        *_phase_a_craft_and_place(),
        *_phase_b_belt_network(),
        *_phase_b_iron(),
        Wait(60),
        *_phase_b_copper(),
        *_phase_b_tin(),
        *_phase_b_silicon(),
        # Final wait: cells ramp up their splitter fire rate (~1
        # plate per ~5 ticks per cell) so the manual stashes
        # accumulate measurable plates before episode end.
        Wait(400),
    ]


def make_advanced_factory_rocket_agent(env_params: EnvParams) -> ScriptedAgent:
    """advanced_factory: scripted agent with Phase A + Phase B (4 cells)."""
    return ScriptedAgent(env_params, Planner(build_advanced_factory_goals()))
