"""Advanced-factory rocket agent — Phase A + Phase B (iron + copper).

Two high-level phases:

- **Phase A** drops a miner + pallet on each non-coal patch (iron,
  copper, tin, silicon, *and* limestone) and the four coal-feed
  miners are deferred to Phase B. Ore flows passively into the
  four patch pallets; the coal miners stall because no belt exists
  east of them yet — Phase B closes that gap by laying the iron
  and copper coal trunks via :func:`place_belt_network`.

- **Phase B** builds a splitter smelter cell on the iron and copper
  patches and lays their coal trunks. Each cell has the new T1
  output design (``output_split=True``):

  * a SPLITTER replaces the old plate-bus PALLET at the same tile;
  * a *manual_stash* PALLET sits north of the splitter — the agent
    withdraws plates from here for hand-crafting (the old
    ``_PLATE_BUS`` constants are renamed ``_*_MANUAL_STASH``);
  * an *automation belt* (south of the splitter) carries the
    splitter's other half-stream into a *sink* PALLET placed by
    Phase B at ``automation_belt + DOWN``. The sink is just a drain
    so the splitter's both-or-nothing fire condition is met when
    the manual stash has space.

The two cell coal trunks are planned together via
:func:`place_belt_network`. The copper coal trunk shares tile
(21, 12) with the copper splitter's south-output flow — the
network planner emits a CROSSING at that tile with encoding 1
(vert=DOWN for the splitter feed, horiz=RIGHT for the coal
trunk) so the two streams pass through without mixing. Iron's
coal trunk is rerouted to (10, 24) → (10, 14) → (8, 14) → (8, 12)
so it lands at the coal_buffer from the south, sidestepping the
splitter automation column entirely.

The central CONVEYOR_BELT auto-craft assembler that earlier
iterations placed at (16, 9) is **removed** in this iteration.
Routing the splitter automation flows around the assembler
module's input pallets requires a multi-tile detour with several
extra crossings. The tin and silicon cells are also dropped —
adding their coal trunks (39 BELT crafts) doubles the bootstrap
load and pushes Phase A past the 8000-tick budget. A follow-up
iteration can add tin / silicon / central-assembler back with
runtime plate sourcing from the iron and copper manual stashes.
This iteration's purpose is to land the splitter design on the
two foundational cells and demonstrate one auto-inserted
crossing in the belt network.

Pre-smelt strategy. ``ProduceInMachine`` picks the *nearest*
furnace/assembler each cycle, so once a cell's furnace is placed
post-Phase-A, any further ``ProduceInFurnace`` would land ore in
the wrong furnace. Phase A pre-smelts every plate the entire run
needs (including the 4 TIN_PLATEs and 1 COPPER_PLATE consumed by
the SPLITTER and CROSSING crafts) before any cell exists, then
Phase B sub-phases only run ``ProduceInAssembler`` against the
pre-placed assembler at (17, 16).

Map layout::

                      silicon (14, 3-5)
                              |
                      [silicon trunk: col 7 north + row 8 east
                       across iron ore tiles]
                              |
       iron (7, 7-9)                copper (22-24, 7-9)
        |                           |
        [iron trunk col 10 +        [copper trunk: col 11 + row 12
         westward bend at row 14]     CROSSING at (21, 12) with
                                      copper splitter S output]
        |                           |
                spawn (16, 16)
                              |
                              | [tin trunk row 26 east]
       coal (7-9, 22-24)            tin (22-24, 22-24)
        4 coal miners
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

# The agent withdraws plates from these for hand-crafting downstream
# Phase B craft lists. Each is the splitter's NORTH output PALLET.
_IRON_MANUAL_STASH = (10, 10)
_COPPER_MANUAL_STASH = (21, 10)

# Sink PALLETs placed by Phase B at the automation lane's downstream
# tile (= splitter+2*DOWN for iron, = splitter+1*DOWN for copper
# where the splitter's S output flows through a CROSSING at
# (21, 12) into the sink at (21, 13)).
_IRON_AUTOMATION_SINK = (10, 13)
_COPPER_AUTOMATION_SINK = (21, 13)

_MAP_SIZE: tuple[int, int] = (32, 32)


# ---------------------------------------------------------------------------
# Phase A — bootstrap everything, then drop miners + pallets
# ---------------------------------------------------------------------------


def _phase_a_bootstrap_mine_and_smelt() -> list[Goal]:
    """Hand-mine + smelt every plate iron and copper Phase B need.

    Per-recipe plate budget:

    * Phase A: 10 WIRE + 5 MINER (5 IRON_PLATE) + 5 PALLET
      (5 TIN_PLATE).
    * Iron cell (with output_split=True): 2 IRON + 5 COPPER + 6 TIN.
    * Copper cell (with output_split=True, automation_belt=False):
      1 IRON + 5 COPPER + 6 TIN.
    * Belt network: 41 BELT (41 IRON + 41 COPPER), 1 CROSSING
      (1 COPPER), 1 cell coal MINER (1 IRON + 1 COPPER + 1 TIN —
      iron's coal miner consumes Phase A's MINER slack), 2 sink
      PALLETs (2 TIN_PLATE + 2 WIRE).
    * REFRACTORY × 2 (iron + copper FURNACE recipes) = 2 LIMESTONE
      + 2 COAL.

    Totals (with +2 slack each): 53 IRON, 67 COPPER, 33 TIN, 2
    LIMESTONE, 155 COAL.
    """
    return [
        MineOre(ItemType.IRON_ORE, 53),
        MineOre(ItemType.COPPER_ORE, 67),
        MineOre(ItemType.TIN_ORE, 33),
        MineOre(ItemType.LIMESTONE, 2),
        MineOre(ItemType.COAL, 155),
        ProduceInFurnace(ItemType.IRON_PLATE, 51),
        ProduceInFurnace(ItemType.COPPER_PLATE, 64),
        ProduceInFurnace(ItemType.TIN_PLATE, 31),
        ProduceInFurnace(ItemType.REFRACTORY, 2),
    ]


def _phase_a_craft_and_place() -> list[Goal]:
    """Craft Phase A's 10 WIRE + 5 MINER + 5 PALLET, then place them.

    Each non-coal patch gets a south-edge DOWN-facing miner pushing
    into a buffer pallet. The fifth crafted miner is consumed by
    Phase B's iron coal feed (each cell phase crafts +1 MINER for
    the next cell, but iron borrows it from Phase A's slack).
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
# Phase B sub-phases — one cell at a time, plus the combined coal-trunk
# belt network and the four sink pallets at the end.
# ---------------------------------------------------------------------------


def _phase_b_iron() -> list[Goal]:
    """Iron splitter cell. Manual stash at (10, 10), automation belt
    at (10, 12) DOWN, sink (placed by phase_b_belt_network later) at
    (10, 13). Coal trunk laid by phase_b_belt_network."""
    return [
        ProduceInAssembler(ItemType.WIRE, 3),
        # 1 cell ARM (no extractor) + 1 SPLITTER. Splitter recipe
        # consumes 1 TIN_PLATE + 1 COAL pre-smelted in Phase A.
        ProduceInAssembler(ItemType.PALLET, 2),  # coal_buffer + manual_stash
        ProduceInAssembler(ItemType.ARM, 1),
        ProduceInAssembler(ItemType.SPLITTER, 1),
        ProduceInAssembler(ItemType.FURNACE, 1),
        # 1 BELT for the cell's automation belt (cell helper emits).
        ProduceInAssembler(ItemType.CONVEYOR_BELT, 1),
        *build_smelter_cell_at(
            (8, 11),
            facing=int(Direction.RIGHT),
            output_split=True,
            map_size=_MAP_SIZE,
        ),
    ]


def _phase_b_copper() -> list[Goal]:
    """Copper splitter cell mirrored across y axis.

    ``automation_belt=False`` because the copper splitter's south
    output (21, 12) sits on the copper coal trunk's row-12 segment.
    :func:`place_belt_network` planted in
    :func:`_phase_b_belt_network` emits a CROSSING at (21, 12)
    instead of a plain belt — vert=DOWN (splitter S output),
    horiz=RIGHT (coal trunk going east toward (23, 12) coal_buffer).
    Sink pallet at (21, 13) catches the splitter's drained S
    output stream.
    """
    return [
        WithdrawFromBusAt(_IRON_MANUAL_STASH, ItemType.IRON_PLATE, 6),
        ProduceInAssembler(ItemType.WIRE, 3),
        ProduceInAssembler(ItemType.PALLET, 2),  # coal_buffer + manual_stash
        ProduceInAssembler(ItemType.ARM, 1),
        ProduceInAssembler(ItemType.SPLITTER, 1),
        ProduceInAssembler(ItemType.FURNACE, 1),
        # No automation_belt for copper — the network planner adds a
        # CROSSING at (21, 12) instead. 1 CROSSING crafted here
        # (recipe: 1 COPPER_PLATE + 1 COAL).
        ProduceInAssembler(ItemType.CROSSING, 1),
        *build_smelter_cell_at(
            (23, 11),
            facing=int(Direction.LEFT),
            output_split=True,
            automation_belt=False,
            map_size=_MAP_SIZE,
        ),
    ]


def _phase_b_belt_network() -> list[Goal]:
    """Lay the iron and copper coal trunks via :func:`place_belt_network`.

    Trunks:

    * **Iron coal**: rerouted from the canonical column-10 corner-at-
      row-12 path to (10, 24) → (10, 14) → (8, 14) → (8, 12) so the
      trunk reaches the coal_buffer from the south (via (8, 13) UP)
      instead of from the east — sidesteps (10, 12), which is iron's
      splitter automation belt.
    * **Copper coal**: canonical (8, 25) → (11, 25) → (11, 12) →
      (23, 12). Tile (21, 12) is shared with the copper splitter's
      south-output flow; the network planner emits a CROSSING with
      vert=DOWN, horiz=RIGHT (encoding 1) so the two streams pass
      through without mixing.

    Plus a single-tile path representing copper splitter's S output
    going DOWN through (21, 12) into sink (21, 13). This is what
    creates the CROSSING when planned alongside the copper coal
    trunk.

    Crafts the 14 (iron) + 27 (copper) + 0 (copper auto, replaced
    by the crossing) = 41 BELTs and 1 CROSSING. Also crafts the 2
    cell coal miners (iron + copper) and the 2 sink PALLETs (iron
    + copper).
    """
    iron_coal_waypoints = [(10, 24), (10, 14), (8, 14), (8, 12)]
    copper_coal_waypoints = [(8, 25), (11, 25), (11, 12), (23, 12)]
    # The copper splitter's S output is implicit at (21, 11). The
    # path's first tile (21, 12) is the auto-stream's first
    # placed-cell tile (becomes a CROSSING when planned alongside
    # copper coal). Sink (21, 13).
    copper_auto_waypoints = [(21, 12), (21, 13)]

    return [
        # Coal miners. The iron miner consumes Phase A's MINER slack;
        # craft 1 more for copper.
        ProduceInAssembler(ItemType.MINER, 1),
        # 14 + 27 + 0 BELTs from the network (copper auto's tile is
        # the crossing, not a belt). Plus 1 CROSSING.
        ProduceInAssembler(ItemType.CONVEYOR_BELT, 41),
        ProduceInAssembler(ItemType.CROSSING, 1),
        # Two sink pallets (iron + copper).
        ProduceInAssembler(ItemType.PALLET, 2),
        # Place the two coal miners before the network so
        # place_belt_network's first belt at each miner's push tile
        # has a stand tile pointing at the miner.
        PlaceMachineAt(MachineType.MINER, (9, 24), int(Direction.RIGHT)),
        PlaceMachineAt(MachineType.MINER, (8, 24), int(Direction.DOWN)),
        # Plan + place the two coal trunks + the copper auto path.
        # The single CROSSING at (21, 12) lands automatically.
        *place_belt_network(
            [
                BeltPath(iron_coal_waypoints, label="iron coal"),
                BeltPath(copper_coal_waypoints, label="copper coal"),
                BeltPath(copper_auto_waypoints, label="copper auto"),
            ],
            map_size=_MAP_SIZE,
        ),
        # Sink pallets — drain the splitter S outputs so each
        # splitter's atomic-fire condition is met whenever the
        # manual stash also has space.
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
        # Lay the coal trunks (and place the four coal miners + sink
        # pallets) before any cell. The trunks' last belts push into
        # the cells' coal_buffer tiles; those tiles are dirt until
        # each cell phase places them, and the trunk belts harmlessly
        # buffer up to 3 coal each in the meantime.
        *_phase_b_belt_network(),
        *_phase_b_iron(),
        Wait(60),
        *_phase_b_copper(),
        # Final wait: cells ramp up their splitter fire rate (~1
        # plate per ~5 ticks per cell) so the manual stashes
        # accumulate measurable plates before episode end.
        Wait(400),
    ]


def make_advanced_factory_rocket_agent(env_params: EnvParams) -> ScriptedAgent:
    """advanced_factory: scripted agent with Phase A + Phase B (4 cells)."""
    return ScriptedAgent(env_params, Planner(build_advanced_factory_goals()))
