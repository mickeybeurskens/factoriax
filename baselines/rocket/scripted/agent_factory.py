"""Factory-oriented variant of the scripted rocket agent.

Sets up per-ore-node factories before launching the rocket chain:

- Each non-coal patch (iron, copper, tin, silicon) gets a miner +
  pallet + local furnace, laid out in a vertical strip just south
  of the patch. The miner is placed facing DOWN so its per-tick
  extraction pushes straight into the pallet; the furnace sits one
  more tile south for the agent to deposit ore and withdraw plates.
- Coal gets a snake: four miners arranged in an L that feed each
  other (M1 → M2 → M3 → M4 → pallet), ending at a pallet on the
  dirt tile east of the patch.

Directions are explicit via :class:`PlaceMachineAt`, which picks a
stand tile opposite the facing direction so the engine's "machine
faces player's direction" rule lands the miner's push in the right
tile.

The rest of the plan (bulk intermediates → rocket sub-assemblies →
rocket) reuses sequential :class:`ProduceInMachine` goals. No
``PipelinedProduce`` — the cross-goal debris problem isn't worth
the speedup here.
"""

from __future__ import annotations

from factoriax.constants import Direction, ItemType, MachineType
from factoriax.state import EnvParams

from .agent import ScriptedAgent, _miner_has_output_predicate
from .goals import (
    DepositInto,
    Goal,
    MineOre,
    PlaceMachine,
    PlaceMachineAt,
    ProduceInAssembler,
    ProduceInFurnace,
    WaitUntil,
    free_tile_near_player,
)
from .planner import Planner

# ---------------------------------------------------------------------------
# Node factory layouts. Each non-coal patch is 3×3; the strip below
# the south edge holds miner → pallet → furnace.
# ---------------------------------------------------------------------------


def _node_factory(
    patch_x: int,
    patch_y_bottom: int,
) -> list[Goal]:
    """Miner + pallet + furnace strip south of an ore patch.

    Placement order matters: furnace first (furthest south, keeps
    the tiles above walkable), then pallet, then miner.

    Args:
        patch_x: X of the patch center (middle column).
        patch_y_bottom: Y of the patch's south edge (a valid ore tile).
    """
    miner_tile = (patch_x, patch_y_bottom)
    pallet_tile = (patch_x, patch_y_bottom + 1)
    furnace_tile = (patch_x, patch_y_bottom + 2)
    return [
        # Furnace points UP — direction doesn't matter for furnaces,
        # but we need *some* facing. Stand is two south, face UP.
        PlaceMachineAt(MachineType.FURNACE, furnace_tile, int(Direction.UP)),
        # Pallet. Stand on the ore tile just north of the pallet,
        # face DOWN.
        PlaceMachineAt(MachineType.PALLET, pallet_tile, int(Direction.DOWN)),
        # Miner on the patch edge. Stand one tile further north (on
        # ore, still walkable) facing DOWN. Miner pushes DOWN into
        # the pallet we just placed.
        PlaceMachineAt(MachineType.MINER, miner_tile, int(Direction.DOWN)),
    ]


def _coal_snake() -> list[Goal]:
    """Four coal miners chained into a pallet east of the patch.

    Layout (coal patch at (7-9, 22-24)):

        (7,22) M1↓
        (7,23) M2→ (8,23) M3→ (9,23) M4→ (10,23) Pallet
    """
    return [
        # Pallet first — the eastern terminus. Stand two east, face
        # LEFT. Pallet ends up at (10, 23).
        PlaceMachineAt(MachineType.PALLET, (10, 23), int(Direction.LEFT)),
        # M4: push east into pallet. Stand at (8, 23) (coal, walkable)
        # facing RIGHT.
        PlaceMachineAt(MachineType.MINER, (9, 23), int(Direction.RIGHT)),
        # M3: push east into M4. Stand at (7, 23) facing RIGHT.
        PlaceMachineAt(MachineType.MINER, (8, 23), int(Direction.RIGHT)),
        # M2: push east into M3. Stand at (6, 23) facing RIGHT.
        PlaceMachineAt(MachineType.MINER, (7, 23), int(Direction.RIGHT)),
        # M1: push south into M2. Stand at (7, 21) facing DOWN.
        PlaceMachineAt(MachineType.MINER, (7, 22), int(Direction.DOWN)),
    ]


def build_factory_rocket_goals() -> list[Goal]:
    """Goal list for the node-factory variant.

    Phase layout:

    A. Hand-mine raw ore for the starter infrastructure.
    B. Hand-smelt starter plates via the pre-placed furnace.
    C. Assemble the 8 miners + 5 pallets + 4 furnaces.
    D. Deploy node factories (iron/copper/tin/silicon) and the
       coal snake with explicit facings so miners push directly
       into pallets.
    E. Wait for automation to spin up.
    F-H. Bulk ore → plates → intermediates → sub-assemblies → rocket.
    I. Final placements for achievements.
    """
    return [
        # ---- Phase A — starter hand-mining ----
        # 8 miners (8 iron + 8 wire = 8 iron + 8 copper + 8 tin),
        # 5 pallets (5 tin + 5 wire = 5 copper + 10 tin),
        # 4 furnaces (4 iron + 4 refractory = 4 iron + 4 coal).
        # Totals: iron 12, copper 13, tin 18, coal 4.
        MineOre(ItemType.IRON_ORE, 15),
        MineOre(ItemType.COPPER_ORE, 14),
        MineOre(ItemType.TIN_ORE, 20),
        MineOre(ItemType.COAL, 5),
        # ---- Phase B — starter smelts ----
        ProduceInFurnace(ItemType.IRON_PLATE, 12),
        ProduceInFurnace(ItemType.COPPER_PLATE, 13),
        ProduceInFurnace(ItemType.TIN_PLATE, 18),
        ProduceInFurnace(ItemType.REFRACTORY, 4),
        # ---- Phase C — starter components ----
        ProduceInAssembler(ItemType.WIRE, 13),
        ProduceInAssembler(ItemType.MINER, 8),
        ProduceInAssembler(ItemType.PALLET, 5),
        ProduceInAssembler(ItemType.FURNACE, 4),
        # ---- Phase D — deploy factories with explicit facings ----
        # Iron patch at (7-9, 7-9). Strip: miner (8, 9) DOWN, pallet
        # (8, 10), furnace (8, 11).
        *_node_factory(patch_x=8, patch_y_bottom=9),
        # Copper patch at (22-24, 7-9).
        *_node_factory(patch_x=23, patch_y_bottom=9),
        # Tin patch at (22-24, 22-24). Strip south of patch.
        *_node_factory(patch_x=23, patch_y_bottom=24),
        # Silicon patch at (14-16, 3-5). Strip south of patch.
        *_node_factory(patch_x=15, patch_y_bottom=5),
        # Coal snake on the coal patch (7-9, 22-24).
        *_coal_snake(),
        # ---- Phase E — let the automation warm up ----
        WaitUntil(_miner_has_output_predicate(), max_ticks=30),
        # ---- Phase F — bulk hand-mining (automation is supplementary) ----
        # The placed miners push directly into their pallets, but we
        # can't count on agent timing to drain them promptly. Hand
        # mining is still the reliable bulk path.
        MineOre(ItemType.IRON_ORE, 45),
        MineOre(ItemType.COPPER_ORE, 50),
        MineOre(ItemType.TIN_ORE, 42),
        MineOre(ItemType.SILICON, 22),
        MineOre(ItemType.COAL, 2),
        # ---- Phase G — bulk smelts (nearest furnace wins) ----
        ProduceInFurnace(ItemType.IRON_PLATE, 45),
        ProduceInFurnace(ItemType.COPPER_PLATE, 50),
        ProduceInFurnace(ItemType.TIN_PLATE, 42),
        ProduceInFurnace(ItemType.WAFER, 22),
        # ---- Phase H — intermediates, components, sub-assemblies ----
        ProduceInAssembler(ItemType.FRAME, 25),
        ProduceInAssembler(ItemType.WIRE, 20),
        ProduceInAssembler(ItemType.CIRCUIT, 20),
        ProduceInAssembler(ItemType.MOTOR, 10),
        ProduceInAssembler(ItemType.SENSOR, 10),
        ProduceInAssembler(ItemType.HULL, 6),
        ProduceInAssembler(ItemType.ENGINE_UNIT, 4),
        ProduceInAssembler(ItemType.AVIONICS, 4),
        ProduceInAssembler(ItemType.ROCKET_CORE, 4),
        # ---- Phase I — final placeables ----
        ProduceInAssembler(ItemType.CONVEYOR_BELT, 5),
        ProduceInAssembler(ItemType.ARM, 1),
        ProduceInAssembler(ItemType.ROCKET, 1),
        PlaceMachine(MachineType.CONVEYOR_BELT, free_tile_near_player()),
        PlaceMachine(MachineType.CONVEYOR_BELT, free_tile_near_player()),
        PlaceMachine(MachineType.CONVEYOR_BELT, free_tile_near_player()),
        PlaceMachine(MachineType.CONVEYOR_BELT, free_tile_near_player()),
        PlaceMachine(MachineType.CONVEYOR_BELT, free_tile_near_player()),
        PlaceMachine(MachineType.ARM, free_tile_near_player()),
        # pallet_filled: drop any plate into any pallet.
        DepositInto(MachineType.PALLET, ItemType.IRON_PLATE),
        # ---- Phase J — capstone ----
        PlaceMachine(MachineType.ROCKET, free_tile_near_player()),
    ]


def make_factory_rocket_agent(env_params: EnvParams) -> ScriptedAgent:
    """Factory: scripted factory-building rocket agent."""
    return ScriptedAgent(env_params, Planner(build_factory_rocket_goals()))
