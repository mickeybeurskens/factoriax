"""Factory-oriented variant of the scripted rocket agent.

Two kinds of factory go in before the rocket chain starts:

1. **Per-ore-node auto-miners.** Each non-coal patch (iron, copper,
   tin, silicon) gets a miner + pallet. The miner is placed facing
   DOWN into the pallet so its per-tick extraction pushes straight
   into a 1000-cap storage bin — the agent doesn't have to stand
   on the patch to mine by hand.
2. **Coal snake.** Four coal miners chained so their pushes funnel
   east into a pallet on the dirt tile beyond the patch. Keeps a
   steady supply of coal for refractory production.
3. **Central smelter/assembler bank.** Two extra furnaces + two
   extra assemblers placed near spawn so :class:`PipelinedProduce`
   can rotate bulk work across 3 furnaces and 3 assemblers.

Placement directions are explicit via :class:`PlaceMachineAt`:
the engine sets a placed machine's ``ent_direction`` to the
player's facing at placement time, so :class:`PlaceMachineAt`
navigates to a stand tile opposite the desired direction and
rotates the player first.

Designed to fit the 8000-step benchmark budget.
"""

from __future__ import annotations

from factoriax.constants import Direction, ItemType, MachineType
from factoriax.state import EnvParams

from .agent import ScriptedAgent, _miner_has_output_predicate
from .goals import (
    DepositInto,
    Goal,
    MineOre,
    PipelinedProduce,
    PlaceMachine,
    PlaceMachineAt,
    ProduceInAssembler,
    ProduceInFurnace,
    WaitUntil,
    free_tile_near_player,
)
from .planner import Planner

# ---------------------------------------------------------------------------
# Node layouts
# ---------------------------------------------------------------------------


def _node_auto_miner(
    patch_x: int,
    patch_y_bottom: int,
) -> list[Goal]:
    """Miner + pallet south of an ore patch.

    Miner is placed on the south-edge tile of the patch facing DOWN;
    pallet goes on the dirt tile below it so the miner's push lands
    there. Per-tick flow: miner extracts 3 ore (capped by
    ``MINER_OUTPUT_CAP``), immediately pushes into the pallet's 1000
    slot, mines again. The pallet therefore accumulates ore
    continuously without agent intervention.
    """
    miner_tile = (patch_x, patch_y_bottom)
    pallet_tile = (patch_x, patch_y_bottom + 1)
    return [
        # Pallet first — stand on the ore tile just north of it,
        # face DOWN.
        PlaceMachineAt(MachineType.PALLET, pallet_tile, int(Direction.DOWN)),
        # Miner on the patch edge. Stand one tile further north (on
        # ore, still walkable) facing DOWN.
        PlaceMachineAt(MachineType.MINER, miner_tile, int(Direction.DOWN)),
    ]


def _coal_snake() -> list[Goal]:
    """Four coal miners chained into a pallet east of the patch.

    Layout (coal patch at (7-9, 22-24)):

        (7,22) M1↓
        (7,23) M2→  (8,23) M3→  (9,23) M4→  (10,23) Pallet
    """
    return [
        # Pallet: stand two east, face LEFT.
        PlaceMachineAt(MachineType.PALLET, (10, 23), int(Direction.LEFT)),
        # M4: push east into pallet. Stand at (8, 23) facing RIGHT.
        PlaceMachineAt(MachineType.MINER, (9, 23), int(Direction.RIGHT)),
        # M3: push east into M4. Stand at (7, 23) facing RIGHT.
        PlaceMachineAt(MachineType.MINER, (8, 23), int(Direction.RIGHT)),
        # M2: push east into M3. Stand at (6, 23) facing RIGHT.
        PlaceMachineAt(MachineType.MINER, (7, 23), int(Direction.RIGHT)),
        # M1: push south into M2. Stand at (7, 21) facing DOWN.
        PlaceMachineAt(MachineType.MINER, (7, 22), int(Direction.DOWN)),
    ]


def build_factory_rocket_goals() -> list[Goal]:
    """Goal list for the factory variant.

    Budget derivation:

    - 4 node miners + 4 node pallets = 4 iron + 4 wire (4 iron + 4
      copper + 4 tin) + 4 tin + 4 wire (4 copper + 4 tin).
    - 4 coal miners + 1 coal pallet = 4 iron + 4 wire + 1 tin + 1
      wire.
    - 2 extra furnaces = 2 iron + 2 refractory (2 coal).
    - 2 extra assemblers = 2 frame + 2 circuit = 2 iron + 2 tin
      + 2 copper + 2 wafer.

    Total starter plates: iron 14, copper 16, tin 19, wafer 2,
    refractory 2. Raw ore (1:1): iron 14, copper 16, tin 19,
    silicon 2, coal 2 (plus mining slack).
    """
    return [
        # ---- Phase A — starter hand-mining ----
        # Needs (items): 12 iron + 15 copper + 20 tin + 2 silicon +
        # 2 coal = 51 raw. Plus a small slack per type.
        MineOre(ItemType.IRON_ORE, 14),
        MineOre(ItemType.COPPER_ORE, 17),
        MineOre(ItemType.TIN_ORE, 23),
        MineOre(ItemType.SILICON, 3),
        MineOre(ItemType.COAL, 3),
        # ---- Phase B — starter smelts via pre-placed furnace ----
        ProduceInFurnace(ItemType.IRON_PLATE, 14),
        ProduceInFurnace(ItemType.COPPER_PLATE, 17),
        ProduceInFurnace(ItemType.TIN_PLATE, 23),
        ProduceInFurnace(ItemType.WAFER, 3),
        ProduceInFurnace(ItemType.REFRACTORY, 3),
        # ---- Phase C — starter factory components ----
        # 13 wires = 8 miners + 5 pallets. 2 frame + 2 circuit feed
        # the 2 extra assemblers.
        ProduceInAssembler(ItemType.WIRE, 13),
        ProduceInAssembler(ItemType.FRAME, 2),
        ProduceInAssembler(ItemType.CIRCUIT, 2),
        ProduceInAssembler(ItemType.MINER, 8),
        ProduceInAssembler(ItemType.PALLET, 5),
        ProduceInAssembler(ItemType.FURNACE, 2),
        ProduceInAssembler(ItemType.ASSEMBLER, 2),
        # ---- Phase D — deploy node auto-miners ----
        *_node_auto_miner(patch_x=8, patch_y_bottom=9),  # iron
        *_node_auto_miner(patch_x=23, patch_y_bottom=9),  # copper
        *_node_auto_miner(patch_x=23, patch_y_bottom=24),  # tin
        *_node_auto_miner(patch_x=15, patch_y_bottom=5),  # silicon
        *_coal_snake(),
        # ---- Phase E — central smelter/assembler bank near spawn ----
        # These support the PipelinedProduce bulk phases below.
        PlaceMachine(MachineType.FURNACE, free_tile_near_player()),
        PlaceMachine(MachineType.FURNACE, free_tile_near_player()),
        PlaceMachine(MachineType.ASSEMBLER, free_tile_near_player()),
        PlaceMachine(MachineType.ASSEMBLER, free_tile_near_player()),
        # ---- Phase F — let miners warm up ----
        WaitUntil(_miner_has_output_predicate(), max_ticks=30),
        # ---- Phase G — bulk mining ----
        # The node miners push ore into their pallets continuously,
        # but draining a pallet is still 1 WITHDRAW per tick, so
        # hand-mining in parallel is the reliable bulk source.
        MineOre(ItemType.IRON_ORE, 45),
        MineOre(ItemType.COPPER_ORE, 45),
        MineOre(ItemType.TIN_ORE, 42),
        MineOre(ItemType.SILICON, 20),
        # ---- Phase H — pipelined bulk smelts across 3 furnaces ----
        PipelinedProduce(ItemType.IRON_PLATE, 45, MachineType.FURNACE, k=3),
        PipelinedProduce(ItemType.COPPER_PLATE, 45, MachineType.FURNACE, k=3),
        PipelinedProduce(ItemType.TIN_PLATE, 42, MachineType.FURNACE, k=3),
        PipelinedProduce(ItemType.WAFER, 20, MachineType.FURNACE, k=3),
        # ---- Phase I — pipelined intermediates across 3 assemblers ----
        PipelinedProduce(ItemType.FRAME, 25, MachineType.ASSEMBLER, k=3),
        PipelinedProduce(ItemType.WIRE, 30, MachineType.ASSEMBLER, k=3),
        PipelinedProduce(ItemType.CIRCUIT, 18, MachineType.ASSEMBLER, k=3),
        PipelinedProduce(ItemType.MOTOR, 10, MachineType.ASSEMBLER, k=3),
        PipelinedProduce(ItemType.SENSOR, 10, MachineType.ASSEMBLER, k=3),
        # ---- Phase J — rocket sub-assemblies ----
        PipelinedProduce(ItemType.HULL, 6, MachineType.ASSEMBLER, k=3),
        PipelinedProduce(ItemType.ENGINE_UNIT, 4, MachineType.ASSEMBLER, k=3),
        PipelinedProduce(ItemType.AVIONICS, 4, MachineType.ASSEMBLER, k=3),
        PipelinedProduce(ItemType.ROCKET_CORE, 4, MachineType.ASSEMBLER, k=3),
        # ---- Phase K — final placeables (sequential to dodge debris) ----
        ProduceInAssembler(ItemType.CONVEYOR_BELT, 5),
        ProduceInAssembler(ItemType.ARM, 1),
        ProduceInAssembler(ItemType.ROCKET, 1),
        PlaceMachine(MachineType.CONVEYOR_BELT, free_tile_near_player()),
        PlaceMachine(MachineType.CONVEYOR_BELT, free_tile_near_player()),
        PlaceMachine(MachineType.CONVEYOR_BELT, free_tile_near_player()),
        PlaceMachine(MachineType.CONVEYOR_BELT, free_tile_near_player()),
        PlaceMachine(MachineType.CONVEYOR_BELT, free_tile_near_player()),
        PlaceMachine(MachineType.ARM, free_tile_near_player()),
        DepositInto(MachineType.PALLET, ItemType.IRON_PLATE),
        # ---- Phase L — capstone ----
        PlaceMachine(MachineType.ROCKET, free_tile_near_player()),
    ]


def make_factory_rocket_agent(env_params: EnvParams) -> ScriptedAgent:
    """Factory: scripted factory-building rocket agent."""
    return ScriptedAgent(env_params, Planner(build_factory_rocket_goals()))
