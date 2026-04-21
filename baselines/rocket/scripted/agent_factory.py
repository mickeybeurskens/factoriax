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


# Coal now works like any other node: miner + pallet south of the
# patch. Earlier designs chained 4 miners into a snake, but coal
# demand (1 per smelt, ~200 per rocket) doesn't benefit from that
# shape — a single miner + 2500-resource patch covers the whole
# episode.


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
        # 5 miners + 5 pallets + 2 extra furnaces + 2 extra assemblers
        # needed as starter infrastructure. Plate totals: iron 9,
        # copper 12, tin 17, wafer 2, refractory 2. Coal for smelts:
        # 9+12+17+2 plates + 2 refractory = 42 coal. Plus slack.
        MineOre(ItemType.IRON_ORE, 11),
        MineOre(ItemType.COPPER_ORE, 14),
        MineOre(ItemType.TIN_ORE, 20),
        MineOre(ItemType.SILICON, 3),
        MineOre(ItemType.COAL, 50),
        # ---- Phase B — starter smelts via pre-placed furnace ----
        ProduceInFurnace(ItemType.IRON_PLATE, 11),
        ProduceInFurnace(ItemType.COPPER_PLATE, 14),
        ProduceInFurnace(ItemType.TIN_PLATE, 20),
        ProduceInFurnace(ItemType.WAFER, 3),
        ProduceInFurnace(ItemType.REFRACTORY, 2),
        # ---- Phase C — starter factory components ----
        # 10 wires = 5 miners + 5 pallets. 2 frame + 2 circuit feed
        # the 2 extra assemblers.
        ProduceInAssembler(ItemType.WIRE, 10),
        ProduceInAssembler(ItemType.FRAME, 2),
        ProduceInAssembler(ItemType.CIRCUIT, 2),
        ProduceInAssembler(ItemType.MINER, 5),
        ProduceInAssembler(ItemType.PALLET, 5),
        ProduceInAssembler(ItemType.FURNACE, 2),
        ProduceInAssembler(ItemType.ASSEMBLER, 2),
        # ---- Phase D — deploy node auto-miners (one per ore type) ----
        # Coal is just another node now — no snake; its 2500-resource
        # patch sustains the ~200-coal-per-rocket demand on its own.
        *_node_auto_miner(patch_x=8, patch_y_bottom=9),  # iron
        *_node_auto_miner(patch_x=23, patch_y_bottom=9),  # copper
        *_node_auto_miner(patch_x=23, patch_y_bottom=24),  # tin
        *_node_auto_miner(patch_x=15, patch_y_bottom=5),  # silicon
        *_node_auto_miner(patch_x=8, patch_y_bottom=24),  # coal
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
        # Bulk coal needs: 45+45+42+20 = 152 plate smelts → 152 coal.
        # Plus slack.
        MineOre(ItemType.IRON_ORE, 45),
        MineOre(ItemType.COPPER_ORE, 45),
        MineOre(ItemType.TIN_ORE, 42),
        MineOre(ItemType.SILICON, 20),
        MineOre(ItemType.COAL, 155),
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
