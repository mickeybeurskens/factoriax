"""Factory-oriented variant of the scripted rocket agent.

Contrasts with :mod:`baselines.rocket.scripted.agent` (the "naive"
agent that runs everything serially through the pre-placed furnace
+ assembler). This variant:

1. Spends a small starter phase to hand-mine just enough ore to
   build 2 extra furnaces + 2 extra assemblers via the pre-placed
   starter machines.
2. Deploys the extra machines near spawn, giving the map 3 active
   furnaces and 3 active assemblers.
3. Uses :class:`PipelinedProduce` to rotate bulk smelting and
   intermediate assembly across the 3-machine batteries, keeping
   each one running while the agent services the others.
4. Continues the rocket sub-assembly chain at one assembler while
   the other two produce components in parallel.
5. Wraps up with the same achievement-fill placements (miner, belt,
   pallet, arm) the naive agent uses.

Composed entirely from existing skills — no engine changes. The
only new piece is :class:`PipelinedProduce` in
:mod:`baselines.rocket.scripted.goals`.
"""

from __future__ import annotations

from factoriax.constants import ItemType, MachineType
from factoriax.state import EnvParams

from .agent import (
    ScriptedAgent,
    _any_assembler_has_output_predicate,
    _miner_has_output_predicate,
)
from .goals import (
    DepositInto,
    Goal,
    MineOre,
    PipelinedProduce,
    PlaceMachine,
    ProduceInAssembler,
    ProduceInFurnace,
    WaitUntil,
    free_tile_near_player,
    on_ore,
)
from .planner import Planner


def build_factory_rocket_goals() -> list[Goal]:
    """Goal list for the factory-building variant.

    Starter budget pays for 2 extra furnaces (2 iron + 2 refractory)
    and 2 extra assemblers (2 frame + 2 circuit = 2 iron + 2 tin +
    2 copper + 2 wafer). Plus slack for starter wires/frames used up
    before the factory is online.

    Bulk targets match the naive agent (see
    :func:`baselines.rocket.scripted.agent.build_rocket_goals` for
    the derivation). The speedup — if any — comes from keeping 3
    furnaces and 3 assemblers running simultaneously instead of one
    at a time.
    """
    _ = _any_assembler_has_output_predicate  # keep import symmetrical
    return [
        # ---- Phase A — starter mining ----
        # Enough for 2 extra furnaces + 2 extra assemblers + slack.
        MineOre(ItemType.IRON_ORE, 8),
        MineOre(ItemType.COPPER_ORE, 5),
        MineOre(ItemType.TIN_ORE, 5),
        MineOre(ItemType.COAL, 3),
        MineOre(ItemType.SILICON, 3),
        # ---- Phase B — starter smelts via pre-placed furnace ----
        ProduceInFurnace(ItemType.IRON_PLATE, 8),
        ProduceInFurnace(ItemType.COPPER_PLATE, 5),
        ProduceInFurnace(ItemType.TIN_PLATE, 5),
        ProduceInFurnace(ItemType.WAFER, 3),
        ProduceInFurnace(ItemType.REFRACTORY, 3),
        # ---- Phase C — starter factory components ----
        # 2 extra furnaces + 2 extra assemblers.
        ProduceInAssembler(ItemType.WIRE, 2),  # buffer wire
        ProduceInAssembler(ItemType.FRAME, 2),  # for 2 assemblers
        ProduceInAssembler(ItemType.CIRCUIT, 2),  # for 2 assemblers
        ProduceInAssembler(ItemType.FURNACE, 2),
        ProduceInAssembler(ItemType.ASSEMBLER, 2),
        # ---- Phase D — deploy extra machines near spawn ----
        PlaceMachine(MachineType.FURNACE, free_tile_near_player()),
        PlaceMachine(MachineType.FURNACE, free_tile_near_player()),
        PlaceMachine(MachineType.ASSEMBLER, free_tile_near_player()),
        PlaceMachine(MachineType.ASSEMBLER, free_tile_near_player()),
        # ---- Phase E — bulk mining (parallel smelting pays off now) ----
        MineOre(ItemType.IRON_ORE, 55),
        MineOre(ItemType.COPPER_ORE, 60),
        MineOre(ItemType.TIN_ORE, 60),
        MineOre(ItemType.COAL, 5),
        MineOre(ItemType.SILICON, 22),
        # ---- Phase F — pipelined bulk smelting across 3 furnaces ----
        PipelinedProduce(ItemType.IRON_PLATE, 50, MachineType.FURNACE, k=3),
        PipelinedProduce(ItemType.COPPER_PLATE, 55, MachineType.FURNACE, k=3),
        PipelinedProduce(ItemType.TIN_PLATE, 55, MachineType.FURNACE, k=3),
        PipelinedProduce(ItemType.WAFER, 19, MachineType.FURNACE, k=3),
        # ---- Phase G — pipelined intermediates across 3 assemblers ----
        PipelinedProduce(ItemType.FRAME, 23, MachineType.ASSEMBLER, k=3),
        PipelinedProduce(ItemType.WIRE, 30, MachineType.ASSEMBLER, k=3),
        PipelinedProduce(ItemType.CIRCUIT, 18, MachineType.ASSEMBLER, k=3),
        PipelinedProduce(ItemType.MOTOR, 10, MachineType.ASSEMBLER, k=3),
        PipelinedProduce(ItemType.SENSOR, 10, MachineType.ASSEMBLER, k=3),
        # ---- Phase H — rocket sub-assemblies (pipelined) ----
        PipelinedProduce(ItemType.HULL, 6, MachineType.ASSEMBLER, k=3),
        PipelinedProduce(ItemType.ENGINE_UNIT, 4, MachineType.ASSEMBLER, k=3),
        PipelinedProduce(ItemType.AVIONICS, 4, MachineType.ASSEMBLER, k=3),
        PipelinedProduce(ItemType.ROCKET_CORE, 4, MachineType.ASSEMBLER, k=3),
        # ---- Phase I — final placeables for achievements ----
        ProduceInAssembler(ItemType.MINER, 3),
        ProduceInAssembler(ItemType.FURNACE, 1),
        ProduceInAssembler(ItemType.CONVEYOR_BELT, 5),
        ProduceInAssembler(ItemType.PALLET, 1),
        ProduceInAssembler(ItemType.ARM, 1),
        ProduceInAssembler(ItemType.ROCKET, 1),
        PlaceMachine(MachineType.MINER, on_ore(ItemType.IRON_ORE)),
        PlaceMachine(MachineType.MINER, on_ore(ItemType.COPPER_ORE)),
        PlaceMachine(MachineType.MINER, on_ore(ItemType.TIN_ORE)),
        WaitUntil(_miner_has_output_predicate(), max_ticks=30),
        PlaceMachine(MachineType.FURNACE, free_tile_near_player()),
        PlaceMachine(MachineType.CONVEYOR_BELT, free_tile_near_player()),
        PlaceMachine(MachineType.CONVEYOR_BELT, free_tile_near_player()),
        PlaceMachine(MachineType.CONVEYOR_BELT, free_tile_near_player()),
        PlaceMachine(MachineType.CONVEYOR_BELT, free_tile_near_player()),
        PlaceMachine(MachineType.CONVEYOR_BELT, free_tile_near_player()),
        PlaceMachine(MachineType.PALLET, free_tile_near_player()),
        PlaceMachine(MachineType.ARM, free_tile_near_player()),
        DepositInto(MachineType.PALLET, ItemType.IRON_PLATE),
        # ---- Phase J — capstone ----
        PlaceMachine(MachineType.ROCKET, free_tile_near_player()),
    ]


def make_factory_rocket_agent(env_params: EnvParams) -> ScriptedAgent:
    """Factory: scripted factory-building rocket agent."""
    return ScriptedAgent(env_params, Planner(build_factory_rocket_goals()))
