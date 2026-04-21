"""Top-level scripted rocket agent.

Wires :class:`Planner` to the observation decoder. Callers hand the
agent a global-observation vector each tick and receive an action.

Usage::

    agent = ScriptedAgent(env_params, planner)
    obs = env.get_obs(...)
    while not agent.is_done:
        action = agent.act(obs)
        obs, state, _, _, _ = env.step_env(key, state, action, env_params)
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from factoriax.constants import Action, ItemType, MachineType
from factoriax.state import EnvParams

from .goals import (
    DepositInto,
    Goal,
    MineOre,
    PlaceMachine,
    ProduceInAssembler,
    ProduceInFurnace,
    WaitUntil,
    free_tile_near_player,
    on_ore,
)
from .planner import Planner
from .skills import Result
from .world_model import WorldView, decode_observation


class ScriptedAgent:
    """Hand-authored rocket-benchmark agent."""

    def __init__(self, env_params: EnvParams, planner: Planner) -> None:
        self.env_params = env_params
        self.planner = planner
        self._last_view: WorldView | None = None

    @property
    def is_done(self) -> bool:
        return self.planner.is_done

    def act(self, obs: np.ndarray) -> int:
        view = decode_observation(
            np.asarray(obs),
            map_height=self.env_params.map_height,
            map_width=self.env_params.map_width,
            max_timesteps=self.env_params.max_timesteps,
        )
        self._last_view = view
        result, action = self.planner.step(view)
        if result is Result.DONE or action is None:
            return int(Action.NOOP)
        return int(action)


# ---------------------------------------------------------------------------
# Rocket-benchmark phase list
# ---------------------------------------------------------------------------


def _miner_has_output_predicate() -> Callable[[WorldView], bool]:
    """Predicate: True when any placed miner has buffered ore."""

    def pred(view: WorldView) -> bool:
        # buffer_type > 0 on any MINER tile — rocket conditions match
        # the miner's ent_buf_count, which the obs surfaces as the
        # buffer_type channel via the tile→entity scatter.
        miner_tiles = view.tiles_with_machine(MachineType.MINER)
        for x, y in miner_tiles:
            if view.buffer_type[y, x] != 0:
                return True
        return False

    return pred


def _any_assembler_has_output_predicate() -> Callable[[WorldView], bool]:
    """Predicate: any assembler shows a non-zero output buffer.

    The ``buffer_type`` obs channel is shared between asm inputs and
    outputs at the tile level, so any non-zero value on an ASSEMBLER
    tile is a sufficient indicator that something has been consumed
    (rocket_conditions checks `ent_asm_out_count > 0` directly in
    state, but the obs can't distinguish inputs from outputs — we
    just wait and trust the recipe timer).
    """

    def pred(view: WorldView) -> bool:
        asm_tiles = view.tiles_with_machine(MachineType.ASSEMBLER)
        for x, y in asm_tiles:
            if view.buffer_type[y, x] != 0:
                return True
        return False

    return pred


def build_rocket_goals() -> list[Goal]:
    """Full goal list for completing every rocket-benchmark achievement.

    Uses only the pre-placed furnace + assembler (plus mining for raw
    ore + placements for the rest of the factory). Hand-crafting is
    masked by the benchmark, so every intermediate flows through
    :func:`ProduceInFurnace` / :func:`ProduceInAssembler`.

    Budget derivation (rocket + every placeable machine achievement):

    - Frames: 20 (rocket) + 1 (assembler) = 21. Plan: 25.
    - Wires:  20 (rocket) + 3 (miners) + 1 (pallet) + 1 (arm) = 25.
      Plan: 30.
    - Circuits: 16 (rocket) + 1 (assembler) = 17. Plan: 20.
    - Motors / sensors: 8 each for rocket. Plan: 10.
    - Plates back out to 1:1 smelts:
        iron 42, tin 47, copper 55 (+slack), wafer 20.
    - Coal: 1 (refractory for the one FURNACE machine achievement).
    """
    return [
        # ---- Phase B — raw ore ----
        # Each smelt now consumes 1 coal in addition to the ore, so
        # coal demand = sum of plate counts + 1 refractory. Plus
        # slack.
        MineOre(ItemType.IRON_ORE, 55),
        MineOre(ItemType.COPPER_ORE, 60),
        MineOre(ItemType.TIN_ORE, 60),
        MineOre(ItemType.COAL, 205),
        MineOre(ItemType.SILICON, 22),
        # ---- Phase C — bulk plate production via the furnace ----
        # Smelts now consume 1 coal per plate (ore + coal → plate).
        ProduceInFurnace(ItemType.IRON_PLATE, 55),
        ProduceInFurnace(ItemType.COPPER_PLATE, 60),
        ProduceInFurnace(ItemType.TIN_PLATE, 60),
        ProduceInFurnace(ItemType.WAFER, 22),
        # REFRACTORY (coal only) — still a 1-input recipe.
        ProduceInFurnace(ItemType.REFRACTORY, 1),
        # ---- Phase D — bulk intermediates via the assembler ----
        # Ordered so upstream ingredients exist before downstream
        # recipes run (frames before motors, wires before sensors …).
        ProduceInAssembler(ItemType.FRAME, 25),
        ProduceInAssembler(ItemType.WIRE, 32),
        ProduceInAssembler(ItemType.CIRCUIT, 20),
        ProduceInAssembler(ItemType.MOTOR, 10),
        ProduceInAssembler(ItemType.SENSOR, 10),
        # ---- Phase E — rocket sub-assemblies ----
        ProduceInAssembler(ItemType.HULL, 6),
        ProduceInAssembler(ItemType.ENGINE_UNIT, 4),
        ProduceInAssembler(ItemType.AVIONICS, 4),
        ProduceInAssembler(ItemType.ROCKET_CORE, 4),
        # ---- Phase F — every placeable machine type (achievement fill) ----
        ProduceInAssembler(ItemType.MINER, 3),
        ProduceInAssembler(ItemType.FURNACE, 1),
        ProduceInAssembler(ItemType.CONVEYOR_BELT, 5),
        ProduceInAssembler(ItemType.PALLET, 1),
        ProduceInAssembler(ItemType.ARM, 1),
        ProduceInAssembler(ItemType.ASSEMBLER, 1),
        ProduceInAssembler(ItemType.ROCKET, 1),
        # ---- Phase G — placements ----
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
        PlaceMachine(MachineType.ASSEMBLER, free_tile_near_player()),
        # ---- Phase H — unlock pallet_filled ----
        # Deposit a leftover plate (raw ore is all smelted by now).
        DepositInto(MachineType.PALLET, ItemType.IRON_PLATE),
        # ---- Phase I — capstone: place the rocket ----
        PlaceMachine(MachineType.ROCKET, free_tile_near_player()),
    ]


def make_scripted_rocket_agent(env_params: EnvParams) -> ScriptedAgent:
    """Factory: scripted agent configured for the rocket benchmark."""
    return ScriptedAgent(env_params, Planner(build_rocket_goals()))
