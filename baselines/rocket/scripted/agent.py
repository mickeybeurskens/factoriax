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
    CraftItem,
    DepositInto,
    Goal,
    MineOre,
    PlaceMachine,
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
    """The full goal list for completing every rocket-benchmark achievement.

    Roughly matches Phases B–H from the design doc. Counts are
    generous so a single depleted patch or a failed craft doesn't
    derail the whole plan.
    """
    return [
        # ---- Phase B — raw ore ----
        # Budgets are generous (~2x measured need) so every downstream
        # CraftItem has enough pool to complete its target.
        MineOre(ItemType.IRON_ORE, 260),
        MineOre(ItemType.COPPER_ORE, 220),
        MineOre(ItemType.TIN_ORE, 90),
        MineOre(ItemType.COAL, 300),
        MineOre(ItemType.SILICON, 45),
        # ---- Phase C — basic handcrafts (Basic achievements) ----
        CraftItem(ItemType.IRON_PLATE, 1),  # smelt_iron
        CraftItem(ItemType.COPPER_PLATE, 1),  # smelt_copper
        CraftItem(ItemType.TIN_PLATE, 1),  # smelt_tin
        CraftItem(ItemType.WAFER, 1),  # smelt_wafer
        CraftItem(ItemType.WIRE, 1),  # craft_wire
        # ---- Phase D — deeper intermediates (Intermediate crafts) ----
        CraftItem(ItemType.CIRCUIT, 1),  # craft_circuit
        CraftItem(ItemType.FRAME, 1),  # craft_frame
        CraftItem(ItemType.MOTOR, 1),  # craft_motor
        CraftItem(ItemType.SENSOR, 1),  # craft_sensor
        # ---- Bulk production — stock up for the full build ----
        # Over-provisioned: every downstream recipe (frames, wires, motors…)
        # draws from these pools, so a tight budget breaks the chain.
        CraftItem(ItemType.IRON_PLATE, 120),
        CraftItem(ItemType.COPPER_PLATE, 100),
        CraftItem(ItemType.TIN_PLATE, 40),
        CraftItem(ItemType.WAFER, 20),
        CraftItem(ItemType.WIRE, 30),
        CraftItem(ItemType.CIRCUIT, 20),
        CraftItem(ItemType.FRAME, 30),
        CraftItem(ItemType.MOTOR, 15),
        CraftItem(ItemType.SENSOR, 12),
        # ---- Craft every placeable machine type ----
        CraftItem(ItemType.MINER, 3),  # craft_miner; need 3 for scaling_up
        CraftItem(ItemType.FURNACE, 1),  # craft_furnace
        CraftItem(ItemType.CONVEYOR_BELT, 5),  # craft_belt; 5 for belt_network
        CraftItem(ItemType.PALLET, 1),  # craft_pallet
        CraftItem(ItemType.ARM, 1),  # craft_arm
        CraftItem(ItemType.ASSEMBLER, 1),  # craft_assembler
        CraftItem(ItemType.ROCKET, 1),  # craft_rocket
        # ---- Place three miners (first on iron, then on copper + tin
        # ore patches for diversity) → place_miner, automated_mining,
        # scaling_up when total reaches 3.
        PlaceMachine(MachineType.MINER, on_ore(ItemType.IRON_ORE)),
        PlaceMachine(MachineType.MINER, on_ore(ItemType.COPPER_ORE)),
        PlaceMachine(MachineType.MINER, on_ore(ItemType.TIN_ORE)),
        WaitUntil(_miner_has_output_predicate(), max_ticks=30),
        # ---- Other placements ----
        PlaceMachine(MachineType.FURNACE, free_tile_near_player()),
        PlaceMachine(MachineType.CONVEYOR_BELT, free_tile_near_player()),
        PlaceMachine(MachineType.CONVEYOR_BELT, free_tile_near_player()),
        PlaceMachine(MachineType.CONVEYOR_BELT, free_tile_near_player()),
        PlaceMachine(MachineType.CONVEYOR_BELT, free_tile_near_player()),
        PlaceMachine(MachineType.CONVEYOR_BELT, free_tile_near_player()),
        PlaceMachine(MachineType.PALLET, free_tile_near_player()),
        PlaceMachine(MachineType.ARM, free_tile_near_player()),
        PlaceMachine(MachineType.ASSEMBLER, free_tile_near_player()),
        # ---- Load the pallet to unlock pallet_filled ----
        DepositInto(MachineType.PALLET, ItemType.IRON_ORE),
        # ---- Load the assembler with WIRE inputs; wait for output.
        # WIRE recipe: 1 iron plate + 2 copper plate, 4 ticks.
        DepositInto(MachineType.ASSEMBLER, ItemType.IRON_PLATE),
        DepositInto(MachineType.ASSEMBLER, ItemType.COPPER_PLATE),
        DepositInto(MachineType.ASSEMBLER, ItemType.COPPER_PLATE),
        WaitUntil(_any_assembler_has_output_predicate(), max_ticks=20),
        # ---- Capstone ----
        PlaceMachine(MachineType.ROCKET, free_tile_near_player()),
    ]


def make_scripted_rocket_agent(env_params: EnvParams) -> ScriptedAgent:
    """Factory: scripted agent configured for the rocket benchmark."""
    return ScriptedAgent(env_params, Planner(build_rocket_goals()))
