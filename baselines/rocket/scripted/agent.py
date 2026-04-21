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

    Budgets are provisioned for the full rocket chain with ~30% slack.
    Raw totals target:

    - Iron ore:  ~200 (→ ~64 iron plates once smelted)
    - Copper ore: ~140 (→ ~46 copper plates)
    - Tin ore:   ~60  (→ ~19 tin plates)
    - Silicon:   ~35  (→ ~11 wafers)
    - Coal:      ~210 (one per smelt)

    Scaled up to ensure CraftItem's forgiving semantics don't leave
    downstream recipes short on ingredients.
    """
    return [
        # ---- Phase B — raw ore ----
        MineOre(ItemType.IRON_ORE, 200),
        MineOre(ItemType.COPPER_ORE, 140),
        MineOre(ItemType.TIN_ORE, 60),
        MineOre(ItemType.COAL, 210),
        MineOre(ItemType.SILICON, 35),
        # ---- Phase C — bulk plate production via the furnace ----
        # Producing N>=1 of each plate fires the smelt_* achievements
        # on the first output. We go straight to bulk sizes sized for
        # the full rocket build so downstream recipes never block on
        # a missing ingredient.
        ProduceInFurnace(ItemType.IRON_PLATE, 64),
        ProduceInFurnace(ItemType.COPPER_PLATE, 46),
        ProduceInFurnace(ItemType.TIN_PLATE, 19),
        ProduceInFurnace(ItemType.WAFER, 11),
        # ---- Phase D — bulk intermediates via the assembler ----
        # Firing-in-bulk covers the craft_* achievements on the first
        # output of each recipe. Intentionally ordered so upstream
        # ingredients exist before downstream recipes run (frames before
        # motors, circuits/wires before sensors, etc.).
        ProduceInAssembler(ItemType.FRAME, 17),
        ProduceInAssembler(ItemType.WIRE, 17),
        ProduceInAssembler(ItemType.CIRCUIT, 10),
        ProduceInAssembler(ItemType.MOTOR, 8),
        ProduceInAssembler(ItemType.SENSOR, 7),
        # ---- Phase F — craft every placeable machine type ----
        ProduceInAssembler(ItemType.MINER, 3),  # craft_miner, scaling_up later
        ProduceInAssembler(ItemType.FURNACE, 1),  # craft_furnace
        ProduceInAssembler(ItemType.CONVEYOR_BELT, 5),  # craft_belt, belt_network later
        ProduceInAssembler(ItemType.PALLET, 1),  # craft_pallet
        ProduceInAssembler(ItemType.ARM, 1),  # craft_arm
        ProduceInAssembler(ItemType.ASSEMBLER, 1),  # craft_assembler
        ProduceInAssembler(ItemType.ROCKET, 1),  # craft_rocket
        # ---- Phase G — placements ----
        # 3 miners on distinct ore patches → place_miner, automated_mining,
        # scaling_up.
        PlaceMachine(MachineType.MINER, on_ore(ItemType.IRON_ORE)),
        PlaceMachine(MachineType.MINER, on_ore(ItemType.COPPER_ORE)),
        PlaceMachine(MachineType.MINER, on_ore(ItemType.TIN_ORE)),
        WaitUntil(_miner_has_output_predicate(), max_ticks=30),
        # Logistics placements and belt network.
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
        DepositInto(MachineType.PALLET, ItemType.IRON_ORE),
        # ---- Phase I — capstone: place the rocket ----
        PlaceMachine(MachineType.ROCKET, free_tile_near_player()),
    ]


def make_scripted_rocket_agent(env_params: EnvParams) -> ScriptedAgent:
    """Factory: scripted agent configured for the rocket benchmark."""
    return ScriptedAgent(env_params, Planner(build_rocket_goals()))
