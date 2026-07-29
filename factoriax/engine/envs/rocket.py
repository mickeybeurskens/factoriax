"""Rocket scenario — achievement-based reward signal with the rocket as capstone.

The scenario defines 38 achievements tiered Craftax-style (1/3/5/8 points,
max 140), ordered to teach the game's natural learning path: gather raw ore,
refine intermediates via a furnace and assembler, build and deploy the rest
of the factory, assemble the three rocket sub-components (hull, engine,
avionics), then launch the rocket.

Setup:

- A **furnace** and **assembler** are pre-placed adjacent to the player's
  spawn. The agent doesn't need to hand-craft the starter machinery — all
  crafting goes through these placed machines. Producing a *second*
  furnace/assembler via the main assembler is how the ``craft_furnace`` /
  ``craft_assembler`` achievements unlock.
- All direct player-crafting actions (``CRAFT_IRON_PLATE`` through
  ``CRAFT_ROCKET``) are **masked** — the env silently replaces them with
  ``NOOP``. Production must flow through the furnace / assembler pipeline.
- ``factoriax.make("Rocket-v1")`` (the :func:`rocket` factory) builds the env
  with the achievement conditions bound as a step hook and the hand-craft mask
  (:data:`ROCKET_BLOCKED_ACTIONS`) already applied.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

from factoriax.engine.achievements import (
    Achievement,
    achievement_fn,
    achievement_weights,
    any_assembler_has_output,
    any_buffer_nonempty,
    has_machines,
    holds_item,
    max_score,
    total_machines,
)
from factoriax.engine.constants import (
    Action,
    BlockType,
    Direction,
    ItemType,
    Machine,
)
from factoriax.engine.envs.base import FactoriaxEnv, achievement_hook
from factoriax.engine.envs.wrappers import ActionMaskWrapper
from factoriax.engine.levels import Level, LevelBuilder
from factoriax.engine.recipes import (
    BASE_RECIPE_BOOK,
    RecipeBalance,
    RecipeBook,
    RecipeOverride,
    RecipeTable,
)
from factoriax.engine.rewards import achievement_reward
from factoriax.engine.state import EnvParams, EnvState

# ---------------------------------------------------------------------------
# Achievement catalogue
# ---------------------------------------------------------------------------


def _ten_machines_placed(state: EnvState) -> jax.Array:
    """Ten machines of any type are on the map."""
    return total_machines(state) >= 10


#: The 38 bits, tiered Craftax-style: 10 Basic (1 pt), 11 Intermediate
#: (3 pt), 13 Advanced (5 pt), 4 Very Advanced (8 pt) — max score 140.
#: Ordered to teach the game's natural learning path.
ROCKET_ACHIEVEMENTS: tuple[Achievement, ...] = (
    # ---- Basic (1 pt) — raw gathering + simplest handcrafts.
    Achievement(
        "collect_iron",
        partial(holds_item, item=int(ItemType.IRON_ORE)),
        name="Collect Iron",
        hint="Mine iron ore.",
        weight=1.0,
    ),
    Achievement(
        "collect_copper",
        partial(holds_item, item=int(ItemType.COPPER_ORE)),
        name="Collect Copper",
        hint="Mine copper ore.",
        weight=1.0,
    ),
    Achievement(
        "collect_tin",
        partial(holds_item, item=int(ItemType.TIN_ORE)),
        name="Collect Tin",
        hint="Mine tin ore.",
        weight=1.0,
    ),
    Achievement(
        "collect_coal",
        partial(holds_item, item=int(ItemType.COAL)),
        name="Collect Coal",
        hint="Mine coal — needed by every smelt.",
        weight=1.0,
    ),
    Achievement(
        "collect_silicon",
        partial(holds_item, item=int(ItemType.SILICON)),
        name="Collect Silicon",
        hint="Mine silicon.",
        weight=1.0,
    ),
    Achievement(
        "smelt_iron",
        partial(holds_item, item=int(ItemType.IRON_PLATE)),
        name="Smelt Iron",
        hint="Hold an iron plate.",
        weight=1.0,
    ),
    Achievement(
        "smelt_copper",
        partial(holds_item, item=int(ItemType.COPPER_PLATE)),
        name="Smelt Copper",
        hint="Hold a copper plate.",
        weight=1.0,
    ),
    Achievement(
        "smelt_tin",
        partial(holds_item, item=int(ItemType.TIN_PLATE)),
        name="Smelt Tin",
        hint="Hold a tin plate.",
        weight=1.0,
    ),
    Achievement(
        "smelt_wafer",
        partial(holds_item, item=int(ItemType.WAFER)),
        name="Smelt Wafer",
        hint="Hold a wafer.",
        weight=1.0,
    ),
    Achievement(
        "craft_wire",
        partial(holds_item, item=int(ItemType.WIRE)),
        name="Craft Wire",
        hint="Combine iron and copper into wire.",
        weight=1.0,
    ),
    # ---- Intermediate (3 pt) — deeper handcrafts + first machines.
    Achievement(
        "craft_circuit",
        partial(holds_item, item=int(ItemType.CIRCUIT)),
        name="Craft Circuit",
        hint="Combine copper and wafer.",
        weight=3.0,
    ),
    Achievement(
        "craft_frame",
        partial(holds_item, item=int(ItemType.FRAME)),
        name="Craft Frame",
        hint="Combine iron and tin plates.",
        weight=3.0,
    ),
    Achievement(
        "craft_motor",
        partial(holds_item, item=int(ItemType.MOTOR)),
        name="Craft Motor",
        hint="Combine frame and wire.",
        weight=3.0,
    ),
    Achievement(
        "craft_sensor",
        partial(holds_item, item=int(ItemType.SENSOR)),
        name="Craft Sensor",
        hint="Combine circuit and wire.",
        weight=3.0,
    ),
    Achievement(
        "craft_miner",
        partial(holds_item, item=int(ItemType.MINER)),
        name="Craft Miner",
        hint="Hold a miner in inventory.",
        weight=3.0,
    ),
    Achievement(
        "place_miner",
        partial(has_machines, machine=int(Machine.MINER)),
        name="Place Miner",
        hint="Place a miner on the map.",
        weight=3.0,
    ),
    Achievement(
        "craft_furnace",
        partial(holds_item, item=int(ItemType.FURNACE)),
        name="Craft Furnace",
        hint="Hold a furnace in inventory.",
        weight=3.0,
    ),
    Achievement(
        "place_furnace",
        partial(has_machines, machine=int(Machine.FURNACE)),
        name="Place Furnace",
        hint="Place a furnace on the map.",
        weight=3.0,
    ),
    Achievement(
        "automated_mining",
        partial(any_buffer_nonempty, machine=int(Machine.MINER)),
        name="Automated Mining",
        hint="Have a placed miner produce ore.",
        weight=3.0,
    ),
    Achievement(
        "craft_belt",
        partial(holds_item, item=int(ItemType.CONVEYOR_BELT)),
        name="Craft Belt",
        hint="Hold a conveyor belt in inventory.",
        weight=3.0,
    ),
    Achievement(
        "place_belt",
        partial(has_machines, machine=int(Machine.CONVEYOR_BELT)),
        name="Place Belt",
        hint="Place a conveyor belt on the map.",
        weight=3.0,
    ),
    # ---- Advanced (5 pt) — logistics + rocket sub-components.
    Achievement(
        "craft_pallet",
        partial(holds_item, item=int(ItemType.PALLET)),
        name="Craft Pallet",
        hint="Hold a pallet in inventory.",
        weight=5.0,
    ),
    Achievement(
        "place_pallet",
        partial(has_machines, machine=int(Machine.PALLET)),
        name="Place Pallet",
        hint="Place a pallet on the map.",
        weight=5.0,
    ),
    Achievement(
        "pallet_filled",
        partial(any_buffer_nonempty, machine=int(Machine.PALLET)),
        name="Pallet Filled",
        hint="Put an item in a pallet.",
        weight=5.0,
    ),
    Achievement(
        "craft_arm",
        partial(holds_item, item=int(ItemType.ARM)),
        name="Craft Arm",
        hint="Hold an arm in inventory.",
        weight=5.0,
    ),
    Achievement(
        "place_arm",
        partial(has_machines, machine=int(Machine.ARM)),
        name="Place Arm",
        hint="Place an arm on the map.",
        weight=5.0,
    ),
    Achievement(
        "craft_assembler",
        partial(holds_item, item=int(ItemType.ASSEMBLER)),
        name="Craft Assembler",
        hint="Hold an assembler in inventory.",
        weight=5.0,
    ),
    Achievement(
        "place_assembler",
        partial(has_machines, machine=int(Machine.ASSEMBLER)),
        name="Place Assembler",
        hint="Place an assembler on the map.",
        weight=5.0,
    ),
    Achievement(
        "first_assembly",
        any_assembler_has_output,
        name="First Assembly",
        hint="Have a placed assembler produce output.",
        weight=5.0,
    ),
    Achievement(
        "belt_network",
        partial(has_machines, machine=int(Machine.CONVEYOR_BELT), count=5),
        name="Belt Network",
        hint="Place five conveyor belts.",
        weight=5.0,
    ),
    Achievement(
        "craft_hull",
        partial(holds_item, item=int(ItemType.HULL)),
        name="Craft Hull",
        hint="Assemble a rocket hull (2 frame + 2 iron plate).",
        weight=5.0,
    ),
    Achievement(
        "craft_engine_unit",
        partial(holds_item, item=int(ItemType.ENGINE_UNIT)),
        name="Craft Engine Unit",
        hint="Assemble a rocket engine (2 motor + 1 wire).",
        weight=5.0,
    ),
    Achievement(
        "craft_avionics",
        partial(holds_item, item=int(ItemType.AVIONICS)),
        name="Craft Avionics",
        hint="Assemble an avionics package (2 circuit + 2 sensor).",
        weight=5.0,
    ),
    Achievement(
        "craft_rocket_core",
        partial(holds_item, item=int(ItemType.ROCKET_CORE)),
        name="Craft Rocket Core",
        hint="Assemble a rocket core (engine + avionics).",
        weight=5.0,
    ),
    # ---- Very Advanced (8 pt) — scale-up and the rocket itself.
    Achievement(
        "scaling_up",
        partial(has_machines, machine=int(Machine.MINER), count=3),
        name="Scaling Up",
        hint="Place three miners at once.",
        weight=8.0,
    ),
    Achievement(
        "industrialist",
        _ten_machines_placed,
        name="Industrialist",
        hint="Place ten machines in total.",
        weight=8.0,
    ),
    Achievement(
        "craft_rocket",
        partial(holds_item, item=int(ItemType.ROCKET)),
        name="Craft Rocket",
        hint="Assemble a rocket into inventory.",
        weight=8.0,
    ),
    Achievement(
        "place_rocket",
        partial(has_machines, machine=int(Machine.ROCKET)),
        name="Place Rocket",
        hint="Place the rocket — goal reached.",
        weight=8.0,
    ),
)
NUM_ROCKET_ACHIEVEMENTS: int = len(ROCKET_ACHIEVEMENTS)

rocket_conditions = achievement_fn(ROCKET_ACHIEVEMENTS)

ROCKET_ACHIEVEMENT_WEIGHTS: jax.Array = achievement_weights(ROCKET_ACHIEVEMENTS)

MAX_ROCKET_SCORE: float = max_score(ROCKET_ACHIEVEMENTS)


# ---------------------------------------------------------------------------
# Reward function for training
# ---------------------------------------------------------------------------


def rocket_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Sparse reward for newly-unlocked rocket-scenario achievements.

    Thin wrapper around :func:`factoriax.engine.rewards.achievement_reward`
    bound to :data:`ROCKET_ACHIEVEMENT_WEIGHTS`. Assumes *prev_state*
    and *new_state* are :class:`~factoriax.engine.state.EnvState` instances
    whose ``achievements_unlocked`` field has been latched by the env's
    ``achievement_fn`` (typically :func:`rocket_conditions`).

    Parameters
    ----------
        prev_state: EnvState before the step.
        new_state: EnvState after the step.

    Parameters
    ----------
    prev_state : EnvState :

    new_state : EnvState :

    params : EnvParams :

    prev_state: EnvState :

    new_state: EnvState :

    params: EnvParams :


    Returns
    -------


    """
    return achievement_reward(
        prev_state, new_state, params, weights=ROCKET_ACHIEVEMENT_WEIGHTS
    )


# ---------------------------------------------------------------------------
# Recipe balance — rocket-specific tuning over BASE_RECIPE_BOOK
# ---------------------------------------------------------------------------

# Mass-production transport recipes are tuned up so a single craft
# cycle yields enough belts / splitters / crossings to wire a 4-cell
# factory without dominating the bootstrap phase. BASE outputs are 1
# per cycle for all three; the factory needs ~50–80 belts, 4 splitters,
# and 0–2 crossings (depending on layout) to wire the rocket chain, so
# 10 / 4 / 4 keeps the bootstrap craft list short.
ROCKET_RECIPE_BALANCE: RecipeBalance = RecipeBalance(
    overrides=(
        (int(ItemType.CONVEYOR_BELT), RecipeOverride(output_count=10)),
        (int(ItemType.SPLITTER), RecipeOverride(output_count=4)),
        (int(ItemType.CROSSING), RecipeOverride(output_count=4)),
    )
)

#: :class:`RecipeBook` used by the rocket scenario — applies
#: :data:`ROCKET_RECIPE_BALANCE` over :data:`BASE_RECIPE_BOOK`.
#: Pass to recipe-driven scripted agents so their BOM math tracks
#: the engine's recipe table.
ROCKET_RECIPE_BOOK: RecipeBook = BASE_RECIPE_BOOK.with_balance(ROCKET_RECIPE_BALANCE)

#: :class:`RecipeTable` projection of :data:`ROCKET_RECIPE_BOOK`.
#: Pass into :class:`~factoriax.engine.state.EnvParams` ``recipe_table`` so
#: the JIT'd engine produces the rebalanced output counts.
ROCKET_RECIPE_TABLE: RecipeTable = RecipeTable.from_book(ROCKET_RECIPE_BOOK)


# ---------------------------------------------------------------------------
# Level construction
# ---------------------------------------------------------------------------

# 32x32 map. Layout (the "streamlined" v2 geometry, see
# docs/rocket_scripted_agent.md):
#
# - Coal occupies the entire left column (x = 0, all 32 rows). Every
#   smelter cell pulls coal east along its own row — no vertical coal
#   trunks to navigate around.
# - The five ore patches sit on cols 3-4 in 2x2 squares, stacked
#   vertically with a 1-tile dirt gap between them (rows 9, 12, 15,
#   18, 21). The 2-tile dirt buffer at cols 1-2 keeps belts and arms
#   off the coal column.
# - Spawn at the map centre with a furnace / assembler pre-placed
#   immediately west / east. Hand-crafting is masked, so production
#   must flow through these two starter machines.
# - Limestone (rows 21-22) is unused for the rocket chain itself but
#   kept so the recipe book's REFRACTORY recipe stays satisfiable.
_MAP_SIZE: int = 32
_ORE_PATCH_SIZE: int = 2  # 2x2 ore squares
# 6300 per tile × 4 tiles per patch ≈ 25 000 ore per patch. Comparable
# to the old 9-tile patch budget, so demand-side recipe planning
# doesn't need to change.
_ORE_RESOURCES_PER_TILE: int = 6300
# Coal column is one tile wide × 32 tiles tall. 28 000 per tile means
# ~900 000 coal — comfortably more than the rocket chain consumes
# even if every smelter and every refractory craft fires worst-case.
# Requires ``BLOCK_MAX_RESOURCES`` to be at least 28 000.
_COAL_RESOURCES_PER_TILE: int = 28000
_COAL_COLUMN_X: int = 0
_SPAWN: tuple[int, int] = (_MAP_SIZE // 2, _MAP_SIZE // 2)
_FURNACE_TILE: tuple[int, int] = (_SPAWN[0] - 1, _SPAWN[1])
_ASSEMBLER_TILE: tuple[int, int] = (_SPAWN[0] + 1, _SPAWN[1])
_PATCH_OFFSETS: list[tuple[int, int, BlockType]] = [
    # (x, y, block) — top-left corner of the 2x2 patch.
    (3, 9, BlockType.IRON),
    (3, 12, BlockType.COPPER),
    (3, 15, BlockType.TIN),
    (3, 18, BlockType.SILICON),
    (3, 21, BlockType.LIMESTONE),
]


def build_rocket_level() -> Level:
    """Construct the canonical 32x32 rocket scenario level.

    Player spawns at :data:`_SPAWN`. Five 2x2 ore patches (iron,
    copper, tin, silicon, limestone) sit on cols 3-4, vertically
    stacked with 1-tile dirt gaps (rows 9, 12, 15, 18, 21). A 1-wide
    coal column fills the entire left edge (x = 0). Each ore tile
    carries :data:`_ORE_RESOURCES_PER_TILE` units; each coal tile
    carries :data:`_COAL_RESOURCES_PER_TILE`. A furnace and an
    assembler are pre-placed one tile west and east of spawn
    respectively.

    Parameters
    ----------

    Returns
    -------


    """
    builder = LevelBuilder(_MAP_SIZE, _MAP_SIZE)
    # Coal column — one tile wide, full map height.
    builder.fill_rect(
        _COAL_COLUMN_X,
        0,
        1,
        _MAP_SIZE,
        BlockType.COAL,
        resources=_COAL_RESOURCES_PER_TILE,
    )
    for x, y, block in _PATCH_OFFSETS:
        builder.fill_rect(
            x,
            y,
            _ORE_PATCH_SIZE,
            _ORE_PATCH_SIZE,
            block,
            resources=_ORE_RESOURCES_PER_TILE,
        )
    builder.set_player_position(*_SPAWN)
    builder.place_machine(
        _FURNACE_TILE[0],
        _FURNACE_TILE[1],
        int(Machine.FURNACE),
        direction=int(Direction.DOWN),
    )
    builder.place_machine(
        _ASSEMBLER_TILE[0],
        _ASSEMBLER_TILE[1],
        int(Machine.ASSEMBLER),
        direction=int(Direction.DOWN),
    )
    return builder.build("rocket_v1")


# ---------------------------------------------------------------------------
# Action mask
# ---------------------------------------------------------------------------

# The rocket scenario forbids all direct player crafting; production must flow
# through the pre-placed furnace / assembler. The mask is the whole CRAFT
# family, derived from the enum so new craftables are blocked automatically.
ROCKET_BLOCKED_ACTIONS: frozenset[int] = frozenset(
    int(a) for a in Action if a.name.startswith("CRAFT_")
)


# ---------------------------------------------------------------------------
# Scenario class
# ---------------------------------------------------------------------------


def rocket(
    *,
    obs: str = "x_ray_local",
    obs_radius: int = 5,
) -> tuple[Any, EnvParams]:
    """Build the canonical Rocket-v1 env with achievement hook and action mask applied.

    Parameters
    ----------
    obs :
        Observation variant passed to :class:`~factoriax.engine.envs.FactoriaxEnv`.
    obs_radius :
        Radius for local observation variants.

    Returns
    -------
    tuple
        ``(env, params)`` ready for gymnax-style rollouts.
    """
    env: Any = FactoriaxEnv(
        level=build_rocket_level(),
        step_hooks=(achievement_hook(rocket_conditions),),
        reward_fn=rocket_reward,
        obs=obs,
        obs_radius=obs_radius,
    )
    env = ActionMaskWrapper(env, tuple(ROCKET_BLOCKED_ACTIONS))
    params = EnvParams(
        max_timesteps=8000,
        recipe_table=ROCKET_RECIPE_TABLE,
    )
    return env, params
