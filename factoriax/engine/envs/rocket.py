"""The Rocket scenario: an achievement reward with the rocket as the last step.

The scenario defines 38 achievements in tiers, in the Craftax style, worth 1,
3, 5, and 8 points. The highest score is 140. The order teaches the natural
path of the game: gather raw ore, refine half-fabricates in a furnace and an
assembler, build and place the rest of the factory, assemble the three rocket
parts (the hull, the engine, and the avionics), then launch the rocket.

The setup:

- A **furnace** and an **assembler** stand next to the spawn tile of the
  player. The agent therefore does not craft the first machines by hand, and
  every craft goes through those two machines. The ``craft_furnace`` and
  ``craft_assembler`` achievements unlock when the main assembler produces a
  *second* furnace or assembler.
- The environment **masks** every direct player-craft action, from
  ``CRAFT_IRON_PLATE`` to ``CRAFT_ROCKET``, and replaces each one with
  ``NOOP``. It reports nothing. Production must flow through the furnace and
  the assembler.
- ``factoriax.make("Rocket-v1")``, which calls the :func:`rocket` factory,
  builds the environment with the achievement conditions as a step hook and
  the hand-craft mask :data:`ROCKET_BLOCKED_ACTIONS` already applied.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import jax

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
    """Test whether the map holds ten machines or more, of any type."""
    return total_machines(state) >= 10


#: The 38 bits, in tiers in the Craftax style: 10 Basic bits worth 1 point, 11
#: Intermediate bits worth 3 points, 13 Advanced bits worth 5 points, and 4
#: Very Advanced bits worth 8 points. The highest score is 140. The order
#: teaches the natural path of the game.
ROCKET_ACHIEVEMENTS: tuple[Achievement, ...] = (
    # ---- Basic, 1 point: raw gathering and the simplest hand crafts.
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
    # ---- Intermediate, 3 points: deeper crafts and the first machines.
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
    # ---- Advanced, 5 points: logistics and the rocket parts.
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
    # ---- Very Advanced, 8 points: a larger factory and the rocket itself.
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
# The reward function for training
# ---------------------------------------------------------------------------


def rocket_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Sparse reward for the rocket-scenario achievements that a step unlocked.

    This function is a short wrapper around
    :func:`factoriax.engine.rewards.achievement_reward`, bound to
    :data:`ROCKET_ACHIEVEMENT_WEIGHTS`. It expects two
    :class:`~factoriax.engine.state.EnvState` objects whose
    ``achievements_unlocked`` field the ``achievement_fn`` of the environment
    already latched. That function is usually :func:`rocket_conditions`.

    Parameters
    ----------
    prev_state
        State just before the step.
    new_state
        State just after the step.
    params
        The function does not read this argument. It is present for the shared
        reward signature.

    Returns
    -------
    jax.Array
        Scalar float32, the weighted sum of the achievements that this step
        unlocked. The value is zero on most steps.
    """
    return achievement_reward(
        prev_state, new_state, params, weights=ROCKET_ACHIEVEMENT_WEIGHTS
    )


# ---------------------------------------------------------------------------
# Recipe balance: the rocket tuning over BASE_RECIPE_BOOK
# ---------------------------------------------------------------------------

# The transport recipes give a larger output here, so one craft cycle produces
# enough belts, splitters, and crossings to connect a factory of four cells.
# The bootstrap phase then stays short. The BASE output is 1 for each of the
# three. The factory needs about 50 to 80 belts, 4 splitters, and 0 to 2
# crossings, which depends on the layout. Outputs of 10, 4, and 4 therefore
# keep the bootstrap craft list short.
ROCKET_RECIPE_BALANCE: RecipeBalance = RecipeBalance(
    overrides=(
        (int(ItemType.CONVEYOR_BELT), RecipeOverride(output_count=10)),
        (int(ItemType.SPLITTER), RecipeOverride(output_count=4)),
        (int(ItemType.CROSSING), RecipeOverride(output_count=4)),
    )
)

#: The :class:`RecipeBook` of the rocket scenario. It applies
#: :data:`ROCKET_RECIPE_BALANCE` over :data:`BASE_RECIPE_BOOK`. Pass it to a
#: scripted agent that reads recipes, so the material totals of that agent
#: match the recipe table of the engine.
ROCKET_RECIPE_BOOK: RecipeBook = BASE_RECIPE_BOOK.with_balance(ROCKET_RECIPE_BALANCE)

#: The :class:`RecipeTable` of :data:`ROCKET_RECIPE_BOOK`. Pass it as the
#: ``recipe_table`` of :class:`~factoriax.engine.state.EnvParams`, so the JIT'd
#: engine produces the new output counts.
ROCKET_RECIPE_TABLE: RecipeTable = RecipeTable.from_book(ROCKET_RECIPE_BOOK)


# ---------------------------------------------------------------------------
# Level construction
# ---------------------------------------------------------------------------

# A 32x32 map. The layout is the "streamlined" v2 geometry. See
# docs/rocket_scripted_agent.md.
#
# - Coal fills the whole left column, at x = 0, over all 32 rows. Every smelter
#   cell pulls coal east along its own row, and no vertical coal line stands in
#   the way.
# - The five ore patches sit on columns 3 and 4, as 2x2 squares, one above the
#   other with a one-tile dirt gap between them, at rows 9, 12, 15, 18, and 21.
#   The two-tile dirt gap at columns 1 and 2 keeps the belts and the arms off
#   the coal column.
# - The player spawns at the centre of the map, with a furnace to the west and
#   an assembler to the east. The mask blocks every hand craft, so production
#   must flow through those two machines.
# - The rocket chain does not use the limestone at rows 21 and 22. It stays so
#   that the REFRACTORY recipe of the recipe book still has its input.
_MAP_SIZE: int = 32
_ORE_PATCH_SIZE: int = 2  # 2x2 ore squares
# 6300 for each tile, over 4 tiles in a patch, is about 25 000 ore in a patch.
# That is close to the old budget of a 9-tile patch, so the demand side of the
# recipe planning needs no change.
_ORE_RESOURCES_PER_TILE: int = 6300
# The coal column is one tile wide and 32 tiles tall. 28 000 for each tile
# gives about 900 000 coal. That is far more than the rocket chain consumes,
# even when every smelter and every refractory craft runs at its highest rate.
# ``BLOCK_MAX_RESOURCES`` must be 28 000 or more.
_COAL_RESOURCES_PER_TILE: int = 28000
_COAL_COLUMN_X: int = 0
_SPAWN: tuple[int, int] = (_MAP_SIZE // 2, _MAP_SIZE // 2)
_FURNACE_TILE: tuple[int, int] = (_SPAWN[0] - 1, _SPAWN[1])
_ASSEMBLER_TILE: tuple[int, int] = (_SPAWN[0] + 1, _SPAWN[1])
_PATCH_OFFSETS: list[tuple[int, int, BlockType]] = [
    # (x, y, block), where x and y are the top-left corner of the 2x2 patch.
    (3, 9, BlockType.IRON),
    (3, 12, BlockType.COPPER),
    (3, 15, BlockType.TIN),
    (3, 18, BlockType.SILICON),
    (3, 21, BlockType.LIMESTONE),
]


def build_rocket_level() -> Level:
    """Build the 32x32 level of the rocket scenario.

    The player spawns at :data:`_SPAWN`. Five 2x2 ore patches, of iron, copper,
    tin, silicon, and limestone, sit on columns 3 and 4. They stand one above
    the other with one-tile dirt gaps, at rows 9, 12, 15, 18, and 21. A coal
    column one tile wide fills the whole left edge, at x = 0. Each ore tile
    holds :data:`_ORE_RESOURCES_PER_TILE` units, and each coal tile holds
    :data:`_COAL_RESOURCES_PER_TILE`. A furnace stands one tile west of the
    spawn, and an assembler stands one tile east of it.

    The function takes no arguments and no PRNG key. The layout is fixed, so
    every reset of this scenario gives the same world.

    Returns
    -------
    Level
        The level of the rocket scenario, ready for
        :func:`factoriax.engine.levels.build_state`.
    """
    builder = LevelBuilder(_MAP_SIZE, _MAP_SIZE)
    # The coal column: one tile wide, over the full height of the map.
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

# The rocket scenario blocks every direct player craft. Production must flow
# through the furnace and the assembler that already stand on the map. The mask
# covers every CRAFT_ action and comes from the enum, so it also blocks a new
# craftable item with no further change.
ROCKET_BLOCKED_ACTIONS: frozenset[int] = frozenset(
    int(a) for a in Action if a.name.startswith("CRAFT_")
)


# ---------------------------------------------------------------------------
# The scenario factory
# ---------------------------------------------------------------------------


def rocket(
    *,
    obs: str = "x_ray_local",
    obs_radius: int = 5,
) -> tuple[Any, EnvParams]:
    """Build the Rocket-v1 environment, with its achievement hook and action mask.

    Parameters
    ----------
    obs :
        Observation variant. The function passes it to
        :class:`~factoriax.engine.envs.FactoriaxEnv`.
    obs_radius :
        Radius for a local observation variant.

    Returns
    -------
    tuple
        ``(env, params)``, ready for a gymnax-style rollout.
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
