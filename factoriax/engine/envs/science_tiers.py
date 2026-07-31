"""ScienceTiers-v1: a throughput reward over a science economy of three tiers.

This scenario has one currency. A lab consumes a science pack of any tier, and
every pack pays a reward of 1 on the tick where the lab consumes it. There are
no achievement bits and no per-tier reward weights. The tier ladder lives in
the recipe book alone. There, the ore cost doubles with each tier while the
pack output goes up by a factor of four:

======  ==============================  ========  ======  ===========
tier    recipe                          ore cost  output  science/ore
======  ==============================  ========  ======  ===========
1       coal + limestone                2         1       0.5
2       tier-1 pack + wire              4         4       1.0
3       tier-2 pack + motor             8         16      2.0
======  ==============================  ========  ======  ===========

A wire is iron plus tin ore. A motor is a wire plus a frame. A frame is copper
plus silicon ore. Every half-fabricate therefore ends in raw ore, and the ore
costs in the table above are exact.

Each tier consumes the pack of the tier below it. To climb, an agent must give
up packs that it can spend at once. The steady throughput for each unit of ore
doubles with each tier, but the return arrives only after the longer chain
runs. That tension is the research target: an agent that stops at a lower tier
against an agent that invests through it.

The world is the shared 16x16 six-patch map, with one 2x2 patch for each ore.
One science lab stands on the tile that the player faces at the start, so an
agent can find the consume-for-reward loop and does not first have to work out
that labs exist. An agent can craft more labs, miners, assemblers, belts, and
arms from raw ore.

The observation defaults match the miner curriculum: the egocentric
``superficial_local`` with radius 7. A policy network therefore transfers
between the two.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from factoriax.engine.constants import ItemType
from factoriax.engine.envs.base import FactoriaxEnv
from factoriax.engine.envs.common import (
    MAP_SIZE,
    ORE_RESOURCES_PER_TILE,
    six_patch_terrain,
)
from factoriax.engine.placement import place_machine
from factoriax.engine.recipes import Recipe, RecipeBook, RecipeTable
from factoriax.engine.state import EnvParams, EnvState

#: Length of an episode. It is longer than the 300 steps of the curriculum,
#: because the tier-3 chain must pay back its setup time before it earns more
#: than tier-1 hand play.
SCIENCE_TIERS_MAX_TIMESTEPS: int = 1000

#: The three packs that a lab consumes, in tier order.
TIER_PACKS: tuple[int, ...] = (
    int(ItemType.TIER1_SCIENCE_PACK),
    int(ItemType.TIER2_SCIENCE_PACK),
    int(ItemType.TIER3_SCIENCE_PACK),
)

#: Raw-ore cost of one pack craft, by tier. Each half-fabricate counts as the
#: ore that it costs.
TIER_ORE_COSTS: tuple[int, ...] = (2, 4, 8)

#: Packs that one craft produces, by tier. The ore cost doubles with each tier
#: and the output goes up by a factor of four, so the science for each unit of
#: ore doubles: 0.5, then 1.0, then 2.0.
TIER_OUTPUTS: tuple[int, ...] = (1, 4, 16)

SCIENCE_TIERS_RECIPES: tuple[Recipe, ...] = (
    # -- The tier materials. Each one ends in raw ore. --
    Recipe(
        output=int(ItemType.WIRE),
        inputs=((int(ItemType.IRON_ORE), 1), (int(ItemType.TIN_ORE), 1)),
        ticks=4,
        name="Wire",
    ),
    Recipe(
        output=int(ItemType.FRAME),
        inputs=((int(ItemType.COPPER_ORE), 1), (int(ItemType.SILICON), 1)),
        ticks=4,
        name="Frame",
    ),
    Recipe(
        output=int(ItemType.MOTOR),
        inputs=((int(ItemType.WIRE), 1), (int(ItemType.FRAME), 1)),
        ticks=6,
        name="Motor",
    ),
    # -- The science packs. Each tier consumes the pack of the tier below. --
    Recipe(
        output=int(ItemType.TIER1_SCIENCE_PACK),
        inputs=((int(ItemType.COAL), 1), (int(ItemType.LIMESTONE), 1)),
        ticks=4,
        output_count=TIER_OUTPUTS[0],
        name="Tier 1 Science Pack",
    ),
    Recipe(
        output=int(ItemType.TIER2_SCIENCE_PACK),
        inputs=(
            (int(ItemType.TIER1_SCIENCE_PACK), 1),
            (int(ItemType.WIRE), 1),
        ),
        ticks=6,
        output_count=TIER_OUTPUTS[1],
        name="Tier 2 Science Pack",
    ),
    Recipe(
        output=int(ItemType.TIER3_SCIENCE_PACK),
        inputs=(
            (int(ItemType.TIER2_SCIENCE_PACK), 1),
            (int(ItemType.MOTOR), 1),
        ),
        ticks=8,
        output_count=TIER_OUTPUTS[2],
        name="Tier 3 Science Pack",
    ),
    # -- The machines. They cost little ore, as in EasyRocket, so automation
    # -- costs time and not a large amount of ore. --
    Recipe(
        output=int(ItemType.MINER),
        inputs=((int(ItemType.LIMESTONE), 1), (int(ItemType.SILICON), 1)),
        ticks=6,
        name="Miner",
    ),
    Recipe(
        output=int(ItemType.ASSEMBLER),
        inputs=((int(ItemType.COAL), 1), (int(ItemType.SILICON), 1)),
        ticks=8,
        name="Assembler",
    ),
    Recipe(
        output=int(ItemType.SCIENCE_LAB),
        inputs=((int(ItemType.IRON_ORE), 1), (int(ItemType.COPPER_ORE), 1)),
        ticks=8,
        name="Science Lab",
    ),
    Recipe(
        output=int(ItemType.CONVEYOR_BELT),
        inputs=((int(ItemType.COAL), 1), (int(ItemType.IRON_ORE), 1)),
        ticks=4,
        name="Conveyor Belt",
    ),
    Recipe(
        output=int(ItemType.ARM),
        inputs=((int(ItemType.TIN_ORE), 1), (int(ItemType.COPPER_ORE), 1)),
        ticks=4,
        name="Arm",
    ),
    Recipe(
        output=int(ItemType.PALLET),
        inputs=((int(ItemType.SILICON), 1), (int(ItemType.IRON_ORE), 1)),
        ticks=4,
        name="Pallet",
    ),
)

SCIENCE_TIERS_RECIPE_BOOK: RecipeBook = RecipeBook(recipes=SCIENCE_TIERS_RECIPES)

SCIENCE_TIERS_RECIPE_TABLE: RecipeTable = RecipeTable.from_book(
    SCIENCE_TIERS_RECIPE_BOOK
)


def _place_lab_at_spawn(key: jax.Array, state: EnvState, params: EnvParams) -> EnvState:
    """Reset hook that places one lab on the tile in front of the player.

    The function calls :func:`place_machine`, so the entity slot comes from the
    same path as a placement in the game. It then removes the temporary lab
    item from the inventory. The world therefore starts with a lab, and the
    player starts with nothing.
    """
    del key
    lab = int(ItemType.SCIENCE_LAB)
    inv = state.player_inventory.at[0, lab].set(1)
    state = state.replace(player_inventory=inv)
    state = place_machine(state, params, 0, lab)
    inv = state.player_inventory.at[0, lab].set(0)
    return state.replace(player_inventory=inv)


def science_tiers_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Count the packs that the labs consumed in this tick, 1 for every tier.

    ``science_consumed_step`` is the consumption delta of one tick, and the
    step sets it to zero at its start. This reward is therefore a dense
    throughput reward with no latch. More science is always more reward, and
    the extra value of a higher tier comes from the output counts in the recipe
    book alone.
    """
    del prev_state, params
    return jnp.sum(new_state.science_consumed_step).astype(jnp.float32)


def science_tiers(
    *,
    obs: str = "superficial_local",
    obs_radius: int = 7,
) -> tuple[FactoriaxEnv, EnvParams]:
    """Build the ScienceTiers-v1 environment.

    The player starts with an empty inventory on the six-patch map, and one lab
    already stands on it. The reward is the number of packs that the labs
    consume in each tick. The score has no upper limit, and the episode runs
    the full 1000 ticks.

    Parameters
    ----------
    obs :
        Observation variant. The function passes it to :class:`FactoriaxEnv`.
    obs_radius :
        Half-width of the egocentric local window. The default radius 7 gives a
        15x15 view on the 16x16 map. A ``_global`` variant ignores it.
    """
    env = FactoriaxEnv(
        terrain_fn=six_patch_terrain,
        reset_hooks=(_place_lab_at_spawn,),
        reward_fn=science_tiers_reward,
        obs=obs,
        obs_radius=obs_radius,
        map_width=MAP_SIZE,
        map_height=MAP_SIZE,
        num_players=1,
        max_machines=100,
    )
    params = EnvParams(
        max_timesteps=SCIENCE_TIERS_MAX_TIMESTEPS,
        recipe_table=SCIENCE_TIERS_RECIPE_TABLE,
        base_resources=ORE_RESOURCES_PER_TILE,
    )
    return env, params
