"""ScienceTiers-v1 — throughput reward over a three-tier science economy.

One scenario, one currency: labs consume science packs (any tier) and
every pack pays reward 1 the tick it is consumed. There are no
achievement bits and no per-tier reward weights — the tier ladder lives
entirely in the recipe book, where ore costs double per tier while pack
output quadruples:

======  ==============================  ========  ======  ===========
tier    recipe                          ore cost  output  science/ore
======  ==============================  ========  ======  ===========
1       coal + limestone                2         1       0.5
2       tier-1 pack + wire              4         4       1.0
3       tier-2 pack + motor             8         16      2.0
======  ==============================  ========  ======  ===========

(wire = iron + tin ore; motor = wire + frame; frame = copper + silicon
ore — every intermediate bottoms out in raw ore, so the per-tier ore
costs above are exact.)

Each tier consumes the pack below it, so climbing means sacrificing
packs that could have been cashed in — steady-state throughput per ore
doubles per tier, but the payback arrives only after the deeper chain
is running. The research target is exactly that tension: agents parked
at a lower-tier local optimum versus agents that invest through it.

The world is the shared 16x16 six-patch map (one 2x2 patch per ore),
with one science lab pre-placed on the tile the player initially faces
so the consume-for-reward loop is discoverable without first deducing
that labs exist. More labs, miners, assemblers, belts, and arms are
all craftable from raw ore.

Obs defaults match the miner curriculum (egocentric
``superficial_local``, radius 7) so policy networks transfer.
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

#: Episode budget. Longer than the curriculum's 300: the tier-3 chain
#: has to amortize its setup time before it out-earns tier-1 hand play.
SCIENCE_TIERS_MAX_TIMESTEPS: int = 1000

#: The three lab-consumable packs, tier order.
TIER_PACKS: tuple[int, ...] = (
    int(ItemType.TIER1_SCIENCE_PACK),
    int(ItemType.TIER2_SCIENCE_PACK),
    int(ItemType.TIER3_SCIENCE_PACK),
)

#: Raw-ore cost per pack craft, by tier (intermediates fully expanded).
TIER_ORE_COSTS: tuple[int, ...] = (2, 4, 8)

#: Packs produced per craft, by tier. Ore costs double per tier while
#: output quadruples, so science-per-ore doubles: 0.5 / 1.0 / 2.0.
TIER_OUTPUTS: tuple[int, ...] = (1, 4, 16)

SCIENCE_TIERS_RECIPES: tuple[Recipe, ...] = (
    # -- Tier materials (all bottom out in raw ore). --
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
    # -- Science packs: each tier consumes the pack below it. --
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
    # -- Machines, ore-cheap (EasyRocket precedent) so automation is a
    # -- time investment rather than a resource cliff. --
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

SCIENCE_TIERS_RECIPE_BOOK: RecipeBook = RecipeBook(
    recipes=SCIENCE_TIERS_RECIPES
)

SCIENCE_TIERS_RECIPE_TABLE: RecipeTable = RecipeTable.from_book(
    SCIENCE_TIERS_RECIPE_BOOK
)


def _place_lab_at_spawn(
    key: jax.Array, state: EnvState, params: EnvParams
) -> EnvState:
    """Reset hook: pre-place one lab on the tile the player faces.

    Routes through :func:`place_machine` so entity allocation matches
    in-game placement exactly; the temporary lab item is stripped from
    the inventory afterwards — the world starts with a lab, the player
    starts with nothing.
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
    """Packs consumed by labs this tick, every tier worth 1.

    ``science_consumed_step`` is the per-tick consumption delta (reset
    at the top of every step), so this is dense throughput reward with
    no latching — more science is always more reward, and the tier
    premium comes from the recipe book's output counts alone.
    """
    del prev_state, params
    return jnp.sum(new_state.science_consumed_step).astype(jnp.float32)


def science_tiers(
    *,
    obs: str = "superficial_local",
    obs_radius: int = 7,
) -> tuple[FactoriaxEnv, EnvParams]:
    """Build the ScienceTiers-v1 env.

    Empty inventory on the six-patch map with one pre-placed lab;
    reward is packs consumed per tick. Unbounded score; the episode
    runs the full 1000-tick budget.

    Parameters
    ----------
    obs :
        Observation variant passed to :class:`FactoriaxEnv`.
    obs_radius :
        Half-width of the egocentric local window (the default radius 7
        gives a 15×15 view on the 16×16 map); ignored for ``_global``
        variants.
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
