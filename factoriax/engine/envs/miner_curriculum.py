"""Miner-curriculum scenarios — staged decomposition of EasyRocket's bootstrap.

Three independent scenarios teach the mine -> craft -> place loop that
opens EasyRocket-v1, in increasing order of composition:

1. ``MineOres-v1`` — mine 5 of every ore type.
2. ``PlaceMiners-v1`` — miners pre-stocked; get six producing on ore.
3. ``MinerBootstrap-v1`` — the full loop from an empty inventory.

Every scenario shares EasyRocket-v1's world (the 16x16 six-patch
terrain from :mod:`factoriax.engine.envs.common`) and recipe table, and
all three default to the same egocentric ``superficial_local`` obs with
radius 7, so one policy network transfers across all stages — and onto
EasyRocket-v1 built with the same obs kwargs — without surgery. The
local default follows the Mining-v1 precedent: flat-MLP policies
plateau on ``superficial_global`` but solve the equivalent task on
``superficial_local`` (see :func:`factoriax.engine.envs.mining.mining`).

Rewards are graded latched achievement bits: one bit per progress
threshold, weight 1.0, latched by
:func:`~factoriax.engine.envs.base.achievement_hook`. Reward density
matches a per-event signal on the way up, but nothing can be re-earned
past the high-water mark — pickup/place cycling and craft spam pay
zero. The mining bits read the cumulative ``items_mined`` counter, so
they are monotone even when the inventory is spent on crafts.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp

from factoriax.engine.constants import MAX_ACHIEVEMENTS, ItemType, Machine
from factoriax.engine.envs.base import FactoriaxEnv, achievement_hook
from factoriax.engine.envs.common import (
    MAP_SIZE,
    ORE_RESOURCES_PER_TILE,
    count_machines,
    producing_miners,
    six_patch_terrain,
)
from factoriax.engine.envs.easy_rocket import EASY_ROCKET_RECIPE_TABLE
from factoriax.engine.rewards import achievement_reward
from factoriax.engine.state import EnvParams, EnvState

#: Shared episode budget. Uniform across the curriculum; the scripted
#: oracles in ``tests/scenarios/test_miner_curriculum.py`` pin each
#: scenario's actual solve time well under it.
_MAX_TIMESTEPS: int = 300

#: Curriculum-wide target: six miners, one per ore patch.
_N_MINERS: int = 6

#: The six raw ore items, one per patch block, in patch order.
_ORE_ITEMS: tuple[int, ...] = (
    int(ItemType.IRON_ORE),
    int(ItemType.COPPER_ORE),
    int(ItemType.TIN_ORE),
    int(ItemType.SILICON),
    int(ItemType.COAL),
    int(ItemType.LIMESTONE),
)

_Condition = Callable[[EnvState], jax.Array]


def _conditions_to_achievement_fn(
    conditions: tuple[_Condition, ...],
) -> Callable[[EnvState], jax.Array]:
    """Stack per-bit conditions and zero-pad to ``MAX_ACHIEVEMENTS``."""

    def achievement_fn(state: EnvState) -> jax.Array:
        bits = jnp.stack([condition(state) for condition in conditions])
        padding = jnp.zeros(MAX_ACHIEVEMENTS - bits.shape[0], dtype=jnp.bool_)
        return jnp.concatenate([bits, padding])

    return achievement_fn


def _unit_weights(num_bits: int) -> jax.Array:
    """Weight 1.0 for the first ``num_bits`` achievement slots."""
    return (
        jnp.zeros(MAX_ACHIEVEMENTS, dtype=jnp.float32).at[:num_bits].set(1.0)
    )


def _mined_at_least(state: EnvState, item: int, count: int) -> jax.Array:
    """Cumulative mining counter for ``item`` has reached ``count``."""
    return state.items_mined[item] >= count


def _holds_at_least(state: EnvState, item: int, count: int) -> jax.Array:
    """Player inventories hold at least ``count`` of ``item``."""
    return jnp.sum(state.player_inventory[:, item]) >= count


def _producing_at_least(state: EnvState, count: int) -> jax.Array:
    """At least ``count`` placed miners have ore in their output buffer.

    A miner on dirt never fills its buffer, so this implies the miners
    sit on ore tiles.
    """
    producing, _ = producing_miners(state)
    return jnp.sum(producing) >= count


def _stock_inventory(
    items: tuple[tuple[int, int], ...],
) -> Callable[[jax.Array, EnvState, EnvParams], EnvState]:
    """Reset hook setting ``(item, count)`` pairs on every player."""

    def hook(key: jax.Array, state: EnvState, params: EnvParams) -> EnvState:
        del key, params
        inventory = state.player_inventory
        for item, count in items:
            inventory = inventory.at[:, item].set(count)
        return state.replace(player_inventory=inventory)

    return hook


# ---------------------------------------------------------------------------
# MineOres-v1
# ---------------------------------------------------------------------------

#: Per ore type, one latched bit per mined item up to this many.
_MINE_PER_ORE: int = 5

#: 30 bits: ``items_mined[ore] >= k`` for k = 1..5, for each of the six
#: ores. ``items_mined`` only ever grows, so the bits are monotone by
#: construction — spending ore on crafts cannot un-earn them.
_MINE_ORES_CONDITIONS: tuple[_Condition, ...] = tuple(
    partial(_mined_at_least, item=item, count=count)
    for item in _ORE_ITEMS
    for count in range(1, _MINE_PER_ORE + 1)
)

NUM_MINE_ORES_ACHIEVEMENTS: int = len(_MINE_ORES_CONDITIONS)

MINE_ORES_MAX_SCORE: float = float(NUM_MINE_ORES_ACHIEVEMENTS)

mine_ores_conditions = _conditions_to_achievement_fn(_MINE_ORES_CONDITIONS)

MINE_ORES_ACHIEVEMENT_WEIGHTS: jax.Array = _unit_weights(
    NUM_MINE_ORES_ACHIEVEMENTS
)


def mine_ores_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Sparse reward for newly latched MineOres-v1 bits."""
    return achievement_reward(
        prev_state, new_state, params, weights=MINE_ORES_ACHIEVEMENT_WEIGHTS
    )


def mine_ores(
    *,
    obs: str = "superficial_local",
    obs_radius: int = 7,
) -> tuple[FactoriaxEnv, EnvParams]:
    """Build the MineOres-v1 env — curriculum stage 1.

    Mine 5 of every ore type on the shared six-patch map. Max score 30.

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
        step_hooks=(achievement_hook(mine_ores_conditions),),
        reward_fn=mine_ores_reward,
        obs=obs,
        obs_radius=obs_radius,
        map_width=MAP_SIZE,
        map_height=MAP_SIZE,
        num_players=1,
        max_machines=100,
    )
    params = EnvParams(
        max_timesteps=_MAX_TIMESTEPS,
        recipe_table=EASY_ROCKET_RECIPE_TABLE,
        base_resources=ORE_RESOURCES_PER_TILE,
    )
    return env, params


# ---------------------------------------------------------------------------
# PlaceMiners-v1
# ---------------------------------------------------------------------------

#: 6 bits: >= k producing miners, k = 1..6. Placement itself is legal
#: on any free tile — the reward, not the engine, enforces "on ore",
#: because only a miner on ore ever fills its output buffer.
_PLACE_MINERS_CONDITIONS: tuple[_Condition, ...] = tuple(
    partial(_producing_at_least, count=count)
    for count in range(1, _N_MINERS + 1)
)

NUM_PLACE_MINERS_ACHIEVEMENTS: int = len(_PLACE_MINERS_CONDITIONS)

PLACE_MINERS_MAX_SCORE: float = float(NUM_PLACE_MINERS_ACHIEVEMENTS)

place_miners_conditions = _conditions_to_achievement_fn(
    _PLACE_MINERS_CONDITIONS
)

PLACE_MINERS_ACHIEVEMENT_WEIGHTS: jax.Array = _unit_weights(
    NUM_PLACE_MINERS_ACHIEVEMENTS
)

_PLACE_MINERS_STOCK: tuple[tuple[int, int], ...] = (
    (int(ItemType.MINER), _N_MINERS),
)


def place_miners_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Sparse reward for newly latched PlaceMiners-v1 bits."""
    return achievement_reward(
        prev_state, new_state, params, weights=PLACE_MINERS_ACHIEVEMENT_WEIGHTS
    )


def all_miners_producing(state: EnvState, params: EnvParams) -> jax.Array:
    """Episode-ending condition: six miners producing at once."""
    del params
    return _producing_at_least(state, _N_MINERS)


def place_miners(
    *,
    obs: str = "superficial_local",
    obs_radius: int = 7,
) -> tuple[FactoriaxEnv, EnvParams]:
    """Build the PlaceMiners-v1 env — curriculum stage 2.

    Inventory starts with six miners; get all six producing on ore.
    Max score 6; the episode ends early once all six produce.

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
        step_hooks=(achievement_hook(place_miners_conditions),),
        reset_hooks=(_stock_inventory(_PLACE_MINERS_STOCK),),
        reward_fn=place_miners_reward,
        done_fn=all_miners_producing,
        obs=obs,
        obs_radius=obs_radius,
        map_width=MAP_SIZE,
        map_height=MAP_SIZE,
        num_players=1,
        max_machines=100,
    )
    params = EnvParams(
        max_timesteps=_MAX_TIMESTEPS,
        recipe_table=EASY_ROCKET_RECIPE_TABLE,
        base_resources=ORE_RESOURCES_PER_TILE,
    )
    return env, params


# ---------------------------------------------------------------------------
# MinerBootstrap-v1
# ---------------------------------------------------------------------------


def _placed_at_least(state: EnvState, count: int) -> jax.Array:
    """At least ``count`` miners are placed on the map."""
    return count_machines(state, int(Machine.MINER)) >= count


#: 14 bits, an EasyRocket-style gate ladder over the full
#: mine -> craft -> place loop, so a from-scratch agent always has a
#: nearby next gate:
#:
#: - mining (4): 1 and 6 of each miner ingredient, on the monotone
#:   ``items_mined`` counter (6 of each = materials for six miners);
#: - craft (1): hold a miner (graded hold thresholds would never fire
#:   for a policy that places miners as it crafts them);
#: - placement (3): 1 / 3 / 6 miners placed — the same thresholds as
#:   EasyRocket's place_miner / place_3_miners / metal_in_motion;
#: - production (6): 1..6 producing miners.
_MINER_BOOTSTRAP_CONDITIONS: tuple[_Condition, ...] = (
    partial(_mined_at_least, item=int(ItemType.LIMESTONE), count=1),
    partial(_mined_at_least, item=int(ItemType.LIMESTONE), count=_N_MINERS),
    partial(_mined_at_least, item=int(ItemType.SILICON), count=1),
    partial(_mined_at_least, item=int(ItemType.SILICON), count=_N_MINERS),
    partial(_holds_at_least, item=int(ItemType.MINER), count=1),
    partial(_placed_at_least, count=1),
    partial(_placed_at_least, count=3),
    partial(_placed_at_least, count=_N_MINERS),
    *(
        partial(_producing_at_least, count=count)
        for count in range(1, _N_MINERS + 1)
    ),
)

NUM_MINER_BOOTSTRAP_ACHIEVEMENTS: int = len(_MINER_BOOTSTRAP_CONDITIONS)

MINER_BOOTSTRAP_MAX_SCORE: float = float(NUM_MINER_BOOTSTRAP_ACHIEVEMENTS)

miner_bootstrap_conditions = _conditions_to_achievement_fn(
    _MINER_BOOTSTRAP_CONDITIONS
)

MINER_BOOTSTRAP_ACHIEVEMENT_WEIGHTS: jax.Array = _unit_weights(
    NUM_MINER_BOOTSTRAP_ACHIEVEMENTS
)


def miner_bootstrap_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Sparse reward for newly latched MinerBootstrap-v1 bits."""
    return achievement_reward(
        prev_state,
        new_state,
        params,
        weights=MINER_BOOTSTRAP_ACHIEVEMENT_WEIGHTS,
    )


def miner_bootstrap(
    *,
    obs: str = "superficial_local",
    obs_radius: int = 7,
) -> tuple[FactoriaxEnv, EnvParams]:
    """Build the MinerBootstrap-v1 env — curriculum stage 3 (capstone).

    Empty inventory; mine limestone + silicon, craft six miners, and get
    all six producing on ore. Max score 14; the episode ends early once
    all six produce.

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
        step_hooks=(achievement_hook(miner_bootstrap_conditions),),
        reward_fn=miner_bootstrap_reward,
        done_fn=all_miners_producing,
        obs=obs,
        obs_radius=obs_radius,
        map_width=MAP_SIZE,
        map_height=MAP_SIZE,
        num_players=1,
        max_machines=100,
    )
    params = EnvParams(
        max_timesteps=_MAX_TIMESTEPS,
        recipe_table=EASY_ROCKET_RECIPE_TABLE,
        base_resources=ORE_RESOURCES_PER_TILE,
    )
    return env, params
