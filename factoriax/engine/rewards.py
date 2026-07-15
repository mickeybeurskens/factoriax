"""Reward functions for the FactoriaX environment.

Reward functions share a common signature ``(prev_state, new_state, params) ->
jax.Array`` so they are interchangeable and composable.  Functions are
pure JAX and fully JIT-compatible; bind static arguments (e.g. the radius
used by :func:`mining_reward`) with :func:`functools.partial` before passing
to :func:`jax.jit`.
"""

import jax
import jax.numpy as jnp

from factoriax.engine.achievements import CORE_ACHIEVEMENT_WEIGHTS
from factoriax.engine.constants import (
    ItemType,
    Machine,
)
from factoriax.engine.state import EnvParams, EnvState
from factoriax.engine.tables import MINEABLE_BLOCKS


def achievement_reward(
    prev_state: EnvState,
    new_state: EnvState,
    params: EnvParams,
    weights: jax.Array = CORE_ACHIEVEMENT_WEIGHTS,
) -> jax.Array:
    """Sparse reward for newly unlocked achievements.
    
    Compares ``achievements_unlocked`` between the two ``EnvState``
    instances and returns the weighted sum of newly satisfied slots.
    The ``weights`` vector controls the magnitude per slot — slots with
    zero weight contribute nothing. Achievements live on
    :class:`~factoriax.engine.state.EnvState` directly; the env's
    ``achievement_fn`` constructor argument latches them each step.
    
    Parameters
    ----------
        prev_state: EnvState immediately before the step.
        new_state: EnvState immediately after the step.

    Parameters
    ----------
    uniformity :
        
    weights :
        Per
    MAX_ACHIEVEMENTS :
        Defaults to the core game weights
    1 :
        0 for each core tutorial milestone
    prev_state : EnvState :
        
    new_state : EnvState :
        
    params : EnvParams :
        
    weights : jax.Array :
        (Default value = CORE_ACHIEVEMENT_WEIGHTS)
    prev_state: EnvState :
        
    new_state: EnvState :
        
    params: EnvParams :
        
    weights: jax.Array :
         (Default value = CORE_ACHIEVEMENT_WEIGHTS)

    Returns
    -------

    
    """
    newly_unlocked = new_state.achievements_unlocked & ~prev_state.achievements_unlocked
    reward: jax.Array = jnp.sum(weights * newly_unlocked)
    return reward


def mining_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Dense reward combining proximity to ore and a bonus for each ore mined.
    
    Two components are summed:
    
    - **Proximity**: ``0.05 / (1 + d)`` where ``d`` is the Manhattan distance
      from the selected player to the nearest ore tile (coal, iron, or copper).
      This is a small shaping signal (max 0.05) that guides the agent toward
      ore without dominating the mining bonus.
    - **Mining bonus**: 20.0 per ore item extracted during this step, computed
      as the delta in ``items_mined`` between ``prev_state`` and ``new_state``
      summed over the three mineable item types.
    
    Parameters
    ----------
        prev_state: State immediately before the step.
        new_state: State immediately after the step.

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

    
    >>> import jax
        >>> import factoriax
        >>> env, params = factoriax.make("EasyRocket-v1")
        >>> _, state = env.reset_env(jax.random.PRNGKey(0), params)
        >>> _, next_state, _, _, _ = env.step_env(
        ...     jax.random.PRNGKey(1), state, 0, params
        ... )
        >>> reward = factoriax.mining_reward(state, next_state, params)
        >>> float(reward) >= 0.0
        True
    """
    player_pos = new_state.player_positions[new_state.selected_player]
    px, py = player_pos[0], player_pos[1]
    map_h, map_w = new_state.map.shape

    grid_y, grid_x = jnp.meshgrid(jnp.arange(map_h), jnp.arange(map_w), indexing="ij")
    dist = jnp.abs(grid_x - px) + jnp.abs(grid_y - py)

    # Identify ore tiles: broadcast (H, W, 1) == (3,) -> (H, W, 3) -> (H, W)
    is_ore = jnp.any(new_state.map[..., None] == MINEABLE_BLOCKS, axis=-1)

    large = jnp.int32(map_h + map_w)
    ore_dist = jnp.where(is_ore, dist, large)
    min_dist = jnp.min(ore_dist)

    proximity: jax.Array = 0.05 / (1.0 + min_dist.astype(jnp.float32))

    ore_items = jnp.array(
        [ItemType.COAL, ItemType.IRON_ORE, ItemType.COPPER_ORE], dtype=jnp.int32
    )
    mined_delta = jnp.sum(
        new_state.items_mined[ore_items] - prev_state.items_mined[ore_items]
    )
    mining_bonus: jax.Array = 20.0 * mined_delta.astype(jnp.float32)

    return proximity + mining_bonus


def sparse_mining_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Sparse reward of 1.0 for each ore item mined during this step.
    
    Counts the total delta across coal, iron, and copper in ``items_mined``
    between the two states.  This signal is zero on every step where nothing
    is extracted, which makes it harder to shape behaviour but trivial to
    interpret: one unit of reward per one unit of ore.
    
    Parameters
    ----------
        prev_state: State immediately before the step.
        new_state: State immediately after the step.

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
    ore_items = jnp.array(
        [ItemType.COAL, ItemType.IRON_ORE, ItemType.COPPER_ORE], dtype=jnp.int32
    )
    delta = jnp.sum(
        new_state.items_mined[ore_items] - prev_state.items_mined[ore_items]
    )
    reward: jax.Array = delta.astype(jnp.float32)
    return reward


def sparse_pallet_crafting_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Sparse reward of 1.0 for each pallet gained via crafting.
    
    Detects crafting by requiring that the player's inventory gained
    pallets *and* lost iron in the same step. Moving pallets between
    inventory and machines changes pallet count without consuming iron,
    so place/pickup/deposit/withdraw exploits yield zero reward.
    
    Parameters
    ----------
        prev_state: State immediately before the step.
        new_state: State immediately after the step.

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

    p = new_state.selected_player
    pallet_delta = (
        new_state.player_inventory[p, ItemType.PALLET]
        - prev_state.player_inventory[p, ItemType.PALLET]
    )
    iron_delta = (
        new_state.player_inventory[p, ItemType.IRON_ORE]
        - prev_state.player_inventory[p, ItemType.IRON_ORE]
    )

    # Crafting consumes iron and produces pallets in the same step.
    is_craft = (pallet_delta > 0) & (iron_delta < 0)
    reward: jax.Array = jnp.where(is_craft, pallet_delta, 0).astype(jnp.float32)
    return reward


def sparse_miner_crafting_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Sparse reward of 1.0 for each miner gained via crafting.
    
    Detects crafting by requiring that the player's inventory gained
    miners *and* lost both iron and copper in the same step. Moving
    miners between inventory and the map via place/pickup does not
    consume resources, so those actions yield zero reward.
    
    Parameters
    ----------
        prev_state: State immediately before the step.
        new_state: State immediately after the step.

    Parameters
    ----------
    uniformity :
        
    prev_state : EnvState :
        
    new_state : EnvState :
        
    params : EnvParams :
        
    prev_state: EnvState :
        
    new_state: EnvState :
        
    params: EnvParams :
        

    Returns
    -------

    
    """

    p = new_state.selected_player
    miner_delta = (
        new_state.player_inventory[p, ItemType.MINER]
        - prev_state.player_inventory[p, ItemType.MINER]
    )
    iron_delta = (
        new_state.player_inventory[p, ItemType.IRON_ORE]
        - prev_state.player_inventory[p, ItemType.IRON_ORE]
    )
    copper_delta = (
        new_state.player_inventory[p, ItemType.COPPER_ORE]
        - prev_state.player_inventory[p, ItemType.COPPER_ORE]
    )

    is_craft = (miner_delta > 0) & (iron_delta < 0) & (copper_delta < 0)
    reward: jax.Array = jnp.where(is_craft, miner_delta, 0).astype(jnp.float32)
    return reward


def miner_output_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Reward for each ore item produced by placed miners.
    
    Counts the total increase in buffer item counts across all
    miner-type entities. This gives a dense signal that fires every
    tick a miner extracts ore.
    
    Parameters
    ----------
        prev_state: State immediately before the step.
        new_state: State immediately after the step.

    Parameters
    ----------
    uniformity :
        
    prev_state : EnvState :
        
    new_state : EnvState :
        
    params : EnvParams :
        
    prev_state: EnvState :
        
    new_state: EnvState :
        
    params: EnvParams :
        

    Returns
    -------

    
    """
    is_miner = (new_state.ent_type == Machine.MINER) & (new_state.ent_y >= 0)
    prev_output = jnp.where(is_miner, prev_state.ent_buf_count, 0)
    new_output = jnp.where(is_miner, new_state.ent_buf_count, 0)
    delta = jnp.sum(new_output) - jnp.sum(prev_output)
    reward: jax.Array = delta.astype(jnp.float32)
    return reward


def pallet_filling_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Reward of 1.0 for each item deposited into a pallet machine.
    
    Counts the total increase in buffer item counts across all
    pallet-type entities. This gives a dense signal for every
    successful deposit action rather than waiting for a full stack.
    
    Parameters
    ----------
        prev_state: State immediately before the step.
        new_state: State immediately after the step.

    Parameters
    ----------
    uniformity :
        
    prev_state : EnvState :
        
    new_state : EnvState :
        
    params : EnvParams :
        
    prev_state: EnvState :
        
    new_state: EnvState :
        
    params: EnvParams :
        

    Returns
    -------

    
    """
    is_pallet = (new_state.ent_type == Machine.PALLET) & (new_state.ent_y >= 0)
    prev_counts = jnp.where(is_pallet, prev_state.ent_buf_count, 0)
    new_counts = jnp.where(is_pallet, new_state.ent_buf_count, 0)
    delta = jnp.sum(new_counts) - jnp.sum(prev_counts)
    reward: jax.Array = delta.astype(jnp.float32)
    return reward


def player_inventory_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Reward for each item gained in the selected player's inventory.
    
    Counts the total increase in item counts across all inventory slots.
    Positive when items are added (withdraw, mine), zero or negative when
    items are consumed (craft, deposit, place).
    
    Parameters
    ----------
        prev_state: State immediately before the step.
        new_state: State immediately after the step.

    Parameters
    ----------
    uniformity :
        
    prev_state : EnvState :
        
    new_state : EnvState :
        
    params : EnvParams :
        
    prev_state: EnvState :
        
    new_state: EnvState :
        
    params: EnvParams :
        

    Returns
    -------

    
    """
    p = new_state.selected_player
    prev_total = jnp.sum(prev_state.player_inventory[p])
    new_total = jnp.sum(new_state.player_inventory[p])
    reward: jax.Array = (new_total - prev_total).astype(jnp.float32)
    return reward
