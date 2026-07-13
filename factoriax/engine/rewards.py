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

# ---------------------------------------------------------------------------
# Proximity helper
# ---------------------------------------------------------------------------


def _proximity(state: EnvState, tile_mask: jax.Array) -> jax.Array:
    """Inverse Manhattan distance from the selected player to the nearest
    True tile in *tile_mask*.
    
    Returns ``1 / (1 + d)`` where *d* is the Manhattan distance, giving
    a value in (0, 1] when at least one tile matches and a small value
    when no tile matches (distance clamped to map_h + map_w).

    Parameters
    ----------
    state :
        Current environment state.
    tile_mask :
        Boolean array of shape ``(map_h, map_w)``.
    state : EnvState :
        
    tile_mask : jax.Array :
        
    state: EnvState :
        
    tile_mask: jax.Array :
        

    Returns
    -------

    
    """
    pos = state.player_positions[state.selected_player]
    px, py = pos[0], pos[1]
    map_h, map_w = state.map.shape
    grid_y, grid_x = jnp.meshgrid(jnp.arange(map_h), jnp.arange(map_w), indexing="ij")
    dist = jnp.abs(grid_x - px) + jnp.abs(grid_y - py)
    large = jnp.int32(map_h + map_w)
    min_dist = jnp.min(jnp.where(tile_mask, dist, large))
    return 1.0 / (1.0 + min_dist.astype(jnp.float32))


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


def miner_throughput_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Reward for ore extracted from blocks by placed miners each tick.
    
    Measures the decrease in ``block_resources`` on tiles that have a
    miner. This counts actual extraction from the ground rather than
    output slot changes, so it is unaffected by the agent withdrawing
    from or ignoring the output slot.
    
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
    is_miner = new_state.machine_types == Machine.MINER
    prev_res = prev_state.block_resources.astype(jnp.int32)
    new_res = new_state.block_resources.astype(jnp.int32)
    depleted = jnp.where(is_miner, prev_res - new_res, 0)
    reward: jax.Array = jnp.sum(jnp.maximum(depleted, 0)).astype(jnp.float32)
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


# ---------------------------------------------------------------------------
# Dense reward functions for basic_skills scenario levels
# ---------------------------------------------------------------------------


def _ore_proximity(state: EnvState) -> jax.Array:
    """Proximity to the nearest mineable ore tile.

    Parameters
    ----------
    state : EnvState :
        
    state: EnvState :
        

    Returns
    -------

    
    """
    is_ore = jnp.any(state.map[..., None] == MINEABLE_BLOCKS, axis=-1)
    return _proximity(state, is_ore)


def _mining_delta(prev: EnvState, new: EnvState) -> jax.Array:
    """Total ore items mined this step.

    Parameters
    ----------
    prev : EnvState :
        
    new : EnvState :
        
    prev: EnvState :
        
    new: EnvState :
        

    Returns
    -------

    
    """
    ore = jnp.array(
        [ItemType.COAL, ItemType.IRON_ORE, ItemType.COPPER_ORE], dtype=jnp.int32
    )
    return jnp.sum(new.items_mined[ore] - prev.items_mined[ore]).astype(jnp.float32)


def _pallet_filling_delta(prev: EnvState, new: EnvState) -> jax.Array:
    """Total items deposited into pallets this step.

    Parameters
    ----------
    prev : EnvState :
        
    new : EnvState :
        
    prev: EnvState :
        
    new: EnvState :
        

    Returns
    -------

    
    """
    is_pallet = (new.ent_type == Machine.PALLET) & (new.ent_y >= 0)
    prev_c = jnp.sum(jnp.where(is_pallet, prev.ent_buf_count, 0))
    new_c = jnp.sum(jnp.where(is_pallet, new.ent_buf_count, 0))
    return (new_c - prev_c).astype(jnp.float32)


def _inventory_delta(prev: EnvState, new: EnvState) -> jax.Array:
    """Net items gained in the selected player's inventory.

    Parameters
    ----------
    prev : EnvState :
        
    new : EnvState :
        
    prev: EnvState :
        
    new: EnvState :
        

    Returns
    -------

    
    """
    p = new.selected_player
    return (
        jnp.sum(new.player_inventory[p]) - jnp.sum(prev.player_inventory[p])
    ).astype(jnp.float32)


def _item_count(state: EnvState, item: int) -> jax.Array:
    """Count of a specific item type in player 0's inventory.

    Parameters
    ----------
    state : EnvState :
        
    item : int :
        
    state: EnvState :
        
    item: int :
        

    Returns
    -------

    
    """
    return state.player_inventory[0, item]


def dense_craft_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Dense reward for crafting levels (craft_pallets, craft_miners).
    
    Combines ore proximity (guides toward materials), mining delta
    (rewards collecting), and a large bonus per item crafted. Crafting
    is detected by checking that a placeable item count increased while
    raw materials decreased.
    
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
    proximity = _ore_proximity(new_state)
    mining = _mining_delta(prev_state, new_state)

    # Detect any crafting: total placeable items increased.
    placeables = jnp.array(
        [
            ItemType.MINER,
            ItemType.PALLET,
            ItemType.CONVEYOR_BELT,
            ItemType.ASSEMBLER,
        ],
        dtype=jnp.int32,
    )
    prev_count = jnp.sum(prev_state.player_inventory[0, placeables])
    new_count = jnp.sum(new_state.player_inventory[0, placeables])
    craft_delta = jnp.maximum(new_count - prev_count, 0)

    return proximity + mining + 10.0 * craft_delta.astype(jnp.float32)


def dense_fill_pallet_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Dense reward for the fill_pallet level.
    
    Combines ore proximity, pallet proximity, mining delta, and pallet
    filling delta.
    
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
    ore_prox = _ore_proximity(new_state)
    pallet_prox = _proximity(new_state, new_state.machine_types == Machine.PALLET)
    mining = _mining_delta(prev_state, new_state)
    filling = _pallet_filling_delta(prev_state, new_state)
    return ore_prox + pallet_prox + mining + 5.0 * filling


def dense_deploy_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Dense reward for deployment levels (deploy_miner, place_and_fuel,
    mining_factory).
    
    Combines ore proximity, mining delta, and miner output delta.
    
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
    proximity = _ore_proximity(new_state)
    mining = _mining_delta(prev_state, new_state)
    output = miner_output_reward(prev_state, new_state, params)
    return proximity + mining + 5.0 * output


def dense_withdraw_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Dense reward for the withdraw_ore level.
    
    Proximity to the nearest miner that still has items in its output
    slot, plus a bonus per item withdrawn into inventory.
    
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
    has_output = (
        (new_state.ent_type == Machine.MINER)
        & (new_state.ent_y >= 0)
        & (new_state.ent_buf_count > 0)
    )
    # Build a spatial mask from entity positions for proximity.
    h, w = new_state.map.shape
    has_output_grid = jnp.zeros((h, w), dtype=jnp.bool_)
    has_output_grid = has_output_grid.at[new_state.ent_y, new_state.ent_x].set(
        has_output
    )
    proximity = _proximity(new_state, has_output_grid)
    inv_gain = _inventory_delta(prev_state, new_state)
    return proximity + 10.0 * jnp.maximum(inv_gain, 0.0)


def dense_deposit_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Dense reward for the deposit_into_pallets level.
    
    Proximity to nearest pallet plus pallet filling bonus.
    
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
    pallet_prox = _proximity(new_state, new_state.machine_types == Machine.PALLET)
    filling = _pallet_filling_delta(prev_state, new_state)
    return pallet_prox + 5.0 * filling


def dense_pickup_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Dense reward for the pickup_machines level.
    
    Proximity to nearest machine on the map plus bonus for picking up
    machines (detected by machine count decrease on map).
    
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
    has_machine = new_state.machine_types != Machine.NONE
    proximity = _proximity(new_state, has_machine)
    prev_count = jnp.sum(prev_state.machine_types != Machine.NONE)
    new_count = jnp.sum(new_state.machine_types != Machine.NONE)
    picked_up = jnp.maximum(prev_count - new_count, 0)
    return proximity + 10.0 * picked_up.astype(jnp.float32)


def dense_belt_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Dense reward for the belt_line level.
    
    Rewards belt placement and items reaching the destination pallet.
    No proximity component (gap tile not identifiable from state).
    
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
    prev_belts = jnp.sum(prev_state.machine_types == Machine.CONVEYOR_BELT)
    new_belts = jnp.sum(new_state.machine_types == Machine.CONVEYOR_BELT)
    belt_placed = jnp.maximum(new_belts - prev_belts, 0)
    filling = _pallet_filling_delta(prev_state, new_state)
    return 5.0 * belt_placed.astype(jnp.float32) + 5.0 * filling


def dense_assembler_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Dense reward for the assembler_production level.
    
    Proximity to assembler, bonus for depositing inputs, and large
    bonus for science packs gained in inventory.
    
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
    asm_prox = _proximity(new_state, new_state.machine_types == Machine.ASSEMBLER)
    # Input deposited = total items in assembler entity buffers.
    is_asm = (new_state.ent_type == Machine.ASSEMBLER) & (new_state.ent_y >= 0)
    prev_inputs = jnp.sum(
        jnp.where(
            is_asm[:, None],
            prev_state.ent_asm_in_count,
            0,
        )
    )
    new_inputs = jnp.sum(
        jnp.where(
            is_asm[:, None],
            new_state.ent_asm_in_count,
            0,
        )
    )
    input_delta = jnp.maximum(new_inputs - prev_inputs, 0).astype(jnp.float32)
    # Science packs gained in player inventory.
    prev_packs = _item_count(prev_state, ItemType.TIER1_SCIENCE_PACK)
    new_packs = _item_count(new_state, ItemType.TIER1_SCIENCE_PACK)
    pack_delta = jnp.maximum(new_packs - prev_packs, 0).astype(jnp.float32)
    return asm_prox + 2.0 * input_delta + 10.0 * pack_delta
