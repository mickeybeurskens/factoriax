"""Reward functions for the FactoriaX environment.

Reward functions share a common signature ``(prev_state, new_state, params) ->
jax.Array`` so they are interchangeable and composable.  Functions are
pure JAX and fully JIT-compatible; bind static arguments (e.g. the radius
used by :func:`mining_reward`) with :func:`functools.partial` before passing
to :func:`jax.jit`.
"""

import jax
import jax.numpy as jnp

from factoriax.achievements import ACHIEVEMENT_REWARDS
from factoriax.constants import (
    MINEABLE_BLOCKS,
    ItemType,
    MachineType,
)
from factoriax.state import EnvParams, EnvState


def achievement_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Sparse reward equal to the sum of rewards for newly unlocked achievements.

    Compares ``achievements_unlocked`` between the two states and returns one
    reward unit per newly satisfied achievement.  The magnitude of each unit
    is defined by :data:`factoriax.achievements.ACHIEVEMENT_REWARDS` (default
    1.0 for all achievements).

    Args:
        prev_state: State immediately before the step.
        new_state: State immediately after the step.
        params: Environment parameters (unused; present for interface uniformity).

    Returns:
        Scalar float32 reward.
    """
    newly_unlocked = new_state.achievements_unlocked & ~prev_state.achievements_unlocked
    reward: jax.Array = jnp.sum(ACHIEVEMENT_REWARDS * newly_unlocked)
    return reward


def mining_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Dense reward combining proximity to ore and a bonus for each ore mined.

    Two components are summed:

    - **Proximity**: ``1 / (1 + d)`` where ``d`` is the Manhattan distance
      from the selected player to the nearest ore tile (coal, iron, or copper).
      This ranges from 0.0 (no ore on map) to 1.0 (player is standing on ore).
    - **Mining bonus**: 5.0 per ore item extracted during this step, computed
      as the delta in ``items_mined`` between ``prev_state`` and ``new_state``
      summed over the three mineable item types.

    Args:
        prev_state: State immediately before the step.
        new_state: State immediately after the step.
        params: Environment parameters (unused; present for interface uniformity).

    Returns:
        Scalar float32 reward.
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

    proximity: jax.Array = 1.0 / (1.0 + min_dist.astype(jnp.float32))

    ore_items = jnp.array(
        [ItemType.COAL, ItemType.IRON, ItemType.COPPER], dtype=jnp.int32
    )
    mined_delta = jnp.sum(
        new_state.items_mined[ore_items] - prev_state.items_mined[ore_items]
    )
    mining_bonus: jax.Array = 5.0 * mined_delta.astype(jnp.float32)

    return proximity + mining_bonus


def sparse_mining_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Sparse reward of 1.0 for each ore item mined during this step.

    Counts the total delta across coal, iron, and copper in ``items_mined``
    between the two states.  This signal is zero on every step where nothing
    is extracted, which makes it harder to shape behaviour but trivial to
    interpret: one unit of reward per one unit of ore.

    Args:
        prev_state: State immediately before the step.
        new_state: State immediately after the step.
        params: Environment parameters (unused; present for interface uniformity).

    Returns:
        Scalar float32 reward.
    """
    ore_items = jnp.array(
        [ItemType.COAL, ItemType.IRON, ItemType.COPPER], dtype=jnp.int32
    )
    delta = jnp.sum(
        new_state.items_mined[ore_items] - prev_state.items_mined[ore_items]
    )
    reward: jax.Array = delta.astype(jnp.float32)
    return reward


def sparse_chest_crafting_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Sparse reward of 1.0 for each chest gained via crafting.

    Detects crafting by requiring that the player's inventory gained
    chests *and* lost iron in the same step. Moving chests between
    inventory and machines changes chest count without consuming iron,
    so place/pickup/deposit/withdraw exploits yield zero reward.

    Args:
        prev_state: State immediately before the step.
        new_state: State immediately after the step.
        params: Environment parameters (unused; present for interface uniformity).

    Returns:
        Scalar float32 reward.
    """

    def _count_item(state: EnvState, item: int) -> jax.Array:
        is_item = state.inventory_items == item
        return jnp.sum(jnp.where(is_item, state.inventory_counts, 0))

    chest_delta = _count_item(new_state, ItemType.CHEST) - _count_item(
        prev_state, ItemType.CHEST
    )
    iron_delta = _count_item(new_state, ItemType.IRON) - _count_item(
        prev_state, ItemType.IRON
    )

    # Crafting consumes iron and produces chests in the same step.
    is_craft = (chest_delta > 0) & (iron_delta < 0)
    reward: jax.Array = jnp.where(is_craft, chest_delta, 0).astype(
        jnp.float32
    )
    return reward


def chest_filling_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Reward of 1.0 for each item deposited into a chest machine.

    Counts the total increase in item counts across all inventory slots
    of chest-type machines.  This gives a dense signal for every
    successful deposit action rather than waiting for a full stack of 64.

    Args:
        prev_state: State immediately before the step.
        new_state: State immediately after the step.
        params: Environment parameters (unused; present for interface uniformity).

    Returns:
        Scalar float32 reward.
    """
    is_chest = new_state.machine_types == MachineType.CHEST
    chest_mask = is_chest[..., None]
    prev_counts = jnp.where(chest_mask, prev_state.machine_inventory_counts, 0)
    new_counts = jnp.where(chest_mask, new_state.machine_inventory_counts, 0)
    delta = jnp.sum(new_counts) - jnp.sum(prev_counts)
    reward: jax.Array = jnp.maximum(delta, 0).astype(jnp.float32)
    return reward
