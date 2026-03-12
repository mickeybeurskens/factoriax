"""Game logic for player movement and environment stepping."""

import jax
import jax.numpy as jnp
from jax import lax

from factoriax.constants import (
    BLOCK_TO_ITEM_ARRAY,
    DIRECTIONS,
    MAX_STACK_SIZE,
    MINEABLE_BLOCKS,
    SOLID_BLOCKS,
    Action,
    BlockType,
    ItemType,
)
from factoriax.state import EnvParams, EnvState


def is_position_in_bounds(
    position: jax.Array, map_width: int, map_height: int
) -> jax.Array:
    """Check if a position is within map boundaries.

    Args:
        position: (x, y) coordinates to check
        map_width: Width of the map
        map_height: Height of the map

    Returns:
        Boolean indicating whether position is in bounds
    """
    x, y = position[0], position[1]
    return (x >= 0) & (x < map_width) & (y >= 0) & (y < map_height)


def get_block_at(state: EnvState, position: jax.Array) -> jax.Array:
    """Get the block type at a position, returning OUT_OF_BOUNDS if outside map.

    Args:
        state: Current environment state
        position: (x, y) coordinates to query

    Returns:
        Block type at the position
    """
    map_height, map_width = state.map.shape
    in_bounds = is_position_in_bounds(position, map_width, map_height)
    block: jax.Array = lax.cond(
        in_bounds,
        lambda: state.map[position[1], position[0]],
        lambda: jnp.int32(BlockType.OUT_OF_BOUNDS),
    )
    return block


def is_position_walkable(state: EnvState, position: jax.Array) -> jax.Array:
    """Check if a position can be walked on.

    A position is walkable if it's in bounds and not a solid block (water, etc).

    Args:
        state: Current environment state
        position: (x, y) coordinates to check

    Returns:
        Boolean indicating whether position is walkable
    """
    block = get_block_at(state, position)
    is_solid = jnp.any(block == SOLID_BLOCKS)
    return ~is_solid


def move_player(state: EnvState, action: int | jax.Array) -> EnvState:
    """Attempt to move the player in the specified direction.

    The player will face the direction of movement regardless of whether
    the move succeeds. Movement only succeeds if the target tile is walkable.

    Args:
        state: Current environment state
        action: Action to take (from Action enum)

    Returns:
        Updated environment state with new player position and direction
    """
    direction = DIRECTIONS[action]
    new_position = state.player_position + direction
    can_move = is_position_walkable(state, new_position)
    final_position = jnp.where(can_move, new_position, state.player_position)
    new_direction = lax.cond(
        action == Action.NOOP,
        lambda: state.player_direction,
        lambda: action,
    )

    return state.replace(  # type: ignore[attr-defined, no-any-return]
        player_position=final_position,
        player_direction=new_direction,
    )


def mine_block(state: EnvState) -> EnvState:
    """Attempt to mine the block at the player's current position.

    Mining succeeds if:
    - The block is mineable (coal, iron, or copper)
    - There is inventory space (existing stack with room or empty slot)

    On success, the block becomes dirt and the item is added to inventory.

    Args:
        state: Current environment state

    Returns:
        Updated environment state with mined block and updated inventory
    """
    px, py = state.player_position[0], state.player_position[1]
    block_type = state.map[py, px]

    is_mineable = jnp.any(block_type == MINEABLE_BLOCKS)
    item_type = BLOCK_TO_ITEM_ARRAY[block_type]

    matching_slot_mask = (state.inventory_items == item_type) & (
        state.inventory_counts < MAX_STACK_SIZE
    )
    has_matching_slot = jnp.any(matching_slot_mask)
    matching_slot_idx = jnp.argmax(matching_slot_mask)

    empty_slot_mask = state.inventory_items == ItemType.EMPTY
    has_empty_slot = jnp.any(empty_slot_mask)
    empty_slot_idx = jnp.argmax(empty_slot_mask)

    can_stack = has_matching_slot
    can_use_empty = ~has_matching_slot & has_empty_slot
    can_add_to_inventory = can_stack | can_use_empty
    can_mine = is_mineable & can_add_to_inventory

    slot_idx = jnp.where(can_stack, matching_slot_idx, empty_slot_idx)

    new_inventory_items = lax.cond(
        can_mine,
        lambda: state.inventory_items.at[slot_idx].set(item_type),
        lambda: state.inventory_items,
    )

    new_inventory_counts = lax.cond(
        can_mine,
        lambda: state.inventory_counts.at[slot_idx].set(
            state.inventory_counts[slot_idx] + 1
        ),
        lambda: state.inventory_counts,
    )

    new_map = lax.cond(
        can_mine,
        lambda: state.map.at[py, px].set(jnp.int32(BlockType.DIRT)),
        lambda: state.map,
    )

    return state.replace(  # type: ignore[attr-defined, no-any-return]
        map=new_map,
        inventory_items=new_inventory_items,
        inventory_counts=new_inventory_counts,
    )


def factoriax_step(
    rng: jax.Array, state: EnvState, action: int | jax.Array, params: EnvParams
) -> tuple[EnvState, float]:
    """Execute one step of the environment.

    Args:
        rng: JAX random key (unused for now, but included for interface consistency)
        state: Current environment state
        action: Action to take
        params: Environment parameters

    Returns:
        Tuple of (new_state, reward)
    """
    state = lax.cond(
        action == Action.MINE,
        lambda: mine_block(state),
        lambda: move_player(state, action),
    )
    state = state.replace(timestep=state.timestep + 1)  # type: ignore[attr-defined]
    reward = 0.0
    return state, reward


def is_game_over(state: EnvState, params: EnvParams) -> jax.Array:
    """Check if the episode has ended.

    Currently only checks if max timesteps has been reached.

    Args:
        state: Current environment state
        params: Environment parameters

    Returns:
        Boolean indicating whether the game is over
    """
    result: jax.Array = jnp.bool_(state.timestep >= params.max_timesteps)
    return result
