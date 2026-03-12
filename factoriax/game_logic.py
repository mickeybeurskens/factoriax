"""Game logic for player movement and environment stepping."""

import jax
import jax.numpy as jnp
from jax import lax

from factoriax.constants import DIRECTIONS, SOLID_BLOCKS, Action, BlockType
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
    block = lax.cond(
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

    return state.replace(
        player_position=final_position,
        player_direction=new_direction,
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
    state = move_player(state, action)
    state = state.replace(timestep=state.timestep + 1)
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
    return state.timestep >= params.max_timesteps
