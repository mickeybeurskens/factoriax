"""Game logic for player movement and environment stepping."""

import jax
import jax.numpy as jnp
from jax import lax

from factoriax.achievements import check_achievements
from factoriax.constants import (
    BLOCK_TO_ITEM_ARRAY,
    DIRECTIONS,
    MAX_STACK_SIZE,
    MINEABLE_BLOCKS,
    SOLID_BLOCKS,
    Action,
    BlockType,
    ItemType,
    MachineType,
)
from factoriax.crafting import cycle_recipe, cycle_slot, start_crafting, update_crafting
from factoriax.machines import update_all_machines
from factoriax.placement import place_machine
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

    A position is walkable if it's in bounds, not a solid block (water, etc),
    and does not have a machine on it.

    Args:
        state: Current environment state
        position: (x, y) coordinates to check

    Returns:
        Boolean indicating whether position is walkable
    """
    block = get_block_at(state, position)
    is_solid = jnp.any(block == SOLID_BLOCKS)

    map_height, map_width = state.map.shape
    in_bounds = is_position_in_bounds(position, map_width, map_height)
    has_machine = lax.cond(
        in_bounds,
        lambda: state.machine_types[position[1], position[0]] != MachineType.NONE,
        lambda: jnp.bool_(False),
    )

    return ~is_solid & ~has_machine


def move_player(
    state: EnvState, action: int | jax.Array, player_idx: int | jax.Array
) -> EnvState:
    """Attempt to move a player in the specified direction.

    The player will face the direction of movement regardless of whether
    the move succeeds. Movement only succeeds if the target tile is walkable.

    Args:
        state: Current environment state
        action: Action to take (from Action enum)
        player_idx: Index of the player to move

    Returns:
        Updated environment state with new player position and direction
    """
    current_position = state.player_positions[player_idx]
    current_direction = state.player_directions[player_idx]

    direction = DIRECTIONS[action]
    new_position = current_position + direction
    can_move = is_position_walkable(state, new_position)
    final_position = jnp.where(can_move, new_position, current_position)
    new_direction = lax.cond(
        action == Action.NOOP,
        lambda: current_direction,
        lambda: jnp.int32(action),
    )

    new_positions = state.player_positions.at[player_idx].set(final_position)
    new_directions = state.player_directions.at[player_idx].set(new_direction)

    return state.replace(  # type: ignore[attr-defined, no-any-return]
        player_positions=new_positions,
        player_directions=new_directions,
    )


def mine_block(state: EnvState, player_idx: int | jax.Array) -> EnvState:
    """Attempt to mine the block at a player's current position.

    Mining succeeds if:
    - The block is mineable (coal, iron, or copper)
    - The block has remaining resources
    - There is inventory space (existing stack with room or empty slot)

    On success, one resource is extracted and added to inventory. The block
    becomes dirt only when all resources are depleted.

    Args:
        state: Current environment state
        player_idx: Index of the player performing the mining

    Returns:
        Updated environment state with updated resources and inventory
    """
    player_pos = state.player_positions[player_idx]
    px, py = player_pos[0], player_pos[1]
    block_type = state.map[py, px]

    is_mineable = jnp.any(block_type == MINEABLE_BLOCKS)
    has_resources = state.block_resources[py, px] > 0
    item_type = BLOCK_TO_ITEM_ARRAY[block_type]

    player_inv_items = state.inventory_items[player_idx]
    player_inv_counts = state.inventory_counts[player_idx]

    matching_slot_mask = (player_inv_items == item_type) & (
        player_inv_counts < MAX_STACK_SIZE
    )
    has_matching_slot = jnp.any(matching_slot_mask)
    matching_slot_idx = jnp.argmax(matching_slot_mask)

    empty_slot_mask = player_inv_items == ItemType.EMPTY
    has_empty_slot = jnp.any(empty_slot_mask)
    empty_slot_idx = jnp.argmax(empty_slot_mask)

    can_stack = has_matching_slot
    can_use_empty = ~has_matching_slot & has_empty_slot
    can_add_to_inventory = can_stack | can_use_empty
    can_mine = is_mineable & has_resources & can_add_to_inventory

    slot_idx = jnp.where(can_stack, matching_slot_idx, empty_slot_idx)

    new_inventory_items = lax.cond(
        can_mine,
        lambda: state.inventory_items.at[player_idx, slot_idx].set(item_type),
        lambda: state.inventory_items,
    )

    new_inventory_counts = lax.cond(
        can_mine,
        lambda: state.inventory_counts.at[player_idx, slot_idx].set(
            player_inv_counts[slot_idx] + 1
        ),
        lambda: state.inventory_counts,
    )

    new_resources = state.block_resources[py, px] - 1
    is_depleted = new_resources <= 0

    new_block_resources = lax.cond(
        can_mine,
        lambda: state.block_resources.at[py, px].set(new_resources),
        lambda: state.block_resources,
    )

    new_map = lax.cond(
        can_mine & is_depleted,
        lambda: state.map.at[py, px].set(jnp.int32(BlockType.DIRT)),
        lambda: state.map,
    )

    new_items_mined = lax.cond(
        can_mine,
        lambda: state.items_mined.at[item_type].add(1),
        lambda: state.items_mined,
    )

    return state.replace(  # type: ignore[attr-defined, no-any-return]
        map=new_map,
        inventory_items=new_inventory_items,
        inventory_counts=new_inventory_counts,
        block_resources=new_block_resources,
        items_mined=new_items_mined,
    )


def _handle_player_action(
    state: EnvState, action: int | jax.Array, player_idx: int | jax.Array
) -> EnvState:
    """Handle a single player action.

    Dispatches to the appropriate handler based on action type.

    Args:
        state: Current environment state
        action: Action to take
        player_idx: Index of the player

    Returns:
        Updated environment state
    """
    is_mine = action == Action.MINE
    is_craft = action == Action.CRAFT
    is_place = action == Action.PLACE
    is_next_slot = action == Action.NEXT_SLOT
    is_prev_slot = action == Action.PREV_SLOT
    is_next_recipe = action == Action.NEXT_RECIPE
    is_prev_recipe = action == Action.PREV_RECIPE

    state = lax.cond(is_mine, lambda s: mine_block(s, player_idx), lambda s: s, state)
    state = lax.cond(
        is_craft, lambda s: start_crafting(s, player_idx), lambda s: s, state
    )
    state = lax.cond(
        is_place, lambda s: place_machine(s, player_idx), lambda s: s, state
    )
    state = lax.cond(
        is_next_slot, lambda s: cycle_slot(s, player_idx, 1), lambda s: s, state
    )
    state = lax.cond(
        is_prev_slot, lambda s: cycle_slot(s, player_idx, -1), lambda s: s, state
    )
    state = lax.cond(
        is_next_recipe, lambda s: cycle_recipe(s, player_idx, 1), lambda s: s, state
    )
    state = lax.cond(
        is_prev_recipe, lambda s: cycle_recipe(s, player_idx, -1), lambda s: s, state
    )

    is_movement = ~(
        is_mine
        | is_craft
        | is_place
        | is_next_slot
        | is_prev_slot
        | is_next_recipe
        | is_prev_recipe
    )
    state = lax.cond(
        is_movement, lambda s: move_player(s, action, player_idx), lambda s: s, state
    )

    return state


def factoriax_step(
    rng: jax.Array, state: EnvState, action: int | jax.Array, params: EnvParams
) -> EnvState:
    """Execute one step of the environment.

    Processes in order:

    1. Selected player action (move, mine, craft, place, or UI actions)
    2. Crafting progress for all players
    3. All machine updates
    4. Achievement tracking
    5. Timestep increment

    Non-selected players perform NOOP.  Reward computation is intentionally
    absent here; it is performed externally by a reward function from
    :mod:`factoriax.rewards`, which receives both the pre-step and
    post-step states.

    Args:
        rng: JAX random key (reserved for future stochastic mechanics).
        state: Current environment state.
        action: Action to take for the selected player.
        params: Environment parameters.

    Returns:
        Updated environment state.
    """
    player_idx = state.selected_player
    state = _handle_player_action(state, action, player_idx)
    state = update_crafting(state)
    state = update_all_machines(state)
    state = check_achievements(state)
    return state.replace(timestep=state.timestep + 1)  # type: ignore[attr-defined, no-any-return]


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
