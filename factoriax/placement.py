"""Machine placement system for the FactoriaX environment."""

import jax
import jax.numpy as jnp
from jax import lax

from factoriax.constants import (
    DIRECTION_OFFSETS,
    ITEM_TO_MACHINE_ARRAY,
    PLACEABLE_ITEMS,
    SOLID_BLOCKS,
    ItemType,
    MachineType,
)
from factoriax.state import EnvState


def get_tile_in_front(
    state: EnvState, player_idx: int | jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Get the coordinates of the tile in front of a player.

    Uses the player's current direction to determine which tile is "in front".

    Args:
        state: Current environment state
        player_idx: Index of the player

    Returns:
        Tuple of (x, y) coordinates of the tile in front
    """
    pos = state.player_positions[player_idx]
    direction = state.player_directions[player_idx]

    dx, dy = 0, 0
    for action, (ox, oy) in DIRECTION_OFFSETS.items():
        dx = jnp.where(direction == action, ox, dx)
        dy = jnp.where(direction == action, oy, dy)

    target_x = pos[0] + dx
    target_y = pos[1] + dy

    return target_x, target_y


def is_valid_placement_tile(
    state: EnvState, x: int | jax.Array, y: int | jax.Array
) -> jax.Array:
    """Check if a tile is valid for machine placement.

    A tile is valid if it is:
    - Within map bounds
    - Not a solid block (water, out of bounds)
    - Does not already have a machine

    Args:
        state: Current environment state
        x: X coordinate of the tile
        y: Y coordinate of the tile

    Returns:
        Boolean indicating if placement is valid
    """
    map_height, map_width = state.map.shape

    in_bounds = (x >= 0) & (x < map_width) & (y >= 0) & (y < map_height)

    block_type = jnp.where(
        in_bounds,
        state.map[y, x],
        -1,
    )
    is_solid = jnp.isin(block_type, SOLID_BLOCKS)

    has_machine = jnp.where(
        in_bounds,
        state.machine_types[y, x] != MachineType.NONE,
        True,
    )

    return in_bounds & ~is_solid & ~has_machine


def is_placeable_item(item_type: int | jax.Array) -> jax.Array:
    """Check if an item type is a placeable machine.

    Args:
        item_type: Item type to check

    Returns:
        Boolean indicating if the item can be placed as a machine
    """
    return jnp.isin(item_type, PLACEABLE_ITEMS)


def place_machine(state: EnvState, player_idx: int | jax.Array) -> EnvState:
    """Place a machine from the player's selected inventory slot.

    Checks if the selected slot contains a placeable item and if the tile
    in front of the player is valid for placement. If both conditions are
    met, removes the item from inventory and places the machine.

    Args:
        state: Current environment state
        player_idx: Index of the player

    Returns:
        Updated state with machine placed (or unchanged if invalid)
    """
    selected_slot = state.selected_slots[player_idx]
    item_type = state.inventory_items[player_idx, selected_slot]
    item_count = state.inventory_counts[player_idx, selected_slot]

    has_item = item_count > 0
    can_place_item = is_placeable_item(item_type)

    target_x, target_y = get_tile_in_front(state, player_idx)
    valid_tile = is_valid_placement_tile(state, target_x, target_y)

    should_place = has_item & can_place_item & valid_tile

    machine_type = ITEM_TO_MACHINE_ARRAY[item_type]

    def do_place(s: EnvState) -> EnvState:
        new_count = item_count - 1
        new_item = jnp.where(new_count == 0, ItemType.EMPTY, item_type)
        direction = s.player_directions[player_idx]

        s = s.replace(
            inventory_items=s.inventory_items.at[player_idx, selected_slot].set(
                new_item
            ),
            inventory_counts=s.inventory_counts.at[player_idx, selected_slot].set(
                new_count
            ),
            machine_types=s.machine_types.at[target_y, target_x].set(machine_type),
            machine_direction=s.machine_direction.at[target_y, target_x].set(
                direction
            ),
        )
        return s

    return lax.cond(should_place, do_place, lambda s: s, state)
