"""Machine placement system for the FactoriaX environment.

Uses the pouch inventory model: placement actions name the specific
item type to place, and pickup transfers all machine inventory counts
directly into the player's pouch.
"""

import jax
import jax.numpy as jnp
from jax import lax

from factoriax.constants import (
    DEFAULT_MACHINE_MAX_HEALTH,
    DIRECTIONS,
    ITEM_TO_MACHINE_ARRAY,
    MACHINE_TO_ITEM_ARRAY,
    MAX_UNDERGROUND_RANGE,
    NUM_ITEM_TYPES,
    PLACEABLE_ITEMS,
    PLAYER_MAX_STACK,
    SOLID_BLOCKS,
    Direction,
    ItemType,
    MachineType,
)
from factoriax.state import EnvState


def get_tile_in_front(
    state: EnvState,
    player_idx: int | jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Get the coordinates of the tile in front of a player.

    Args:
        state: Current environment state.
        player_idx: Index of the player.

    Returns:
        Tuple of (x, y) coordinates of the tile in front.
    """
    pos = state.player_positions[player_idx]
    direction = state.player_directions[player_idx]
    d = jnp.clip(direction, 0, DIRECTIONS.shape[0] - 1)
    offset = DIRECTIONS[d]
    return pos[0] + offset[0], pos[1] + offset[1]


def is_valid_placement_tile(
    state: EnvState,
    x: int | jax.Array,
    y: int | jax.Array,
) -> jax.Array:
    """Check if a tile is valid for machine placement.

    Args:
        state: Current environment state.
        x: X coordinate of the tile.
        y: Y coordinate of the tile.

    Returns:
        Boolean indicating if placement is valid.
    """
    map_height, map_width = state.map.shape
    in_bounds = (x >= 0) & (x < map_width) & (y >= 0) & (y < map_height)
    block_type = jnp.where(in_bounds, state.map[y, x], -1)
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
        item_type: Item type to check.

    Returns:
        Boolean indicating if the item can be placed.
    """
    return jnp.isin(item_type, PLACEABLE_ITEMS)


def _determine_underground_role(
    state: EnvState,
    tx: int | jax.Array,
    ty: int | jax.Array,
    direction: int | jax.Array,
) -> jax.Array:
    """Determine whether to place an entry or exit underground belt.

    Scans backward (opposite to *direction*) up to
    :data:`MAX_UNDERGROUND_RANGE` tiles looking for an unpaired
    :data:`UNDERGROUND_ENTRY` facing the same direction. "Unpaired"
    means no :data:`UNDERGROUND_EXIT` exists between the entry and
    the current tile.

    Args:
        state: Current environment state.
        tx: Target tile x.
        ty: Target tile y.
        direction: Facing direction of the placed belt.

    Returns:
        :data:`MachineType.UNDERGROUND_EXIT` if an unpaired entry was
        found, otherwise :data:`MachineType.UNDERGROUND_ENTRY`.
    """
    h, w = state.machine_types.shape
    dx = DIRECTIONS[direction, 0]
    dy = DIRECTIONS[direction, 1]

    found_entry = jnp.bool_(False)

    for offset in range(1, MAX_UNDERGROUND_RANGE + 1):
        bx = jnp.clip(jnp.int32(tx) - dx * offset, 0, w - 1)
        by = jnp.clip(jnp.int32(ty) - dy * offset, 0, h - 1)

        is_entry = state.machine_types[by, bx] == MachineType.UNDERGROUND_ENTRY
        same_dir = state.machine_direction[by, bx] == direction
        is_exit = state.machine_types[by, bx] == MachineType.UNDERGROUND_EXIT

        # If we hit an exit first, no unpaired entry behind it.
        found_entry = found_entry & ~is_exit
        # If we hit an entry with matching direction, mark as found.
        found_entry = found_entry | (is_entry & same_dir)

    return jnp.where(
        found_entry,
        jnp.int32(MachineType.UNDERGROUND_EXIT),
        jnp.int32(MachineType.UNDERGROUND_ENTRY),
    )


def place_machine(
    state: EnvState,
    player_idx: int | jax.Array,
    item_type: int | jax.Array,
) -> EnvState:
    """Place a machine of the given item type.

    The item type comes from the compound action (e.g. PLACE_BELT maps
    to ItemType.CONVEYOR_BELT). Checks the player has the item and the
    tile in front is valid.

    Args:
        state: Current environment state.
        player_idx: Index of the player.
        item_type: ItemType of the machine to place.

    Returns:
        Updated state with machine placed (or unchanged if invalid).
    """
    item_type = jnp.int32(item_type)
    player_count = state.player_inventory[player_idx, item_type]

    has_item = player_count > 0
    can_place_item = is_placeable_item(item_type)

    target_x, target_y = get_tile_in_front(state, player_idx)
    valid_tile = is_valid_placement_tile(state, target_x, target_y)

    should_place = has_item & can_place_item & valid_tile

    default_machine_type = ITEM_TO_MACHINE_ARRAY[item_type]

    def do_place(s: EnvState) -> EnvState:
        direction = s.player_directions[player_idx]

        # Underground belts: determine entry vs exit from context.
        mt = jnp.where(
            item_type == int(ItemType.UNDERGROUND_BELT),
            _determine_underground_role(s, target_x, target_y, direction),
            default_machine_type,
        )

        # Arms face opposite to player for intuitive interaction.
        opposite = jnp.int32(0)
        opposite = jnp.where(
            direction == Direction.LEFT,
            Direction.RIGHT,
            opposite,
        )
        opposite = jnp.where(
            direction == Direction.RIGHT,
            Direction.LEFT,
            opposite,
        )
        opposite = jnp.where(
            direction == Direction.UP,
            Direction.DOWN,
            opposite,
        )
        opposite = jnp.where(
            direction == Direction.DOWN,
            Direction.UP,
            opposite,
        )
        placed_dir = jnp.where(
            mt == MachineType.ARM,
            opposite,
            direction,
        )

        new_inv = s.player_inventory.at[player_idx, item_type].set(
            player_count - 1,
        )

        return s.replace(
            player_inventory=new_inv,
            machine_types=s.machine_types.at[target_y, target_x].set(mt),
            machine_direction=s.machine_direction.at[target_y, target_x].set(
                placed_dir
            ),
            machine_health=s.machine_health.at[target_y, target_x].set(
                DEFAULT_MACHINE_MAX_HEALTH,
            ),
        )

    return lax.cond(should_place, do_place, lambda s: s, state)


def can_fit_in_player(
    player_counts: jax.Array,
    items_to_add: jax.Array,
) -> jax.Array:
    """Check if all items can fit into a player's pouch inventory.

    Simulates adding each item type's count to the player's inventory.
    Returns True only if every item fits within PLAYER_MAX_STACK.

    Args:
        player_counts: Player inventory counts, shape (NUM_ITEM_TYPES,).
        items_to_add: Counts to add per item type, shape (NUM_ITEM_TYPES,).

    Returns:
        Boolean indicating all items fit.
    """
    combined = player_counts + items_to_add
    within_limits = jnp.all(combined <= PLAYER_MAX_STACK)
    return within_limits


def pickup_machine(
    state: EnvState,
    player_idx: int | jax.Array,
) -> EnvState:
    """Pick up a machine from the tile in front of the player.

    Transfers all machine inventory contents into the player's pouch,
    then adds the machine item itself. No-op if the tile has no machine,
    is out of bounds, or the player lacks capacity.

    Args:
        state: Current environment state.
        player_idx: Index of the player.

    Returns:
        Updated state with machine removed (or unchanged if invalid).
    """
    target_x, target_y = get_tile_in_front(state, player_idx)
    map_height, map_width = state.map.shape

    in_bounds = (
        (target_x >= 0)
        & (target_x < map_width)
        & (target_y >= 0)
        & (target_y < map_height)
    )

    machine_type = jnp.where(
        in_bounds,
        state.machine_types[target_y, target_x],
        MachineType.NONE,
    )
    has_machine = machine_type != MachineType.NONE

    machine_item = MACHINE_TO_ITEM_ARRAY[machine_type]

    machine_inv = jnp.where(
        in_bounds,
        state.machine_inventory[target_y, target_x],
        jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.int16),
    ).astype(jnp.int32)

    # Items to add: machine contents + the machine item itself.
    items_to_add = machine_inv.at[machine_item].add(1)

    player_counts = state.player_inventory[player_idx]
    fits = can_fit_in_player(player_counts, items_to_add)

    should_pickup = in_bounds & has_machine & fits

    def do_pickup(s: EnvState) -> EnvState:
        new_player_inv = s.player_inventory[player_idx] + items_to_add
        s = s.replace(
            player_inventory=s.player_inventory.at[player_idx].set(
                new_player_inv,
            ),
            machine_types=s.machine_types.at[target_y, target_x].set(
                MachineType.NONE,
            ),
            machine_health=s.machine_health.at[target_y, target_x].set(0),
            machine_power=s.machine_power.at[target_y, target_x].set(0),
            machine_inventory=s.machine_inventory.at[target_y, target_x].set(
                jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.int16)
            ),
            machine_selected_recipe=s.machine_selected_recipe.at[
                target_y, target_x
            ].set(0),
            machine_direction=s.machine_direction.at[target_y, target_x].set(0),
        )
        return s

    return lax.cond(should_pickup, do_pickup, lambda s: s, state)
