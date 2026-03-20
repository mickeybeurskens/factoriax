"""Machine placement system for the FactoriaX environment."""

import jax
import jax.numpy as jnp
from jax import lax

from factoriax.constants import (
    DIRECTION_OFFSETS,
    ITEM_TO_MACHINE_ARRAY,
    MACHINE_TO_ITEM_ARRAY,
    MAX_MACHINE_INVENTORY_SLOTS,
    MAX_STACK_SIZE,
    NUM_INVENTORY_SLOTS,
    PLACEABLE_ITEMS,
    SOLID_BLOCKS,
    Action,
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

        # Arms deposit forward and pick from behind.  Flipping the
        # direction so the output faces the player makes placement
        # intuitive: face a source machine, place the arm, and it
        # grabs from what you are looking at.
        opposite = jnp.int32(Action.NOOP)
        opposite = jnp.where(direction == Action.LEFT, Action.RIGHT, opposite)
        opposite = jnp.where(direction == Action.RIGHT, Action.LEFT, opposite)
        opposite = jnp.where(direction == Action.UP, Action.DOWN, opposite)
        opposite = jnp.where(direction == Action.DOWN, Action.UP, opposite)
        placed_dir = jnp.where(
            machine_type == MachineType.ARM, opposite, direction
        )

        s = s.replace(
            inventory_items=s.inventory_items.at[player_idx, selected_slot].set(
                new_item
            ),
            inventory_counts=s.inventory_counts.at[player_idx, selected_slot].set(
                new_count
            ),
            machine_types=s.machine_types.at[target_y, target_x].set(machine_type),
            machine_direction=s.machine_direction.at[target_y, target_x].set(
                placed_dir
            ),
        )
        return s

    return lax.cond(should_place, do_place, lambda s: s, state)


def can_fit_in_inventory(
    player_items: jax.Array,
    player_counts: jax.Array,
    items_to_add: jax.Array,
    counts_to_add: jax.Array,
) -> jax.Array:
    """Check if all item stacks can fit into a player's inventory.

    Simulates adding each item stack sequentially using the same stacking
    logic as add_item_to_inventory. Returns True only if every item fits
    with zero remainder.

    Args:
        player_items: Current inventory item types, shape (NUM_INVENTORY_SLOTS,)
        player_counts: Current inventory counts, shape (NUM_INVENTORY_SLOTS,)
        items_to_add: Item types to add, shape (N,)
        counts_to_add: Counts to add per item, shape (N,)

    Returns:
        Boolean indicating all items fit
    """

    def simulate_add_one(
        carry: tuple[jax.Array, jax.Array, jax.Array],
        idx: jax.Array,
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], None]:
        inv_items, inv_counts, all_fit = carry
        item_type = items_to_add[idx]
        amount = counts_to_add[idx]
        is_nonempty = (item_type != ItemType.EMPTY) & (amount > 0)

        def add_to_inv(
            state: tuple[jax.Array, jax.Array, jax.Array],
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            inv_i, inv_c, remaining = state

            def add_to_slot(
                carry: tuple[jax.Array, jax.Array, jax.Array],
                slot_idx: jax.Array,
            ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], None]:
                i_arr, c_arr, rem = carry
                slot_item = i_arr[slot_idx]
                slot_count = c_arr[slot_idx]

                can_stack = (slot_item == item_type) & (slot_count < MAX_STACK_SIZE)
                is_empty = slot_item == ItemType.EMPTY

                space = jnp.where(can_stack, MAX_STACK_SIZE - slot_count, 0)
                space = jnp.where(is_empty, MAX_STACK_SIZE, space)
                to_add = jnp.minimum(rem, space)

                new_count = slot_count + to_add
                new_rem = rem - to_add

                i_arr = jnp.where(
                    is_empty & (to_add > 0),
                    i_arr.at[slot_idx].set(item_type),
                    i_arr,
                )
                c_arr = c_arr.at[slot_idx].set(new_count)

                return (i_arr, c_arr, new_rem), None

            (inv_i, inv_c, remaining), _ = lax.scan(
                add_to_slot,
                (inv_i, inv_c, remaining),
                jnp.arange(NUM_INVENTORY_SLOTS),
            )
            return inv_i, inv_c, remaining

        inv_items, inv_counts, leftover = lax.cond(
            is_nonempty,
            add_to_inv,
            lambda s: (s[0], s[1], jnp.int32(0)),
            (inv_items, inv_counts, amount),
        )
        all_fit = all_fit & (leftover == 0)
        return (inv_items, inv_counts, all_fit), None

    num_items = items_to_add.shape[0]
    (_, _, all_fit), _ = lax.scan(
        simulate_add_one,
        (player_items, player_counts, jnp.bool_(True)),
        jnp.arange(num_items),
    )
    return all_fit


def pickup_machine(state: EnvState, player_idx: int | jax.Array) -> EnvState:
    """Pick up a machine from the tile in front of the player.

    Removes the machine from the tile, transfers all its inventory contents
    into the player's inventory, and returns the machine item itself to the
    player. The pickup is a no-op if:
    - The tile in front has no machine
    - The tile is out of bounds
    - The player lacks inventory space for all contents plus the machine item

    Args:
        state: Current environment state
        player_idx: Index of the player

    Returns:
        Updated state with machine removed (or unchanged if invalid)
    """
    from factoriax.crafting import add_item_to_inventory

    target_x, target_y = get_tile_in_front(state, player_idx)
    map_height, map_width = state.map.shape

    in_bounds = (
        (target_x >= 0)
        & (target_x < map_width)
        & (target_y >= 0)
        & (target_y < map_height)
    )

    machine_type = jnp.where(
        in_bounds, state.machine_types[target_y, target_x], MachineType.NONE
    )
    has_machine = machine_type != MachineType.NONE

    machine_item = MACHINE_TO_ITEM_ARRAY[machine_type]

    machine_inv_items = jnp.where(
        in_bounds,
        state.machine_inventory_items[target_y, target_x],
        jnp.zeros(MAX_MACHINE_INVENTORY_SLOTS, dtype=jnp.int32),
    )
    machine_inv_counts = jnp.where(
        in_bounds,
        state.machine_inventory_counts[target_y, target_x],
        jnp.zeros(MAX_MACHINE_INVENTORY_SLOTS, dtype=jnp.int16),
    )

    all_items = jnp.concatenate(
        [machine_inv_items, jnp.array([machine_item], dtype=jnp.int32)]
    )
    all_counts = jnp.concatenate([machine_inv_counts, jnp.array([1], dtype=jnp.int32)])

    player_items = state.inventory_items[player_idx]
    player_counts = state.inventory_counts[player_idx]
    fits = can_fit_in_inventory(player_items, player_counts, all_items, all_counts)

    should_pickup = in_bounds & has_machine & fits

    def do_pickup(s: EnvState) -> EnvState:
        def transfer_slot(s: EnvState, slot_idx: jax.Array) -> tuple[EnvState, None]:
            item = machine_inv_items[slot_idx]
            count = machine_inv_counts[slot_idx].astype(jnp.int32)
            has_item = (item != ItemType.EMPTY) & (count > 0)
            s = lax.cond(
                has_item,
                lambda st: add_item_to_inventory(st, player_idx, item, count),
                lambda st: st,
                s,
            )
            return s, None

        s, _ = lax.scan(transfer_slot, s, jnp.arange(MAX_MACHINE_INVENTORY_SLOTS))

        s = add_item_to_inventory(s, player_idx, machine_item, 1)

        s = s.replace(
            machine_types=s.machine_types.at[target_y, target_x].set(MachineType.NONE),
            machine_power=s.machine_power.at[target_y, target_x].set(0),
            machine_inventory_items=s.machine_inventory_items.at[
                target_y, target_x
            ].set(jnp.zeros(MAX_MACHINE_INVENTORY_SLOTS, dtype=jnp.int32)),
            machine_inventory_counts=s.machine_inventory_counts.at[
                target_y, target_x
            ].set(jnp.zeros(MAX_MACHINE_INVENTORY_SLOTS, dtype=jnp.int16)),
            machine_selected_recipe=s.machine_selected_recipe.at[
                target_y, target_x
            ].set(0),
            machine_selected_slot=s.machine_selected_slot.at[target_y, target_x].set(0),
            machine_direction=s.machine_direction.at[target_y, target_x].set(0),
        )
        return s

    return lax.cond(should_pickup, do_pickup, lambda s: s, state)
