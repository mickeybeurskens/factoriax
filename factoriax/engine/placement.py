"""Machine placement and pickup for the FactoriaX environment.

Placement consumes an item from the player's inventory and creates a
machine on the tile in front. Pickup reverses this, returning the
machine item plus any buffer contents to the player.
"""

import jax
import jax.numpy as jnp

from factoriax.engine.constants import Machine
from factoriax.engine.state import EnvParams, EnvState
from factoriax.engine.tables import (
    DIRECTIONS,
    ITEM_TO_MACHINE_ARRAY,
    MACHINE_TO_ITEM_ARRAY,
    PLACEABLE_ITEMS,
    PLAYER_MAX_STACK,
    SOLID_BLOCKS,
)


def get_tile_in_front(
    state: EnvState,
    player_idx: int | jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Get the tile coordinates in front of a player.

    Args:
        state: Current environment state.
        player_idx: Player index.

    Returns:
        Tuple of (x, y) coordinates.
    """
    pos = state.player_positions[player_idx]
    direction = state.player_directions[player_idx]
    offset = DIRECTIONS[direction.astype(jnp.int32)]
    target = pos + offset.astype(jnp.int16)
    return target[0], target[1]


def is_placeable_item(item_type: int | jax.Array) -> jax.Array:
    """Check if an item type can be placed as a machine.

    Args:
        item_type: ItemType to check.

    Returns:
        Boolean scalar.
    """
    return jnp.any(item_type == PLACEABLE_ITEMS)


def is_valid_placement_tile(
    state: EnvState,
    tx: jax.Array,
    ty: jax.Array,
) -> jax.Array:
    """Check if a tile is valid for machine placement.

    Args:
        state: Current environment state.
        tx: Target tile x.
        ty: Target tile y.

    Returns:
        Boolean: valid for placement.
    """
    h, w = state.map.shape
    in_bounds = (tx >= 0) & (tx < w) & (ty >= 0) & (ty < h)
    sx = jnp.clip(tx, 0, w - 1)
    sy = jnp.clip(ty, 0, h - 1)
    block = state.map[sy, sx]
    is_solid = jnp.any(block == SOLID_BLOCKS)
    has_machine = state.machine_types[sy, sx] != Machine.NONE
    return in_bounds & ~is_solid & ~has_machine


def place_machine(
    state: EnvState,
    params: EnvParams,
    player_idx: int | jax.Array,
    item_type: int | jax.Array,
) -> EnvState:
    """Place a machine on the tile in front of the player.

    Allocates an entity slot for the new machine and updates both the
    grid (machine_types, tile_entity) and entity arrays. The new
    entity is initialized to its configured maximum health
    (``params.machine_config.max_health[machine_type]``).

    Args:
        state: Current environment state.
        params: Environment parameters; supplies the per-type max
            health used to seed ``ent_health``.
        player_idx: Player index.
        item_type: ItemType of the machine to place.

    Returns:
        Updated state with machine placed (or unchanged).
    """
    item_type = jnp.int32(item_type)
    player_count = state.player_inventory[player_idx, item_type]
    has_item = player_count > 0
    can_place = is_placeable_item(item_type)

    tx, ty = get_tile_in_front(state, player_idx)
    valid_tile = is_valid_placement_tile(state, tx, ty)

    # Find the first inactive entity slot (ent_y < 0).
    max_e = state.ent_y.shape[0]
    free_mask = state.ent_y < 0
    # argmax returns first True index; if none free, returns 0.
    free_idx = jnp.argmax(free_mask)
    has_slot = free_mask[free_idx]

    should_place = has_item & can_place & valid_tile & has_slot
    sx = jnp.clip(tx, 0, state.map.shape[1] - 1)
    sy = jnp.clip(ty, 0, state.map.shape[0] - 1)

    mt = ITEM_TO_MACHINE_ARRAY[item_type]
    direction = state.player_directions[player_idx]
    full_hp = params.machine_config.max_health[mt]

    new_count = jnp.where(should_place, player_count - 1, player_count)
    new_mt = jnp.where(
        should_place,
        mt.astype(jnp.int8),
        state.machine_types[sy, sx],
    )

    # Allocate entity slot.
    safe_idx = jnp.clip(free_idx, 0, max_e - 1)
    new_ent_y = jnp.where(
        should_place,
        state.ent_y.at[safe_idx].set(sy.astype(jnp.int16)),
        state.ent_y,
    )
    new_ent_x = jnp.where(
        should_place,
        state.ent_x.at[safe_idx].set(sx.astype(jnp.int16)),
        state.ent_x,
    )
    new_ent_type = jnp.where(
        should_place,
        state.ent_type.at[safe_idx].set(mt.astype(jnp.int8)),
        state.ent_type,
    )
    new_ent_dir = jnp.where(
        should_place,
        state.ent_direction.at[safe_idx].set(
            direction.astype(jnp.int8),
        ),
        state.ent_direction,
    )
    new_ent_health = jnp.where(
        should_place,
        state.ent_health.at[safe_idx].set(full_hp.astype(jnp.int16)),
        state.ent_health,
    )
    # Clear the reused slot's buffer so a new machine never inherits
    # stale contents left in a previously-inactive slot.
    new_ent_buf_type = jnp.where(
        should_place,
        state.ent_buf_type.at[safe_idx].set(jnp.int8(0)),
        state.ent_buf_type,
    )
    new_ent_buf_count = jnp.where(
        should_place,
        state.ent_buf_count.at[safe_idx].set(jnp.int16(0)),
        state.ent_buf_count,
    )
    new_tile_entity = jnp.where(
        should_place,
        state.tile_entity.at[sy, sx].set(safe_idx.astype(jnp.int16)),
        state.tile_entity,
    )

    return state.replace(
        player_inventory=state.player_inventory.at[player_idx, item_type].set(
            new_count.astype(jnp.int16)
        ),
        machine_types=state.machine_types.at[sy, sx].set(new_mt),
        tile_entity=new_tile_entity,
        ent_y=new_ent_y,
        ent_x=new_ent_x,
        ent_type=new_ent_type,
        ent_direction=new_ent_dir,
        ent_health=new_ent_health,
        ent_buf_type=new_ent_buf_type,
        ent_buf_count=new_ent_buf_count,
    )


def pickup_machine(
    state: EnvState,
    params: EnvParams,
    player_idx: int | jax.Array,
) -> EnvState:
    """Pick up the machine in front of the player.

    Returns the machine item and any buffer contents to the player.
    Deactivates the entity slot and clears tile_entity.

    Pickup is gated on the target being at full health for its type;
    damaged machines must be repaired (or destroyed by a wrapper)
    before they can be picked up. On success, the freed entity slot's
    ``ent_health`` is cleared to ``0``.

    Args:
        state: Current environment state.
        params: Environment parameters; supplies the per-type max
            health used for the full-HP gate.
        player_idx: Player index.

    Returns:
        Updated state with machine removed (or unchanged).
    """
    tx, ty = get_tile_in_front(state, player_idx)
    h, w = state.map.shape
    in_bounds = (tx >= 0) & (tx < w) & (ty >= 0) & (ty < h)
    sx = jnp.clip(tx, 0, w - 1)
    sy = jnp.clip(ty, 0, h - 1)

    mt = jnp.where(in_bounds, state.machine_types[sy, sx], Machine.NONE)
    has_machine = mt != Machine.NONE
    machine_item = MACHINE_TO_ITEM_ARRAY[mt.astype(jnp.int32)]

    # Check player has space for the machine item.
    player_count = state.player_inventory[player_idx, machine_item]
    player_max = PLAYER_MAX_STACK[machine_item]
    fits = player_count < player_max

    # Entity lookup for the target tile.
    max_e = state.ent_y.shape[0]
    eidx_raw = state.tile_entity[sy, sx]
    eidx = jnp.clip(eidx_raw, 0, max_e - 1)

    # Gate pickup on full health for the entity's type.
    target_type = state.ent_type[eidx]
    full_hp = params.machine_config.max_health[target_type]
    is_full_health = state.ent_health[eidx] >= full_hp
    should_pickup = in_bounds & has_machine & fits & is_full_health

    # Return machine item to player.
    new_inv = state.player_inventory.at[player_idx, machine_item].add(
        jnp.where(should_pickup, jnp.int16(1), jnp.int16(0)),
    )
    # Also return buffer contents.
    buf_item = state.ent_buf_type[eidx]
    buf_count = state.ent_buf_count[eidx]
    has_buf = (buf_count > 0) & should_pickup
    new_inv = new_inv.at[player_idx, buf_item.astype(jnp.int32)].add(
        jnp.where(has_buf, buf_count, jnp.int16(0)),
    )

    # Clear grid.
    cur_mt = state.machine_types[sy, sx]
    new_machine_types = state.machine_types.at[sy, sx].set(
        jnp.where(should_pickup, jnp.int8(Machine.NONE), cur_mt),
    )
    new_tile_entity = jnp.where(
        should_pickup,
        state.tile_entity.at[sy, sx].set(jnp.int16(-1)),
        state.tile_entity,
    )

    # Deactivate entity slot (set ent_y to -1, clear all fields).
    new_ent_y = jnp.where(
        should_pickup,
        state.ent_y.at[eidx].set(jnp.int16(-1)),
        state.ent_y,
    )
    new_ent_x = jnp.where(
        should_pickup,
        state.ent_x.at[eidx].set(jnp.int16(-1)),
        state.ent_x,
    )
    new_ent_type = jnp.where(
        should_pickup,
        state.ent_type.at[eidx].set(jnp.int8(0)),
        state.ent_type,
    )
    new_ent_dir = jnp.where(
        should_pickup,
        state.ent_direction.at[eidx].set(jnp.int8(0)),
        state.ent_direction,
    )
    new_ent_power = jnp.where(
        should_pickup,
        state.ent_power.at[eidx].set(jnp.int16(0)),
        state.ent_power,
    )
    new_ent_buf_type = jnp.where(
        should_pickup,
        state.ent_buf_type.at[eidx].set(jnp.int8(0)),
        state.ent_buf_type,
    )
    new_ent_buf_count = jnp.where(
        should_pickup,
        state.ent_buf_count.at[eidx].set(jnp.int16(0)),
        state.ent_buf_count,
    )
    new_asm_in_type = jnp.where(
        should_pickup,
        state.ent_asm_in_type.at[eidx].set(
            jnp.zeros(2, dtype=jnp.int8),
        ),
        state.ent_asm_in_type,
    )
    new_asm_in_count = jnp.where(
        should_pickup,
        state.ent_asm_in_count.at[eidx].set(
            jnp.zeros(2, dtype=jnp.int16),
        ),
        state.ent_asm_in_count,
    )
    new_asm_out_type = jnp.where(
        should_pickup,
        state.ent_asm_out_type.at[eidx].set(jnp.int8(0)),
        state.ent_asm_out_type,
    )
    new_asm_out_count = jnp.where(
        should_pickup,
        state.ent_asm_out_count.at[eidx].set(jnp.int16(0)),
        state.ent_asm_out_count,
    )
    new_ent_health = jnp.where(
        should_pickup,
        state.ent_health.at[eidx].set(jnp.int16(0)),
        state.ent_health,
    )

    return state.replace(
        player_inventory=new_inv,
        machine_types=new_machine_types,
        tile_entity=new_tile_entity,
        ent_y=new_ent_y,
        ent_x=new_ent_x,
        ent_type=new_ent_type,
        ent_direction=new_ent_dir,
        ent_power=new_ent_power,
        ent_buf_type=new_ent_buf_type,
        ent_buf_count=new_ent_buf_count,
        ent_asm_in_type=new_asm_in_type,
        ent_asm_in_count=new_asm_in_count,
        ent_asm_out_type=new_asm_out_type,
        ent_asm_out_count=new_asm_out_count,
        ent_health=new_ent_health,
    )


def apply_repair(
    state: EnvState,
    params: EnvParams,
    player_idx: int | jax.Array,
) -> EnvState:
    """Restore the entity in front of the player to full health.

    The base engine implements REPAIR as a full restore with no item
    cost. Wrappers that want a different policy (per-tick repair,
    inventory consumption, partial restore) pre-empt
    :data:`~factoriax.engine.constants.Action.REPAIR` by rewriting it to
    :data:`~factoriax.engine.constants.Action.NOOP` before calling
    ``step_env`` and then applying their own update to
    ``state.ent_health``.

    No-op when the target tile is out of bounds, contains no entity,
    or the entity is already at full health for its type. The
    function is JIT-compatible and pure: it never raises.

    Args:
        state: Current environment state.
        params: Environment parameters; supplies per-type max health.
        player_idx: Player index.

    Returns:
        Updated state with the target entity's health restored, or
        unchanged when no valid target exists.
    """
    tx, ty = get_tile_in_front(state, player_idx)
    h, w = state.map.shape
    in_bounds = (tx >= 0) & (tx < w) & (ty >= 0) & (ty < h)
    sx = jnp.clip(tx, 0, w - 1)
    sy = jnp.clip(ty, 0, h - 1)

    max_e = state.ent_y.shape[0]
    eidx_raw = state.tile_entity[sy, sx]
    has_entity = in_bounds & (eidx_raw >= 0)
    eidx = jnp.clip(eidx_raw, 0, max_e - 1)

    target_type = state.ent_type[eidx]
    full_hp = params.machine_config.max_health[target_type]
    cur_hp = state.ent_health[eidx]
    needs_repair = cur_hp < full_hp
    should_repair = has_entity & needs_repair

    new_hp = jnp.where(should_repair, full_hp.astype(jnp.int16), cur_hp)
    new_ent_health = state.ent_health.at[eidx].set(new_hp)
    return state.replace(
        ent_health=jnp.where(should_repair, new_ent_health, state.ent_health),
    )


def set_machine_direction(
    state: EnvState,
    player_idx: int | jax.Array,
    target_dir: int | jax.Array,
) -> EnvState:
    """Set the direction of the machine in front of the player.

    Sets the entity's direction to *target_dir* absolutely. No-op
    if the tile has no machine or is out of bounds.

    Args:
        state: Current environment state.
        player_idx: Player index.
        target_dir: Target Direction value to set.

    Returns:
        Updated state with the machine direction set (or unchanged).
    """
    tx, ty = get_tile_in_front(state, player_idx)
    h, w = state.map.shape
    in_bounds = (tx >= 0) & (tx < w) & (ty >= 0) & (ty < h)
    sx = jnp.clip(tx, 0, w - 1)
    sy = jnp.clip(ty, 0, h - 1)

    mt = jnp.where(
        in_bounds,
        state.machine_types[sy, sx],
        Machine.NONE,
    )
    has_machine = mt != Machine.NONE
    should_set = in_bounds & has_machine

    max_e = state.ent_y.shape[0]
    eidx = jnp.clip(state.tile_entity[sy, sx], 0, max_e - 1)

    old_dir = state.ent_direction[eidx]
    new_dir = jnp.int8(target_dir)
    updated_dirs = state.ent_direction.at[eidx].set(
        jnp.where(should_set, new_dir, old_dir),
    )
    return state.replace(
        ent_direction=jnp.where(
            should_set,
            updated_dirs,
            state.ent_direction,
        ),
    )
