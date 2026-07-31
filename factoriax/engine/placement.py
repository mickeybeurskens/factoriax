"""Put machines on the map, take them off again, and aim them.

Every function here acts on one tile: the tile that the player faces. A player
who carries a machine item turns that item into a machine on the tile. A player
can also turn the machine on that tile back into an item. Nothing in this
module runs on its own. Each function answers a player action from
:mod:`factoriax.engine.step`.

A placed machine lives in two places at the same time, and the two must agree.
``state.machine_types[y, x]`` gives the kind of machine on a tile.
``state.tile_entity[y, x]`` gives the entity slot that holds its contents, or
``-1`` for an empty tile. The slot itself carries the position, the facing, the
health, and the contents of the machine. Every function here that changes one
of the three changes all three.

The slots are a pool of fixed size, ``E`` of them, fixed when the level is
built. A slot is free when ``ent_y < 0``. :func:`place_machine` takes the free
slot with the lowest number, and :func:`pickup_machine` gives one back. A long
game therefore reuses slots and grows no array. A map with every slot in use
refuses a new placement, and the player keeps the item.

Every function uses a mask and not a branch, because these functions run under
``jit``. An action that cannot happen is not an error and raises nothing. It
writes the state back unchanged. A caller cannot separate a refused placement
from a placement that worked by the return value alone. It must compare the
state before and after the call.
"""

import jax
import jax.numpy as jnp

from factoriax.engine.constants import Machine
from factoriax.engine.state import EnvParams, EnvState
from factoriax.engine.tables import (
    DIRECTIONS,
    ITEM_TO_MACHINE_ARRAY,
    MACHINE_MAX_HEALTH,
    MACHINE_TO_ITEM_ARRAY,
    PLACEABLE_ITEMS,
    PLAYER_MAX_STACK,
    SOLID_BLOCKS,
)


def get_tile_in_front(
    state: EnvState,
    player_idx: int | jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return the tile one step ahead of a player, in the way they face.

    Every player action in this module targets this tile. A player never acts
    on the tile under them.

    The function does not clip the result, so the result can name a tile
    outside the map. A player on an edge who faces outward gets such a tile.
    Callers test the bounds themselves and mask. This function raises nothing.

    Parameters
    ----------
    state
        State to read. The function reads ``player_positions`` and
        ``player_directions``.
    player_idx
        Player to look ahead of. This value indexes the first axis of both
        player arrays.

    Returns
    -------
    tuple of jax.Array
        ``(tx, ty)``, the column first and the row second, in the ``(x, y)``
        order of ``player_positions``. A grid lookup takes them in the other
        order, as ``state.map[ty, tx]``. A player with the unset direction 0
        gets their own tile back, because row 0 of ``DIRECTIONS`` is a zero
        offset.
    """
    pos = state.player_positions[player_idx]
    direction = state.player_directions[player_idx]
    offset = DIRECTIONS[direction.astype(jnp.int32)]
    target = pos + offset.astype(jnp.int16)
    return target[0], target[1]


def is_placeable_item(item_type: int | jax.Array) -> jax.Array:
    """Report whether an item becomes a machine on the map.

    Most items are materials, and no action can place them.
    ``PLACEABLE_ITEMS`` lists the items that a player can place, and
    ``ITEM_TO_MACHINE_ARRAY`` maps each one to a machine kind.

    Parameters
    ----------
    item_type
        Item id to test. This must be a scalar. The function does not
        broadcast over an array of items, because the reduction collapses
        every axis.

    Returns
    -------
    jax.Array
        Scalar bool. True when the item appears in ``PLACEABLE_ITEMS``.
    """
    return jnp.any(item_type == PLACEABLE_ITEMS)


def is_valid_placement_tile(
    state: EnvState,
    tx: jax.Array,
    ty: jax.Array,
) -> jax.Array:
    """Report whether a machine can stand on a tile.

    Three conditions refuse a tile. The tile is outside the map, or its block
    is one of ``SOLID_BLOCKS`` such as stone or water, or a machine already
    stands on it. An ore block is not solid, so a player can place a miner on
    the patch that it works.

    The function ignores the players. A tile with another player on it passes
    this test, so a player can place a machine under another player.

    Parameters
    ----------
    state
        State to read. The function reads ``map`` and ``machine_types``.
    tx
        Target column. The value can fall outside the map, and this function
        exists to refuse that case.
    ty
        Target row. The value can fall outside the map.

    Returns
    -------
    jax.Array
        Bool, with the shape of ``tx``. True only when all three conditions
        pass.
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
    """Spend one carried item to stand a machine on the tile in front.

    Four conditions must all hold. The player carries one of the item or more,
    the item is placeable, the target tile is free, and a free entity slot
    exists. If one condition fails, the state comes back unchanged and the
    player keeps the item. The function does not report which condition failed.

    The new machine takes the facing of the player, so a player must stand in
    the correct direction before placement to aim a belt or an arm. The machine
    starts at the ``MACHINE_MAX_HEALTH`` of its kind, with an empty buffer.

    The function clears the buffer of the reused slot, and nothing else. It
    leaves the craft fields ``ent_power`` and the ``ent_asm_`` arrays as they
    were. This is safe only because :func:`pickup_machine` sets all of them to
    zero when it frees a slot.

    CAUTION: A new path that frees a slot must also clear those fields. Without
    that, a newly placed assembler starts with a half-finished craft.

    Parameters
    ----------
    state
        State to read.
    params
        The function does not read this argument. It is present so that the
        player-action functions in this module share one signature, and
        :mod:`factoriax.engine.step` can dispatch to them with no special case.
    player_idx
        Player that places the machine.
    item_type
        Item to spend. ``ITEM_TO_MACHINE_ARRAY`` turns it into a machine kind.

    Returns
    -------
    EnvState
        New state, with ``player_inventory``, ``machine_types``,
        ``tile_entity``, and the ``ent_y``, ``ent_x``, ``ent_type``,
        ``ent_direction``, ``ent_health``, ``ent_buf_type``, and
        ``ent_buf_count`` of the new slot updated. The state is unchanged when
        the function refuses the placement.
    """
    item_type = jnp.int32(item_type)
    player_count = state.player_inventory[player_idx, item_type]
    has_item = player_count > 0
    can_place = is_placeable_item(item_type)

    tx, ty = get_tile_in_front(state, player_idx)
    valid_tile = is_valid_placement_tile(state, tx, ty)

    # Find the first free entity slot, which holds ent_y < 0.
    max_e = state.ent_y.shape[0]
    free_mask = state.ent_y < 0
    # argmax gives the first True index, and 0 when no slot is free.
    free_idx = jnp.argmax(free_mask)
    has_slot = free_mask[free_idx]

    should_place = has_item & can_place & valid_tile & has_slot
    sx = jnp.clip(tx, 0, state.map.shape[1] - 1)
    sy = jnp.clip(ty, 0, state.map.shape[0] - 1)

    mt = ITEM_TO_MACHINE_ARRAY[item_type]
    direction = state.player_directions[player_idx]
    full_hp = MACHINE_MAX_HEALTH[mt]

    new_count = jnp.where(should_place, player_count - 1, player_count)
    new_mt = jnp.where(
        should_place,
        mt.astype(jnp.int8),
        state.machine_types[sy, sx],
    )

    # Claim the entity slot.
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
    # Clear the buffer of the reused slot, so a new machine never starts with
    # old contents left in a slot that was free.
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
    """Take the machine in front of the player back into the inventory.

    This function is the reverse of :func:`place_machine`. The machine becomes
    an item again, the tile empties, and the function clears the entity slot
    and returns it to the free pool for the next placement.

    Everything that the machine held comes back with it: ``ent_buf``, both
    ``ent_asm_in`` slots, and ``ent_asm_out``. A pickup of a working assembler
    therefore costs its progress, but none of its items.

    Three conditions gate the pickup. A machine stands on the tile, the machine
    is at full health for its kind, and the whole payout fits in the inventory
    of the player. No stack can pass ``PLAYER_MAX_STACK``.

    The room test is all or nothing. A payout of only the part that fits
    destroys the rest together with the machine. The function therefore refuses
    a pickup that passes a stack limit, and the player must withdraw some of
    the contents first. This is the one case where a full inventory blocks a
    pickup that all other conditions allow.

    Parameters
    ----------
    state
        State to read.
    params
        The function does not read this argument. It is present for the shared
        player-action signature.
    player_idx
        Player that picks the machine up.

    Returns
    -------
    EnvState
        New state, with ``player_inventory``, ``machine_types``,
        ``tile_entity``, and every ``ent_`` field of the freed slot updated.
        The state is unchanged when the function refuses the pickup.
    """
    tx, ty = get_tile_in_front(state, player_idx)
    h, w = state.map.shape
    in_bounds = (tx >= 0) & (tx < w) & (ty >= 0) & (ty < h)
    sx = jnp.clip(tx, 0, w - 1)
    sy = jnp.clip(ty, 0, h - 1)

    mt = jnp.where(in_bounds, state.machine_types[sy, sx], Machine.NONE)
    has_machine = mt != Machine.NONE
    machine_item = MACHINE_TO_ITEM_ARRAY[mt.astype(jnp.int32)]

    # Entity lookup for the target tile.
    max_e = state.ent_y.shape[0]
    eidx_raw = state.tile_entity[sy, sx]
    eidx = jnp.clip(eidx_raw, 0, max_e - 1)

    # Total the whole payout for each item type before any decision: the
    # machine item, and every slot in which the machine stores items. A
    # repeated index is safe here, because this adds and does not overwrite. An
    # assembler with the same item in both input slots therefore totals
    # correctly.
    num_items = state.player_inventory.shape[1]
    payout = jnp.zeros(num_items, dtype=jnp.int32)
    payout = payout.at[machine_item].add(jnp.int32(1))
    for slot_type, slot_count in (
        (state.ent_buf_type[eidx], state.ent_buf_count[eidx]),
        (state.ent_asm_in_type[eidx, 0], state.ent_asm_in_count[eidx, 0]),
        (state.ent_asm_in_type[eidx, 1], state.ent_asm_in_count[eidx, 1]),
        (state.ent_asm_out_type[eidx], state.ent_asm_out_count[eidx]),
    ):
        payout = payout.at[slot_type.astype(jnp.int32)].add(
            jnp.where(slot_count > 0, slot_count.astype(jnp.int32), jnp.int32(0)),
        )

    # All or nothing. A payout of only the part that fits destroys the rest
    # together with the machine, so a pickup that passes a stack limit fails
    # and the player must withdraw first.
    held = state.player_inventory[player_idx].astype(jnp.int32)
    fits = jnp.all(held + payout <= PLAYER_MAX_STACK.astype(jnp.int32))

    # Gate the pickup on full health for the type of the entity.
    target_type = state.ent_type[eidx]
    full_hp = MACHINE_MAX_HEALTH[target_type]
    is_full_health = state.ent_health[eidx] >= full_hp
    should_pickup = in_bounds & has_machine & fits & is_full_health

    new_inv = state.player_inventory.at[player_idx].add(
        jnp.where(should_pickup, payout, jnp.int32(0)).astype(
            state.player_inventory.dtype,
        ),
    )

    # Clear the grid.
    cur_mt = state.machine_types[sy, sx]
    new_machine_types = state.machine_types.at[sy, sx].set(
        jnp.where(should_pickup, jnp.int8(Machine.NONE), cur_mt),
    )
    new_tile_entity = jnp.where(
        should_pickup,
        state.tile_entity.at[sy, sx].set(jnp.int16(-1)),
        state.tile_entity,
    )

    # Free the entity slot: set ent_y to -1 and clear every other field.
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

    The base engine treats REPAIR as a full restore with no item cost. A
    wrapper that wants another rule, such as a repair for each tick, an item
    cost, or a partial restore, rewrites
    :data:`~factoriax.engine.constants.Action.REPAIR` to
    :data:`~factoriax.engine.constants.Action.NOOP` before it calls
    ``step_env``. The wrapper then applies its own update to
    ``state.ent_health``.

    The function does nothing when the target tile is outside the map, when the
    tile holds no entity, or when the entity is already at full health for its
    type.

    No code path in the engine lowers ``ent_health``. Placement sets it to full
    and pickup clears it to zero, so in the base engine every placed machine is
    already at full health. This function therefore has nothing to restore. A
    caller can reach it, but it changes nothing. ``ISSUES.md`` records this.

    Parameters
    ----------
    state
        State to read.
    params
        The function does not read this argument. It is present for the shared
        player-action signature.
    player_idx
        Player that repairs the machine.

    Returns
    -------
    EnvState
        New state, with ``ent_health`` raised to full on the target slot. The
        state is unchanged in every other case.
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
    full_hp = MACHINE_MAX_HEALTH[target_type]
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
    """Aim the machine in front of the player at a given direction.

    A player turns a placed machine to route a belt line, and does not pick
    every piece up again. The function writes the direction directly and does
    not turn the machine by one step. A second call with the same argument
    therefore changes nothing.

    The facing of a machine means a different thing for each kind. It gives the
    tile that a miner, a belt, or a splitter pushes into, and the two tiles
    that an arm joins. On a ``CROSSING`` the one byte holds both axis
    directions, so a write aims the vertical flow and the horizontal flow
    together. ``CROSSING_AXIS_DIRS`` unpacks the pair.

    The function does nothing when the tile is outside the map or holds no
    machine.

    Parameters
    ----------
    state
        State to read.
    player_idx
        Player that aims the machine.
    target_dir
        ``Direction`` value to write. The function does not validate it. It
        stores a value outside 1 to 4 as given, and the passes that index a
        direction table with it are defined over 0 to 4 only.

    Returns
    -------
    EnvState
        New state, with ``ent_direction`` updated on the target slot. The state
        is unchanged in every other case.
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
