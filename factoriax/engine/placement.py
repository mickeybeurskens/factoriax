"""Put machines on the map, take them off again, and aim them.

Everything here acts on one tile: the one the player faces. A player
carrying a machine item turns it into a machine standing on that tile, or
turns the machine standing there back into an item. Nothing in this module
runs on its own; each function answers a player action from
:mod:`factoriax.engine.step`.

A placed machine lives in two places at once and the pair has to agree.
``state.machine_types[y, x]`` says what kind of machine stands on a tile,
and ``state.tile_entity[y, x]`` says which entity slot holds its contents,
or ``-1`` for an empty tile. The slot itself carries position, facing,
health, and whatever the machine is holding. Every function here that
changes one of the three changes all three.

Slots are a fixed-size pool, ``E`` of them, decided when the level is built.
A slot is free when ``ent_y < 0``. :func:`place_machine` takes the
lowest-numbered free slot and :func:`pickup_machine` gives one back, so a
long game reuses slots rather than growing. A map with every slot taken
refuses further placement, and the player keeps the item.

Every function is masked rather than branched, because these run under
``jit``. An action that cannot happen is not an error and does not raise;
it writes the state back unchanged. A caller cannot tell a refused
placement from a successful one by the return value alone, only by looking
at what changed.
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

    Every player action in this module targets this tile. A player never
    acts on the tile they stand on.

    The result is not clipped and can name a tile off the map, which is what
    a player standing on an edge and facing outward gets. Callers test the
    bounds themselves and mask; nothing here raises.

    Parameters
    ----------
    state
        State to read. Uses ``player_positions`` and ``player_directions``.
    player_idx
        Which player to look ahead of. Indexes the leading axis of both
        player arrays.

    Returns
    -------
    tuple of jax.Array
        ``(tx, ty)``, column first and row second, matching the ``(x, y)``
        order of ``player_positions``. Grid lookups take them the other way
        round, as ``state.map[ty, tx]``. A player whose direction is the
        unset value 0 gets their own tile back, because ``DIRECTIONS`` row 0
        is a zero offset.
    """
    pos = state.player_positions[player_idx]
    direction = state.player_directions[player_idx]
    offset = DIRECTIONS[direction.astype(jnp.int32)]
    target = pos + offset.astype(jnp.int16)
    return target[0], target[1]


def is_placeable_item(item_type: int | jax.Array) -> jax.Array:
    """Report whether an item turns into a machine when placed.

    Most items are materials and cannot be placed. The placeable ones are
    listed in ``PLACEABLE_ITEMS``, and each maps to a machine kind through
    ``ITEM_TO_MACHINE_ARRAY``.

    Parameters
    ----------
    item_type
        Item id to test. A scalar; this does not broadcast over an array of
        candidates, because the reduction collapses every axis.

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

    Three things disqualify a tile: it is off the map, its block is one of
    ``SOLID_BLOCKS`` such as stone or water, or a machine already stands
    there. Ore blocks are not solid, which is what lets a miner be placed on
    the patch it works.

    Players are not considered. A tile another player stands on passes this
    test, so a machine can be placed underneath them.

    Parameters
    ----------
    state
        State to read. Uses ``map`` and ``machine_types``.
    tx
        Target column. May be out of bounds; that is one of the cases this
        function exists to reject.
    ty
        Target row. May be out of bounds.

    Returns
    -------
    jax.Array
        Bool, shaped like ``tx``. True only when all three conditions pass.
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

    Four things all have to hold: the player carries at least one of the
    item, the item is placeable, the target tile is free, and a free entity
    slot exists. When any fails the state comes back unchanged and the
    player keeps the item. Nothing reports which one failed.

    The new machine faces the way the player was facing, so aiming a belt or
    an arm means standing the right way round before placing it. It starts
    at ``MACHINE_MAX_HEALTH`` for its kind and with an empty buffer.

    Only the buffer is cleared on the reused slot. The crafting fields,
    ``ent_power`` and the ``ent_asm_`` arrays, are left as they were, which
    is safe only because :func:`pickup_machine` zeroes all of them when it
    frees a slot. A future path that frees a slot some other way has to
    clear them too, or a newly placed assembler inherits a half-finished
    craft.

    Parameters
    ----------
    state
        State to read.
    params
        Unused. Present so the player-action functions in this module share
        one signature and :mod:`factoriax.engine.step` can dispatch to them
        without special-casing.
    player_idx
        Which player is placing.
    item_type
        Item to spend. Turned into a machine kind by
        ``ITEM_TO_MACHINE_ARRAY``.

    Returns
    -------
    EnvState
        New state with ``player_inventory``, ``machine_types``,
        ``tile_entity``, and the placed slot's ``ent_y``, ``ent_x``,
        ``ent_type``, ``ent_direction``, ``ent_health``, ``ent_buf_type``,
        and ``ent_buf_count`` updated. Unchanged when the placement is
        refused.
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
    full_hp = MACHINE_MAX_HEALTH[mt]

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
    """Take the machine in front of the player back into the inventory.

    The reverse of :func:`place_machine`. The machine becomes an item again,
    the tile empties, and the entity slot is cleared and returned to the
    free pool for the next placement to claim.

    Everything the machine was holding comes back with it: ``ent_buf``, both
    ``ent_asm_in`` slots, and ``ent_asm_out``. Picking up a working assembler
    therefore costs its progress but none of its items.

    Three conditions gate it: a machine stands on the tile, the machine is at
    full health for its kind, and the whole payout fits in the player's
    inventory without pushing any stack past ``PLAYER_MAX_STACK``.

    The room check is all or nothing. Paying out only what fits would destroy
    the remainder along with the machine, so a pickup that would overflow is
    refused outright and the player has to withdraw some of the contents
    first. That is the one case where a full inventory blocks a pickup that
    would otherwise succeed.

    Parameters
    ----------
    state
        State to read.
    params
        Unused. Present for the shared player-action signature.
    player_idx
        Which player is picking up.

    Returns
    -------
    EnvState
        New state with ``player_inventory``, ``machine_types``,
        ``tile_entity``, and every ``ent_`` field of the freed slot updated.
        Unchanged when the pickup is refused.
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

    # Total the whole payout per item type before deciding anything: the
    # machine item plus every slot the machine stores items in. Repeated
    # indices are fine here because this accumulates rather than overwrites,
    # which is what lets an assembler holding the same item in both input
    # slots total correctly.
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

    # All or nothing. Paying out only what fits would destroy the rest along
    # with the machine, so a pickup that would overflow any stack is refused
    # and the player is left to withdraw first.
    held = state.player_inventory[player_idx].astype(jnp.int32)
    fits = jnp.all(held + payout <= PLAYER_MAX_STACK.astype(jnp.int32))

    # Gate pickup on full health for the entity's type.
    target_type = state.ent_type[eidx]
    full_hp = MACHINE_MAX_HEALTH[target_type]
    is_full_health = state.ent_health[eidx] >= full_hp
    should_pickup = in_bounds & has_machine & fits & is_full_health

    new_inv = state.player_inventory.at[player_idx].add(
        jnp.where(should_pickup, payout, jnp.int32(0)).astype(
            state.player_inventory.dtype,
        ),
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
    or the entity is already at full health for its type.

    No code path in the engine lowers ``ent_health``. Placement sets it
    full and pickup clears it to zero, so in the base engine every placed
    machine is already at full health and this function has nothing to
    restore. It is reachable but its effect is not. Recorded in
    ``ISSUES.md``.

    Parameters
    ----------
    state
        State to read.
    params
        Unused. Present for the shared player-action signature.
    player_idx
        Which player is repairing.

    Returns
    -------
    EnvState
        New state with ``ent_health`` raised to full on the targeted slot.
        Unchanged in every other case.
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

    Turning a placed machine is how a belt line is routed without picking
    every piece up again. The direction is set outright, not rotated by a
    step, so repeating the action with the same argument changes nothing.

    A machine's facing means different things per kind: which tile a miner,
    belt, or splitter pushes into, and which two tiles an arm bridges. On a
    ``CROSSING`` the one byte packs both axis directions at once, so setting
    it re-aims the vertical and horizontal flows together. ``CROSSING_AXIS_DIRS``
    unpacks the pair.

    No-op when the tile is out of bounds or holds no machine.

    Parameters
    ----------
    state
        State to read.
    player_idx
        Which player is aiming.
    target_dir
        ``Direction`` value to write. Not validated: a value outside 1 to 4
        is stored as given, and the passes that index a direction table with
        it are only defined over 0 to 4.

    Returns
    -------
    EnvState
        New state with ``ent_direction`` updated on the targeted slot.
        Unchanged in every other case.
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
