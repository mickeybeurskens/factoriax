"""Move the factory forward by one step.

A factory is a set of machines on tiles that pass items to each other. This
module performs those passes. Ore leaves the ground, items become other items,
and both move between machines that do not touch. Every object on the map that
acts without a player acts here.

Four passes do the work. :func:`update_all_machines` runs them in order, and a
step calls that function:

:func:`run_miners`
    Ore comes out of the ground and goes into the machine that the miner faces.
:func:`run_assemblers`
    Assemblers and furnaces turn input items into an output item, over time.
:func:`run_conveyor_belts`
    Belts, splitters, and crossings carry items across the map.
:func:`run_arms`
    Arms lift one item between two machines one tile apart.

A pass takes a state and returns a new state. A machine runs whether or not a
player is near it. Another module handles the player actions.

Machine state lives in the ``ent_`` arrays of
:class:`~factoriax.engine.state.EnvState`. An entity id indexes them, not a
tile. There are ``E`` slots, fixed when the level is built, and a machine holds
one slot for as long as it stands on the map. An entity finds its neighbours
through ``state.tile_entity``. That array maps a tile back to the entity id on
it, or to ``-1`` for an empty tile.

Every pass runs over all ``E`` slots at the same time, free slots included.
Under ``jit`` there is no per-entity branch to skip a slot, so a free slot
computes a neighbour, a transfer, and a receipt like every other slot. A mask
makes this safe, not a skip. Code keeps a result only where
``active = state.ent_y >= 0``.

CAUTION: Put the ``active`` mask on every line that keeps a gathered result. A
free slot holds ``ent_y < 0``, and the position clip folds that onto tile
(0, 0). The slot then reads as a neighbour of the machine next to that corner.
One line without the mask raises nothing, and an item appears in a slot that
holds no machine.

Two results follow. First, the cost of a step follows ``E`` and the map shape,
and not the number of machines that a player built. An empty factory therefore
costs the same as a full one. Second, items move by scatter-gather. The sending
side computes what it pushes. The receiving side reaches back and computes the
same decision from its own position. Neither side writes into the slot of the
other.
"""

import jax.numpy as jnp

from factoriax.engine.constants import (
    BlockType,
    ItemType,
    Machine,
)
from factoriax.engine.state import EnvParams, EnvState
from factoriax.engine.tables import (
    BLOCK_TO_ITEM_ARRAY,
    CROSSING_AXIS_DIRS,
    CROSSING_HORIZ_SLOT,
    CROSSING_VERT_SLOT,
    MACHINE_HAS_INPUT_SLOTS,
    MACHINE_MAX_STACK,
    SPLITTER_PERP_OUTPUTS,
)

# Row and column step for each ``Direction`` value, so ``_DY[d], _DX[d]`` moves
# one tile in direction ``d``. Index 0 is the unset direction and does not move.
_DY: tuple[int, ...] = (0, 0, 0, -1, 1)
_DX: tuple[int, ...] = (0, -1, 1, 0, 0)

#: Ore that a miner mines into its own buffer. With the default
#: ``miner_mining_rate`` a miner reaches this value in one step. It then stays
#: idle until an arm, a belt, or a player empties the buffer. This value is not
#: the capacity of the buffer. ``MACHINE_MAX_STACK`` gives a miner much more
#: room, and a push from a belt or an arm fills the buffer past this value. A
#: pallet is still the only large storage.
MINER_OUTPUT_CAP: int = 3


def _subtract_buffer(
    cond: jnp.ndarray,
    buf_type: jnp.ndarray,
    buf_count: jnp.ndarray,
    amt: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Take ``amt`` items out of a buffer slot wherever ``cond`` holds.

    Every transfer in this module has a side that pays, and this function is
    that side. The item type is the reason to share it. A slot that reaches
    zero must also clear its type, because the rest of the engine reads a
    cleared type as "free to accept any item". A count that drops with no type
    change leaves the slot committed to an item that it no longer holds.

    The function works elementwise over the entity slots and clamps nothing. A
    subtraction of more than the slot holds leaves a negative count and an old
    type. Every caller therefore passes an ``amt`` that the source count
    already limits.

    Parameters
    ----------
    cond
        Slots to subtract from. Shape ``(E,)``, bool.
    buf_type
        Item id in each slot. Shape ``(E,)``, int8.
    buf_count
        Items in each slot. Shape ``(E,)``, int16.
    amt
        Amount to remove from each slot. Shape ``(E,)``, int16. The function
        reads it only where ``cond`` holds.

    Returns
    -------
    tuple of jnp.ndarray
        The new ``(buf_type, buf_count)`` pair. Slots outside ``cond`` come
        back unchanged.
    """
    new_c = jnp.where(cond, buf_count - amt, buf_count)
    return jnp.where(cond & (new_c == 0), jnp.int8(0), buf_type), new_c


def _lookup_neighbor(
    ey: jnp.ndarray,
    ex: jnp.ndarray,
    dy: int,
    dx: int,
    h: int,
    w: int,
    tile_entity: jnp.ndarray,
    n: int,
) -> tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
    """Find the entity one tile away in a fixed direction, for every entity.

    Every pass in this module needs the identity of the machine next to an
    entity. This function answers that for all entities at the same time. It
    also returns the flags that a caller needs to separate a real neighbour
    from an artefact of the question.

    The artefacts come from the question itself, which this function asks for
    every slot. Some entities sit on an edge of the map, and some hold no
    machine. A branch cannot remove either case, so both get an answer and a
    flag.

    The clip pulls a tile past the edge back onto the map, so an entity on an
    edge finds itself. ``diff`` is False in exactly that case. A caller puts
    ``diff`` in its transfer mask, so a machine on an edge does not push into
    its own slot. ``safe`` is the entity index, clipped into range for the same
    reason. A caller gathers with ``safe`` at all times and drops the result
    later.

    Parameters
    ----------
    ey
        Row of every entity, already clipped to the map. Shape ``(E,)``.
    ex
        Column of every entity, already clipped to the map. Shape ``(E,)``.
    dy
        Row offset for the step. This is a Python int, so the direction loop of
        the caller unrolls at trace time.
    dx
        Column offset for the step.
    h
        Map height in tiles.
    w
        Map width in tiles.
    tile_entity
        ``state.tile_entity``: the entity id on each tile, ``-1`` where the
        tile holds no machine. Shape ``(H, W)``.
    n
        Highest entity index that exists. The function uses it as the clip
        bound for the gather index. Callers pass ``E - 1``.

    Returns
    -------
    tuple of jnp.ndarray
        ``(ny, nx, eidx, valid, diff, safe)``, each with the shape of ``ey``.
        ``ny`` and ``nx`` are the clipped neighbour tile. ``eidx`` is the
        entity on that tile, ``-1`` for an empty tile. ``valid`` is
        ``eidx >= 0``. ``diff`` is False when the clip folded the neighbour
        back onto ``(ey, ex)``. ``safe`` is ``eidx``, clipped into ``[0, n]``,
        so a gather stays in bounds. What that gather reads has no meaning
        unless ``valid`` and ``diff`` both hold.
    """
    ny = jnp.clip(ey + dy, 0, h - 1)
    nx = jnp.clip(ex + dx, 0, w - 1)
    eidx = tile_entity[ny, nx]
    return ny, nx, eidx, eidx >= 0, (ny != ey) | (nx != ex), jnp.clip(eidx, 0, n)


def update_all_machines(
    state: EnvState,
    params: EnvParams,
) -> EnvState:
    """Run every machine on the map for one step.

    A step calls this function. It runs the four passes in one fixed order:
    miners, then assemblers and furnaces, then the belt network, then arms.

    The order sets the distance that an item can travel in one step. A miner
    pushes ore into a pallet, and an arm can lift that ore back out in the same
    step, because the arms run last. An arm drops an item onto a belt, and that
    item waits for the next step, because the belt pass already ran. A new
    order changes throughput, so the order is part of the contract and not an
    implementation detail.

    This function does not run the science labs.
    ``factoriax.engine.step.run_labs`` empties their input slots as a separate
    part of the full step.

    Parameters
    ----------
    state
        State to advance.
    params
        Supplies ``miner_mining_rate`` and ``recipe_table``.

    Returns
    -------
    EnvState
        New state with the entity arrays, ``map``, ``block_resources``, and
        ``items_mined`` advanced by one step.
    """
    state = run_miners(state, params)
    state = run_assemblers(state, params)
    state = run_conveyor_belts(state, params)
    state = run_arms(state, params)
    return state


def run_miners(
    state: EnvState,
    params: EnvParams,
) -> EnvState:
    """Extract ore under every miner and push the result one tile forward.

    A miner is where items enter the world. Every machine after a miner only
    moves or changes what the miners produce.

    A miner works the tile under it, not a tile next to it. A player puts it on
    the ore patch, with its facing towards the machine that receives the
    output. In each step it takes ``params.miner_mining_rate`` ore and pushes
    its buffer into the machine in front of it. A miner therefore holds two
    decisions: its tile and its facing.

    Two limits apply to the amount mined: the ore left in the tile, and the
    room left under :data:`MINER_OUTPUT_CAP`. That limit is small, so a miner
    that nothing empties fills its buffer and then stays idle. The machine in
    front limits the push, and not the mining. A miner in front of a full
    pallet therefore still takes ore out of the ground, and then stops with the
    ore in its buffer.

    A tile with no ore left becomes ``BlockType.DIRT``, and that miner stops.
    Other tiles of the same patch can still hold ore. A whole patch therefore
    needs one miner for each tile.

    The push needs a real machine in front. That machine must be of a kind that
    holds items, and it must hold nothing or the same item. A miner in front of
    open ground, or in front of a machine with no room, keeps its ore and
    stops. A miner pushes in one direction only, the direction that it faces.

    Parameters
    ----------
    state
        State to read.
    params
        Supplies ``miner_mining_rate``.

    Returns
    -------
    EnvState
        New state, with ``map``, ``block_resources``, ``ent_buf_type``,
        ``ent_buf_count``, and ``items_mined`` updated. ``items_mined`` counts
        coal, iron ore, copper ore, tin ore, and silicon only. A miner still
        mines and buffers ore of any other item id, but the total ignores it.
    """
    h, w = state.map.shape
    active = state.ent_y >= 0
    is_miner = (state.ent_type == Machine.MINER) & active

    # Gather grid data at the miner positions, clipped so the read is safe.
    ey = jnp.clip(state.ent_y, 0, h - 1)
    ex = jnp.clip(state.ent_x, 0, w - 1)
    resources = state.block_resources[ey, ex]
    block_item = BLOCK_TO_ITEM_ARRAY[state.map[ey, ex].astype(jnp.int32)]

    has_resources = resources > 0
    buf_empty = state.ent_buf_count == 0
    buf_same = state.ent_buf_type == block_item.astype(jnp.int8)
    buf_ok = buf_empty | buf_same
    has_space = state.ent_buf_count < jnp.int16(MINER_OUTPUT_CAP)

    can_mine = is_miner & has_resources & buf_ok & has_space
    mine_amt = jnp.where(
        can_mine,
        jnp.int16(params.miner_mining_rate),
        jnp.int16(0),
    )
    mine_amt = jnp.minimum(mine_amt, resources)
    # Clamp to the free output space, for miners only. Every other kind already
    # holds mine_amt=0, and can hold more than MINER_OUTPUT_CAP.
    slot_left = jnp.where(
        is_miner,
        jnp.int16(MINER_OUTPUT_CAP) - state.ent_buf_count,
        jnp.int16(0),
    )
    mine_amt = jnp.minimum(mine_amt, jnp.maximum(slot_left, jnp.int16(0)))
    mined = mine_amt > 0

    # Update entity state.
    new_buf_type = jnp.where(
        mined,
        block_item.astype(jnp.int8),
        state.ent_buf_type,
    )
    new_buf_count = state.ent_buf_count + mine_amt

    # Scatter the updates back to the grid: block_resources and map.
    new_resources = state.block_resources.at[ey, ex].add(
        jnp.where(mined, -mine_amt, jnp.int16(0)),
    )
    is_depleted = (new_resources[ey, ex] <= 0) & mined
    # Add a delta, and do not set the value. Every free slot clips to tile
    # (0, 0) and adds a second, stale write on that index, and duplicate
    # scatter indices resolve in an unspecified order. A lane with ore left
    # contributes zero.
    to_dirt = jnp.int8(int(BlockType.DIRT)) - state.map[ey, ex]
    new_map = state.map.at[ey, ex].add(
        jnp.where(is_depleted, to_dirt, jnp.int8(0)),
    )

    # Total the mined items.
    mined_flat = jnp.zeros(len(ItemType), dtype=jnp.int32)
    for item_id in (
        int(ItemType.COAL),
        int(ItemType.IRON_ORE),
        int(ItemType.COPPER_ORE),
        int(ItemType.TIN_ORE),
        int(ItemType.SILICON),
    ):
        mined_flat = mined_flat.at[item_id].set(
            jnp.sum(
                jnp.where(
                    (block_item == item_id) & mined,
                    mine_amt.astype(jnp.int32),
                    0,
                )
            ),
        )

    # --- Push the buffer to the neighbour in the facing direction ---
    n = new_buf_type.shape[0] - 1
    for d in range(1, 5):
        dy, dx = _DY[d], _DX[d]
        facing_d = (state.ent_direction == d) & is_miner

        _, _, dn_eidx, dn_valid, dn_diff, dn_safe = _lookup_neighbor(
            ey, ex, dy, dx, h, w, state.tile_entity, n
        )
        dn_bc = new_buf_count[dn_safe]
        dn_bt = new_buf_type[dn_safe]
        dn_max = MACHINE_MAX_STACK[state.ent_type[dn_safe].astype(jnp.int32)]
        dn_empty = dn_bc == 0
        dn_same = dn_bt == new_buf_type
        has_buf = new_buf_count > 0

        can_push = (
            facing_d
            & has_buf
            & dn_valid
            & dn_diff
            & (dn_max > 0)
            & (dn_empty | (dn_same & (dn_bc < dn_max)))
        )
        xfer = jnp.where(can_push, new_buf_count, jnp.int16(0))
        xfer = jnp.minimum(xfer, dn_max - dn_bc)

        # Gather: each entity tests whether a miner behind it, in direction
        # -d, pushes towards it.
        _, _, up_eidx, _, up_diff, up_safe = _lookup_neighbor(
            ey, ex, -dy, -dx, h, w, state.tile_entity, n
        )
        # Gate on an active receiver. Every free slot clips to tile (0, 0), so
        # without this gate each one takes a copy of the push aimed at the
        # real neighbour of (0, 0). The item then reaches slots that a later
        # placement uses.
        incoming = active & can_push[up_safe] & (up_eidx >= 0) & up_diff
        new_buf_type = jnp.where(incoming, new_buf_type[up_safe], new_buf_type)
        new_buf_count = jnp.where(
            incoming, new_buf_count + xfer[up_safe], new_buf_count
        )

        new_buf_type, new_buf_count = _subtract_buffer(
            can_push, new_buf_type, new_buf_count, xfer
        )

    return state.replace(
        map=new_map,
        block_resources=new_resources,
        ent_buf_type=new_buf_type,
        ent_buf_count=new_buf_count,
        items_mined=state.items_mined + mined_flat,
    )


def run_arms(state: EnvState, params: EnvParams) -> EnvState:
    """Move one item through every arm, from the tile behind to the one ahead.

    Machines do not reach into each other. An arm connects two of them. It is
    also the only way to empty a machine that does not push on its own, such as
    an assembler or a pallet.

    An arm joins the two tiles on each side of it. In each step it takes one
    item from the machine behind it, on the tile opposite to its facing, and
    gives that item to the machine in front. One arm moves one item in one
    step, and it needs both ends. With no source, or with no room in front,
    nothing moves.

    The machine behind decides which slot the item comes from. A finished craft
    stays in ``ent_asm_out`` and blocks the next craft, so an arm empties that
    slot first and keeps an assembler or a furnace at work. An arm empties the
    ``ent_buf`` of a machine that only stores, such as a miner, a pallet, or a
    belt.

    The machine in front decides which slot the item lands in. An assembler, a
    furnace, or a science lab takes the delivery into ``ent_asm_in``. It uses
    slot 0 when that slot is empty or already holds the same item, and slot 1
    in every other case. An input slot fills to the ``MACHINE_MAX_STACK`` of
    the machine, the same limit that the buffer route uses. An arm that feeds a
    machine which consumes nothing therefore stops, and the slot does not grow
    without bound. Every other machine receives into ``ent_buf``, also up to
    ``MACHINE_MAX_STACK``.

    An arm and a player deposit are the two ways to give a science lab a pack
    that it consumes. ``factoriax.engine.step.run_labs`` reads ``ent_asm_in``
    and nothing else, and both routes write there. A belt can point at a lab,
    and nothing refuses it. That belt delivers into the ``ent_buf`` of the lab,
    where the items stay until an arm or a player takes them back out.

    An arm never takes from its own ``ent_buf``, but that buffer is not closed.
    ``MACHINE_MAX_STACK`` gives an arm room for one item, and a belt that faces
    an arm pushes into it. The next arm treats a loaded arm as an ordinary
    source and empties it like any other machine.

    Parameters
    ----------
    state
        State to read.
    params
        The function does not read this argument. It is present so that every
        pass in this module takes the same pair, and
        :func:`update_all_machines` can call them one after the other.

    Returns
    -------
    EnvState
        New state, with ``ent_buf_type``, ``ent_buf_count``,
        ``ent_asm_in_type``, ``ent_asm_in_count``, ``ent_asm_out_type``, and
        ``ent_asm_out_count`` updated.
    """
    h, w = state.map.shape
    active = state.ent_y >= 0
    is_arm = (state.ent_type == Machine.ARM) & active
    # Assemblers, furnaces, and science labs take a delivery into
    # ``ent_asm_in``. One table decides that in every path: see
    # ``MACHINE_HAS_INPUT_SLOTS``.
    self_has_input_slots = MACHINE_HAS_INPUT_SLOTS[state.ent_type.astype(jnp.int32)]
    self_max_stack = MACHINE_MAX_STACK[state.ent_type.astype(jnp.int32)]

    ey = jnp.clip(state.ent_y, 0, h - 1)
    ex = jnp.clip(state.ent_x, 0, w - 1)

    buf_type = state.ent_buf_type
    buf_count = state.ent_buf_count
    out_type = state.ent_asm_out_type
    out_count = state.ent_asm_out_count
    # Per-slot scalars for the asm_in updates. One jnp.stack at the return
    # keeps the direction loop free of an allocation in each pass.
    in_t0 = state.ent_asm_in_type[..., 0]
    in_c0 = state.ent_asm_in_count[..., 0]
    in_t1 = state.ent_asm_in_type[..., 1]
    in_c1 = state.ent_asm_in_count[..., 1]

    n = buf_type.shape[0] - 1
    for d in range(1, 5):
        dy, dx = _DY[d], _DX[d]
        facing_d = (state.ent_direction == d) & is_arm

        # The source is the tile behind, opposite to the facing. Read asm_out
        # first, which holds the recipe output of an assembler or a furnace, so
        # a neighbouring arm can empty that slot. Read buf for every other
        # kind: miners, pallets, and belts. The rcv_* values on the gather
        # destination side read the same tile as src_*, so the code below
        # reuses them.
        _, _, src_eidx, src_valid, src_diff, src_safe = _lookup_neighbor(
            ey, ex, -dy, -dx, h, w, state.tile_entity, n
        )
        src_out_t = out_type[src_safe]
        src_out_c = out_count[src_safe]
        src_buf_t = buf_type[src_safe]
        src_buf_c = buf_count[src_safe]
        src_use_out = src_out_c > 0
        src_bt = jnp.where(src_use_out, src_out_t, src_buf_t)
        src_bc = jnp.where(src_use_out, src_out_c, src_buf_c)
        src_has = src_valid & src_diff & (src_bc > 0)

        # The destination is the tile in front, in the facing direction. A
        # destination with input slots takes the push into ``ent_asm_in``: slot
        # 0 when it is empty or holds the same item, and slot 1 in every other
        # case. Every other kind takes the push into ``ent_buf``. The giv_*
        # values on the gather source side read the same tile as dst_*.
        _, _, dst_eidx, dst_valid, dst_diff, dst_safe = _lookup_neighbor(
            ey, ex, dy, dx, h, w, state.tile_entity, n
        )
        dst_type = state.ent_type[dst_safe]
        dst_has_input_slots = MACHINE_HAS_INPUT_SLOTS[dst_type.astype(jnp.int32)]

        dst_bc = buf_count[dst_safe]
        dst_bt = buf_type[dst_safe]
        dst_max = MACHINE_MAX_STACK[dst_type.astype(jnp.int32)]
        dst_empty = dst_bc == 0
        dst_same = dst_bt == src_bt
        dst_space = dst_bc < dst_max
        # An empty destination does not always accept an item. A machine with a
        # MACHINE_MAX_STACK of 0 reads as empty and holds nothing.
        dst_buf_accepts = (dst_max > 0) & (dst_empty | (dst_same & dst_space))

        # Which input slot accepts: slot 0 when it is empty, or when it holds
        # the same item and has room, and slot 1 in every other case. The
        # capacity comes from the same MACHINE_MAX_STACK that the buffer route
        # uses, so an input slot fills and does not grow without bound.
        dst_in_t0 = in_t0[dst_safe]
        dst_in_c0 = in_c0[dst_safe]
        dst_in_t1 = in_t1[dst_safe]
        dst_in_c1 = in_c1[dst_safe]
        dst_s0_ok = (dst_in_c0 == 0) | ((dst_in_t0 == src_bt) & (dst_in_c0 < dst_max))
        dst_s1_ok = (dst_in_c1 == 0) | ((dst_in_t1 == src_bt) & (dst_in_c1 < dst_max))
        dst_slot_accepts = (dst_max > 0) & (dst_s0_ok | dst_s1_ok)
        dst_accepts = jnp.where(dst_has_input_slots, dst_slot_accepts, dst_buf_accepts)

        can_xfer = facing_d & src_has & dst_valid & dst_diff & dst_accepts

        # Gather on the destination side: the entity at (ey-dy, ex-dx), which
        # is the src_* tile. Gate on an active receiver. Every free slot clips
        # to tile (0, 0), so without this gate each one takes a copy of a
        # transfer aimed at the neighbour of (0, 0), and makes an item from
        # nothing.
        receiving = active & can_xfer[src_safe] & (src_eidx >= 0) & src_diff
        rcv_bt = src_bt[src_safe]

        # The ent_buf route. A machine with input slots uses the other route.
        receives_buf = receiving & ~self_has_input_slots
        buf_type = jnp.where(receives_buf, rcv_bt, buf_type)
        buf_count = jnp.where(receives_buf, buf_count + jnp.int16(1), buf_count)

        # The ent_asm_in route. Take slot 0 when it is empty, or when it holds
        # the same item and has room, and slot 1 in every other case. Per-slot
        # scalar updates keep ``jnp.stack`` out of the loop.
        receives_slots = receiving & self_has_input_slots
        self_s0_ok = (in_c0 == 0) | ((in_t0 == rcv_bt) & (in_c0 < self_max_stack))
        self_s1_ok = (in_c1 == 0) | ((in_t1 == rcv_bt) & (in_c1 < self_max_stack))
        to_s0 = receives_slots & self_s0_ok
        to_s1 = receives_slots & ~self_s0_ok & self_s1_ok
        in_t0 = jnp.where(to_s0, rcv_bt, in_t0)
        in_c0 = jnp.where(to_s0, in_c0 + jnp.int16(1), in_c0)
        in_t1 = jnp.where(to_s1, rcv_bt, in_t1)
        in_c1 = jnp.where(to_s1, in_c1 + jnp.int16(1), in_c1)

        # Gather on the source side: the entity at (ey+dy, ex+dx), which is the
        # dst_* tile. Subtract from asm_out when it holds items, and from buf
        # in every other case. The same active gate applies here. Without it, a
        # free slot pays for a transfer that it never made and goes negative.
        giving = active & can_xfer[dst_safe] & (dst_eidx >= 0) & dst_diff
        gave_out = giving & (out_count > 0)
        gave_buf = giving & ~(out_count > 0)
        out_type, out_count = _subtract_buffer(
            gave_out, out_type, out_count, jnp.int16(1)
        )
        buf_type, buf_count = _subtract_buffer(
            gave_buf, buf_type, buf_count, jnp.int16(1)
        )

    return state.replace(
        ent_buf_type=buf_type,
        ent_buf_count=buf_count,
        ent_asm_in_type=jnp.stack([in_t0, in_t1], axis=-1),
        ent_asm_in_count=jnp.stack([in_c0, in_c1], axis=-1),
        ent_asm_out_type=out_type,
        ent_asm_out_count=out_count,
    )


def run_assemblers(state: EnvState, params: EnvParams) -> EnvState:
    """Advance every assembler and furnace one step: feed, count down, finish.

    This function is where one item becomes another. A miner gives only what
    the ground already holds. A recipe here is the only way to get an item that
    no tile contains.

    A machine holds two input slots, a countdown, and one output slot. It
    starts when it holds the inputs of a recipe, counts down while it works,
    and puts the result in its output slot. An assembler and a furnace run this
    same cycle. They differ only in the recipes that they accept, and
    ``params.recipe_table.machine_type`` fixes that for each recipe.

    Two conditions keep the cycle in motion, and both belong to the caller and
    not to this function. The inputs must arrive, and the output must leave.

    The inputs arrive by one of two routes: an arm that delivers into
    ``ent_asm_in``, or the pull at the start of this pass. The pull takes one
    item from each neighbouring conveyor belt that faces the machine. Only a
    belt feeds a machine this way. The function leaves a pallet, a miner, a
    splitter, or an arm next to the machine alone. Storage next to an assembler
    therefore stays full, and a belt that runs past the side is not a feed.

    The output never leaves on its own. Nothing in this pass empties
    ``ent_asm_out``, so a machine that finished a craft stays stopped until a
    player, an arm, or a belt takes the result away. A pallet is therefore the
    only way to buffer a chain.

    The rest is timing. The machine consumes the inputs on the step that starts
    a craft, not on the step that finishes it. The countdown is the ``ticks``
    value of the recipe, in ``ent_power``. The output appears on the step that
    begins with ``ent_power == 1``, which is the step after the step that
    lowered it to 1. A recipe of ``T`` ticks therefore takes ``T + 1`` calls,
    and that count includes the call that starts it. A recipe takes one or two
    inputs, and the matcher accepts the two slots in both orders. The slot that
    an arm or a belt filled therefore does not matter.

    Parameters
    ----------
    state
        State to read.
    params
        Supplies ``recipe_table``. The recipe count sets the length of an
        unrolled match loop. A table of a different size therefore traces again
        under ``jit``, and a table of the same size with new numbers does not.

    Returns
    -------
    EnvState
        New state, with ``ent_power``, ``ent_buf_type``, ``ent_buf_count``,
        ``ent_asm_in_type``, ``ent_asm_in_count``, ``ent_asm_out_type``, and
        ``ent_asm_out_count`` updated.
    """
    table = params.recipe_table
    h, w = state.map.shape
    active = state.ent_y >= 0
    # Assemblers and furnaces share the same state layout and the same code
    # path. They differ only in the recipes that they match in Phase 3, through
    # params.recipe_table.machine_type. A science lab is absent here. It has
    # the same input slots but runs no recipe.
    is_assembler_or_furnace = (
        (state.ent_type == Machine.ASSEMBLER) | (state.ent_type == Machine.FURNACE)
    ) & active

    ey = jnp.clip(state.ent_y, 0, h - 1)
    ex = jnp.clip(state.ent_x, 0, w - 1)

    buf_type = state.ent_buf_type
    buf_count = state.ent_buf_count
    in_t0 = state.ent_asm_in_type[..., 0]
    in_c0 = state.ent_asm_in_count[..., 0]
    in_t1 = state.ent_asm_in_type[..., 1]
    in_c1 = state.ent_asm_in_count[..., 1]

    # --- Phase 0: pull from each neighbouring belt that faces the machine ---
    #
    # Only a CONVEYOR_BELT that points at the machine feeds it. Without this
    # rule the machine empties a pallet next to it, and a belt that runs past
    # the side counts as a feed.
    opposite_dir = {1: 2, 2: 1, 3: 4, 4: 3}
    n = buf_type.shape[0] - 1
    for d in range(1, 5):
        dy, dx = _DY[d], _DX[d]
        _, _, nb_eidx, nb_valid, nb_diff, nb_safe = _lookup_neighbor(
            ey, ex, dy, dx, h, w, state.tile_entity, n
        )
        nb_bt = buf_type[nb_safe]
        nb_bc = buf_count[nb_safe]
        nb_type = state.ent_type[nb_safe]
        nb_dir = state.ent_direction[nb_safe]
        nb_is_belt = nb_type == Machine.CONVEYOR_BELT
        nb_facing_self = nb_dir == jnp.int8(opposite_dir[d])
        nb_eligible = nb_valid & nb_diff & (nb_bc > 0) & nb_is_belt & nb_facing_self

        s0_ok = (in_c0 == 0) | (in_t0 == nb_bt)
        s1_ok = (in_c1 == 0) | (in_t1 == nb_bt)
        tk0 = is_assembler_or_furnace & nb_eligible & s0_ok
        tk1 = is_assembler_or_furnace & nb_eligible & ~tk0 & s1_ok
        tk = tk0 | tk1

        in_t0 = jnp.where(tk0, nb_bt, in_t0)
        in_c0 = jnp.where(tk0, in_c0 + jnp.int16(1), in_c0)
        in_t1 = jnp.where(tk1, nb_bt, in_t1)
        in_c1 = jnp.where(tk1, in_c1 + jnp.int16(1), in_c1)

        # Gather: each entity tests whether an assembler or a furnace on the
        # opposite side, in direction d, pulls from it. Gate on an active
        # payer. Every free slot clips to tile (0, 0), so without this gate
        # each one pays for a pull aimed at the neighbour of (0, 0) and goes
        # negative.
        _, _, asm_eidx, _, asm_diff, asm_safe = _lookup_neighbor(
            ey, ex, -dy, -dx, h, w, state.tile_entity, n
        )
        taken = active & tk[asm_safe] & (asm_eidx >= 0) & asm_diff
        buf_type, buf_count = _subtract_buffer(taken, buf_type, buf_count, jnp.int16(1))

    # --- Phase 1: finish the crafts that hold power == 1 ---
    completing = is_assembler_or_furnace & (state.ent_power == 1)
    out_empty = state.ent_asm_out_count == 0
    can_complete = completing & (state.ent_asm_out_type != 0) & out_empty

    # Find the per-recipe output count through the reverse index on the output
    # type. ``can_complete`` already needs ``ent_asm_out_type != 0``, so the
    # recipe index is valid wherever the where-mask fires. The clip protects
    # the lanes that the mask drops. The size comes from the table that this
    # call received, not from the base book, because a scenario supplies its
    # own. The shapes are static, so a new count only causes a recompile.
    num_recipes = table.outputs.shape[0]

    completing_ridx = table.output_to_recipe[state.ent_asm_out_type.astype(jnp.int32)]
    safe_completing_ridx = jnp.clip(completing_ridx, 0, num_recipes - 1)
    yield_count = table.output_counts[safe_completing_ridx].astype(jnp.int16)

    new_out_count = jnp.where(
        can_complete,
        yield_count,
        state.ent_asm_out_count,
    )
    new_out_type = state.ent_asm_out_type
    new_power = jnp.where(completing, jnp.int16(0), state.ent_power)

    # --- Phase 2: count down the crafts that hold power > 1 ---
    progressing = is_assembler_or_furnace & (new_power > 1)
    new_power = jnp.where(progressing, new_power - jnp.int16(1), new_power)

    # --- Phase 3: start new crafts ---
    # A recipe takes 1 or 2 inputs. A 1-input recipe carries the padding
    # (EMPTY, 0) in its second slot. The matcher tests both slot orders, so an
    # input can land in either slot. params.recipe_table.machine_type ties each
    # recipe to its machine type. ``asm_out_count == 0`` is part of the idle
    # gate. There is no Phase 4 and no buffer to empty into, so something must
    # withdraw a held output before the machine starts a new cycle.
    idle = is_assembler_or_furnace & (new_power == 0) & (new_out_count == 0)
    matched = jnp.int32(-1)
    for r in range(num_recipes):
        # A 1-input recipe holds the padding (EMPTY, 0) in its unused table
        # slot, so a match also needs the matching machine slot to be empty.
        rt_a = table.input_items[r, 0]
        ra_a = table.input_counts[r, 0]
        rt_b = table.input_items[r, 1]
        ra_b = table.input_counts[r, 1]
        rmt = table.machine_type[r]
        o1 = (in_t0 == rt_a) & (in_c0 >= ra_a) & (in_t1 == rt_b) & (in_c1 >= ra_b)
        o2 = (in_t0 == rt_b) & (in_c0 >= ra_b) & (in_t1 == rt_a) & (in_c1 >= ra_a)
        type_ok = state.ent_type == rmt
        matched = jnp.where((o1 | o2) & idle & type_ok, jnp.int32(r), matched)

    can_start = matched >= 0
    ridx = jnp.clip(matched, 0, num_recipes - 1)
    craft_t = table.ticks[ridx].astype(jnp.int16)
    out_item = table.outputs[ridx].astype(jnp.int8)

    new_power = jnp.where(can_start, craft_t, new_power)
    new_out_type = jnp.where(can_start, out_item, new_out_type)
    in_t0 = jnp.where(can_start, jnp.int8(0), in_t0)
    in_c0 = jnp.where(can_start, jnp.int16(0), in_c0)
    in_t1 = jnp.where(can_start, jnp.int8(0), in_t1)
    in_c1 = jnp.where(can_start, jnp.int16(0), in_c1)

    # There is no Phase 4. The output slot holds the recipe output until a
    # player, an arm, or a belt or pallet after it takes the item out. A pallet
    # is the only buffer in the game.

    return state.replace(
        ent_power=new_power,
        ent_buf_type=buf_type,
        ent_buf_count=buf_count,
        ent_asm_in_type=jnp.stack([in_t0, in_t1], axis=-1),
        ent_asm_in_count=jnp.stack([in_c0, in_c1], axis=-1),
        ent_asm_out_type=new_out_type,
        ent_asm_out_count=new_out_count,
    )


def run_conveyor_belts(state: EnvState, params: EnvParams) -> EnvState:
    """Advance the belt network one step: belts, splitters, and crossings.

    An arm moves an item one tile. A belt carries items across a map, and a
    chain of belts connects a mine on one side to an assembler on the other.

    Every machine in the network moves items in the direction that it faces,
    one tile in each step, and each one holds only a few items. A belt line
    therefore behaves like a line. When the far end stops taking items, the
    tile behind it fills, then the tile behind that one, and the stall travels
    back to the source.

    Three kinds share the network. They differ only in the shape of the
    junction that they make.

    ``CONVEYOR_BELT``
        The plain link. It pushes its whole buffer into the tile that it faces,
        up to the room at the destination.

    ``SPLITTER``
        A fork. It outputs to the two tiles perpendicular to its facing, one
        item to each side. It fires only while it holds 2 items or more, so an
        even split stays even under a steady flow and favours neither side.
        The two sides commit on their own, so a splitter with one blocked side
        still feeds the other side.

    ``CROSSING``
        An overpass. Two streams pass through at right angles and never mix,
        because each axis has its own buffer and its own direction. The
        crossing refuses a push aimed at the output face of an axis. Each axis
        holds 2 items and moves 1 item in each step, so an axis that holds 1
        item accepts and forwards in the same step. An axis that already holds
        2 items forwards but refuses, because a sender tests the destination
        against the count at the start of the step, before the destination
        pushed.

    One kind of neighbour refuses a delivery. A belt or a splitter does not
    push into an assembler or a furnace. Those two take items through their own
    pull or through an arm. A belt line that ends at one of them therefore
    holds its items until the machine takes them.

    A science lab is not in that set, and this difference matters before you
    connect one. A belt pushes into the ``ent_buf`` of a lab and the lab
    accepts, but ``factoriax.engine.step.run_labs`` reads ``ent_asm_in``.
    Nothing consumes what lands in the buffer.

    CAUTION: Feed a lab with an arm or with a player deposit. A belt into a lab
    fills it with packs that the lab never spends.

    Internally, a crossing stores its two streams in the slot pair that an
    assembler uses for inputs: ``ent_asm_in[:, CROSSING_VERT_SLOT]`` for the
    vertical stream and ``ent_asm_in[:, CROSSING_HORIZ_SLOT]`` for the
    horizontal stream. Its two axis directions sit in the single
    ``ent_direction`` byte, and ``CROSSING_AXIS_DIRS`` unpacks them.

    Parameters
    ----------
    state
        State to read.
    params
        The function does not read this argument. It is present so that every
        pass in this module takes the same pair, and
        :func:`update_all_machines` can call them one after the other.

    Returns
    -------
    EnvState
        New state, with ``ent_buf_type``, ``ent_buf_count``,
        ``ent_asm_in_type``, and ``ent_asm_in_count`` updated.
    """
    h, w = state.map.shape
    active = state.ent_y >= 0
    is_belt = (state.ent_type == Machine.CONVEYOR_BELT) & active
    is_splitter = (state.ent_type == Machine.SPLITTER) & active
    is_crossing = (state.ent_type == Machine.CROSSING) & active

    ey = jnp.clip(state.ent_y, 0, h - 1)
    ex = jnp.clip(state.ent_x, 0, w - 1)

    buf_type = state.ent_buf_type
    buf_count = state.ent_buf_count
    asm_in_type = state.ent_asm_in_type
    asm_in_count = state.ent_asm_in_count

    dir_index = state.ent_direction.astype(jnp.int32)

    # Splitter output sides, from its facing. An index into the (5, 2) lookup
    # gives an (N, 2) per-entity table of the two perpendicular directions that
    # this splitter pushes to.
    splitter_outputs = SPLITTER_PERP_OUTPUTS[dir_index]

    # Crossing output direction for each axis, out of the single ent_direction
    # byte. An index into the (5, 2) lookup gives (N, 2).
    crossing_axes = CROSSING_AXIS_DIRS[dir_index]
    crossing_vert_dir = crossing_axes[:, 0]  # output direction of vertical axis
    crossing_horiz_dir = crossing_axes[:, 1]  # output direction of horizontal axis

    # A splitter fires both perpendicular outputs when it holds a pair. Each
    # side commits on its own in the d-loop below (see the docstring).
    splitter_ready = is_splitter & (buf_count >= 2)

    n = buf_type.shape[0] - 1
    for d in range(1, 5):
        dy, dx = _DY[d], _DX[d]

        # One crossing axis fires in each pass: the axis with output direction
        # d. ``axis`` is the slot that it pushes from and receives into.
        if d in (3, 4):  # UP, DOWN: vertical axis
            axis = CROSSING_VERT_SLOT
            crossing_pusher_d = is_crossing & (crossing_vert_dir == d)
            crossing_dst_axis_dir = crossing_vert_dir  # used dest-side
        else:  # LEFT, RIGHT: horizontal axis
            axis = CROSSING_HORIZ_SLOT
            crossing_pusher_d = is_crossing & (crossing_horiz_dir == d)
            crossing_dst_axis_dir = crossing_horiz_dir

        belt_pusher_d = (state.ent_direction == d) & is_belt
        splitter_pusher_d = splitter_ready & (
            (splitter_outputs[:, 0] == d) | (splitter_outputs[:, 1] == d)
        )

        # Combined pusher mask. No two of the three masks overlap, because no
        # two of their machine types overlap.
        pusher_d = belt_pusher_d | splitter_pusher_d | crossing_pusher_d
        is_axis_pusher = crossing_pusher_d
        is_buf_pusher = belt_pusher_d | splitter_pusher_d

        # Per-entity source: the axis slot for a crossing, and ent_buf for a
        # belt or a splitter.
        axis_slot_count = asm_in_count[:, axis]
        axis_slot_type = asm_in_type[:, axis]
        src_count = jnp.where(is_axis_pusher, axis_slot_count, buf_count)
        src_type = jnp.where(is_axis_pusher, axis_slot_type, buf_type)

        # Destination side.
        _, _, dn_eidx, dn_valid, dn_diff, dn_safe = _lookup_neighbor(
            ey, ex, dy, dx, h, w, state.tile_entity, n
        )
        dn_type = state.ent_type[dn_safe]
        dn_is_crossing = dn_type == Machine.CROSSING
        # Refuse a push into an assembler or a furnace. Each one takes items
        # through its own pull or through an arm, so a belt that points at one
        # fills instead. A science lab is absent from this set on purpose: see
        # the module docstring and ``ISSUES.md``.
        dn_is_assembler_or_furnace = (dn_type == Machine.ASSEMBLER) | (
            dn_type == Machine.FURNACE
        )
        # A crossing accepts only on the input face of the axis, which is the
        # face opposite to its output. A belt that pushes DOWN therefore enters
        # a downward axis from the north.
        dn_axis_dir = crossing_dst_axis_dir[dn_safe]
        crossing_accepts_d = dn_is_crossing & (dn_axis_dir == d)

        # Read the buffer route of the destination: ent_buf for a destination
        # that is not a crossing, and the axis slot above for a crossing.
        dn_bt_buf = buf_type[dn_safe]
        dn_bc_buf = buf_count[dn_safe]
        dn_bt_axis = axis_slot_type[dn_safe]
        dn_bc_axis = axis_slot_count[dn_safe]
        dn_bt = jnp.where(dn_is_crossing, dn_bt_axis, dn_bt_buf)
        dn_bc = jnp.where(dn_is_crossing, dn_bc_axis, dn_bc_buf)
        dn_max = MACHINE_MAX_STACK[dn_type.astype(jnp.int32)]

        dn_empty = dn_bc == 0
        dn_same = dn_bt == src_type
        dn_space = dn_bc < dn_max

        # A crossing destination refuses a push on the wrong side. A push in
        # direction d must match the input direction of that axis.
        dn_accepts = (~dn_is_crossing) | crossing_accepts_d

        has_item = src_count > 0
        can_push = (
            pusher_d
            & has_item
            & dn_valid
            & dn_diff
            & dn_accepts
            & ~dn_is_assembler_or_furnace
            & (dn_max > 0)
            & (dn_empty | (dn_same & dn_space))
        )

        # A belt pushes every item that fits. A splitter and a crossing push
        # exactly one item for each fire of an axis.
        push_target = jnp.where(
            is_axis_pusher | splitter_pusher_d, jnp.int16(1), src_count
        )
        xfer = jnp.where(can_push, push_target, jnp.int16(0))
        xfer = jnp.minimum(xfer, dn_max - dn_bc)

        # Gather: each entity tests whether a pusher in direction d, on the
        # tile at -d, aims at it. The receivers take one of two routes. A belt,
        # a splitter, or a pallet receives through ent_buf. A crossing receives
        # through ent_asm_in[*, axis], with the same axis constant that the
        # pushers of this pass use.
        _, _, up_eidx, _, up_diff, up_safe = _lookup_neighbor(
            ey, ex, -dy, -dx, h, w, state.tile_entity, n
        )
        # Gate on an active receiver. Every free slot clips to tile (0, 0), so
        # without this gate a belt or a splitter that pushes towards the
        # neighbour of (0, 0) copies its item into every free slot through the
        # buffer route below, where ~is_crossing is true.
        incoming = active & can_push[up_safe] & (up_eidx >= 0) & up_diff
        in_type = src_type[up_safe]
        in_xfer = xfer[up_safe]

        # Gather on the buf route: every receiver that is not a crossing. An
        # assembler or a furnace cannot appear here, because can_push already
        # removed them.
        receives_buf = incoming & ~is_crossing
        buf_type = jnp.where(receives_buf, in_type, buf_type)
        buf_count = jnp.where(receives_buf, buf_count + in_xfer, buf_count)

        # Gather on the axis route: a crossing receiver, into the axis slot of
        # this pass.
        receives_axis = incoming & is_crossing
        axis_slot_type_after = jnp.where(receives_axis, in_type, axis_slot_type)
        axis_slot_count_after = jnp.where(
            receives_axis, axis_slot_count + in_xfer, axis_slot_count
        )

        # Scatter. The subtraction reads the values after the gather, so an
        # entity that received and pushed in the same pass ends at
        # old + in_xfer - xfer.
        buf_type, buf_count = _subtract_buffer(
            can_push & is_buf_pusher, buf_type, buf_count, xfer
        )
        axis_slot_type_after, axis_slot_count_after = _subtract_buffer(
            can_push & is_axis_pusher, axis_slot_type_after, axis_slot_count_after, xfer
        )

        # Persist axis-slot updates back into the (N, 2) ent_asm_in arrays.
        asm_in_type = asm_in_type.at[:, axis].set(axis_slot_type_after)
        asm_in_count = asm_in_count.at[:, axis].set(axis_slot_count_after)

    return state.replace(
        ent_buf_type=buf_type,
        ent_buf_count=buf_count,
        ent_asm_in_type=asm_in_type,
        ent_asm_in_count=asm_in_count,
    )
