"""Move the factory forward by one step.

A factory is machines standing on tiles, handing items to each other. This
module is that handoff. Ore leaves the ground, items become other items, and
both travel between machines that do not touch. Everything on the map that
acts on its own acts here.

Four passes do the work. :func:`update_all_machines` runs them in order and
is what a step calls:

:func:`run_miners`
    Ore out of the ground, into the machine the miner faces.
:func:`run_assemblers`
    Assemblers and furnaces turn input items into an output item, over time.
:func:`run_conveyor_belts`
    Belts, splitters, and crossings carry items across the map.
:func:`run_arms`
    Arms lift one item between two machines a tile apart.

A pass takes a state and returns a new one. Machines run whether or not a
player is nearby, and player actions are handled elsewhere.

Machine state lives in the ``ent_`` arrays of
:class:`~factoriax.engine.state.EnvState`, indexed by entity id rather than by
tile. There are ``E`` slots, fixed when the level is built, and a machine
holds one for as long as it stands on the map. An entity finds its neighbours
through ``state.tile_entity``, which maps a tile back to the entity id
standing on it, or to ``-1`` for an empty tile.

Every pass runs over all ``E`` slots at once, free ones included. Under
``jit`` there is no per-entity branch to skip a slot with, so a free slot
computes a neighbour, a transfer, and a receipt like everybody else. What
makes it harmless is a mask, not a skip: a result is kept only where
``active = state.ent_y >= 0``. A free slot holds ``ent_y < 0``, which the
position clip folds onto tile (0, 0), so it reads as a neighbour of whatever
stands beside that corner. Miss the mask on one line and nothing raises. An
item appears in a slot that holds no machine.

Two things follow. Cost tracks ``E`` and the map shape rather than how much
has been built, so an empty factory costs what a full one costs. And items
move by scatter-gather: the sending side works out what it would push, the
receiving side reaches back and recomputes the same decision from its own
point of view, and neither writes into the other's slot.
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

# Row and column step per ``Direction`` value, so ``_DY[d], _DX[d]`` moves one
# tile in direction ``d``. Index 0 is the unset direction and stays put.
_DY: tuple[int, ...] = (0, 0, 0, -1, 1)
_DX: tuple[int, ...] = (0, -1, 1, 0, 0)

#: Ore a miner will mine into its own buffer. At the default
#: ``miner_mining_rate`` a miner reaches this in one step and then idles until
#: an arm, a belt, or a player drains it. This is not the buffer's capacity.
#: ``MACHINE_MAX_STACK`` gives a miner far more room than this, and a push
#: from a belt or an arm fills it past the cap. Pallets stay the only large
#: storage.
MINER_OUTPUT_CAP: int = 3


def _subtract_buffer(
    cond: jnp.ndarray,
    buf_type: jnp.ndarray,
    buf_count: jnp.ndarray,
    amt: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Take ``amt`` items out of a buffer slot wherever ``cond`` holds.

    Every transfer in this module has a side that pays, and this is it. What
    makes it worth sharing is the item type. A slot that reaches zero has to
    forget what it held, because the rest of the engine reads a cleared type
    as "free to accept anything". Drop a count without that step and the slot
    still looks committed to an item it no longer has.

    Elementwise over entity slots, and nothing is clamped. Subtracting more
    than a slot holds leaves a negative count and a stale type, so every
    caller passes an ``amt`` already capped by the source count.

    Parameters
    ----------
    cond
        Which slots to subtract from. Shape ``(E,)``, bool.
    buf_type
        Item id held per slot. Shape ``(E,)``, int8.
    buf_count
        Items held per slot. Shape ``(E,)``, int16.
    amt
        Amount to remove per slot. Shape ``(E,)``, int16. Read only where
        ``cond`` holds.

    Returns
    -------
    tuple of jnp.ndarray
        The updated ``(buf_type, buf_count)`` pair. Slots outside ``cond``
        come back untouched.
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

    Every pass here needs to know who is standing next door. This answers that
    for all entities at once, and hands back the flags a caller needs to tell
    a real neighbour from an artefact of having asked.

    The artefacts come from asking on behalf of every slot. Some of those
    entities sit on an edge of the map, and some hold no machine at all.
    Neither can be turned away with a branch, so both get an answer and a flag
    instead.

    A tile past the edge is clipped back on, which makes an edge entity look
    up itself. ``diff`` is False in exactly that case, and callers put it in
    their transfer mask so an edge machine does not push into its own slot.
    ``safe`` is the entity index clipped into range for the same reason: a
    caller gathers with it unconditionally and drops what it read afterwards.

    Parameters
    ----------
    ey
        Row of every entity, already clipped to the map. Shape ``(E,)``.
    ex
        Column of every entity, already clipped to the map. Shape ``(E,)``.
    dy
        Row offset to step, a Python int so the caller's direction loop
        unrolls at trace time.
    dx
        Column offset to step.
    h
        Map height in tiles.
    w
        Map width in tiles.
    tile_entity
        ``state.tile_entity``: entity id per tile, ``-1`` where the tile
        holds no machine. Shape ``(H, W)``.
    n
        Highest entity index that exists, used as the clip bound for the
        gather index. Callers pass ``E - 1``.

    Returns
    -------
    tuple of jnp.ndarray
        ``(ny, nx, eidx, valid, diff, safe)``, each shaped like ``ey``.
        ``ny`` and ``nx`` are the clipped neighbour tile. ``eidx`` is the
        entity standing there, ``-1`` for an empty tile. ``valid`` is
        ``eidx >= 0``. ``diff`` is False when the clip folded the neighbour
        back onto ``(ey, ex)``. ``safe`` is ``eidx`` clipped into ``[0, n]``
        so a gather stays in bounds; what it reads means nothing unless
        ``valid`` and ``diff`` both hold.
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

    This is the entry point a step calls. It runs the four passes in one
    fixed order: miners, then assemblers and furnaces, then the belt network,
    then arms.

    The order sets how far an item can travel in a single step. Ore a miner
    pushes into a pallet can be lifted back out by an arm in that same step,
    because arms run last. An item an arm drops onto a belt waits for the next
    step, because the belt pass has already gone by. Reordering the calls
    changes throughput, so the order is part of the contract rather than an
    implementation detail.

    Science labs are not driven here. ``factoriax.engine.step.run_labs``
    drains their input slots as a separate part of the full step.

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

    Mining is where items enter the world. Everything downstream of a miner
    is a rearrangement of what miners produce.

    A miner works the tile it stands on, not one beside it. Place it on the
    ore patch, facing whatever should receive the output. Each step it takes
    ``params.miner_mining_rate`` ore and pushes its buffer into the machine
    in front, so a working miner is two decisions: where it stands and which
    way it looks.

    Two limits apply to the amount mined: what the tile still holds, and the
    room left under :data:`MINER_OUTPUT_CAP`. That cap is small, so a miner
    nothing drains fills up and idles rather than stockpiling. The machine in
    front limits the push instead, not the mining, so a miner facing a full
    pallet still takes ore out of the ground and then stalls holding it.

    A tile that runs out turns to ``BlockType.DIRT`` and that miner stops,
    even when other tiles of the same patch still hold ore. Working a whole
    patch takes one miner per tile.

    The push needs a real machine in front, of a kind that can hold items and
    holding either nothing or the same item already. A miner facing open
    ground, or facing something with no room, keeps its ore and stalls. It
    pushes in no direction but the one it faces.

    Parameters
    ----------
    state
        State to read.
    params
        Supplies ``miner_mining_rate``.

    Returns
    -------
    EnvState
        New state with ``map``, ``block_resources``, ``ent_buf_type``,
        ``ent_buf_count``, and ``items_mined`` updated. ``items_mined``
        counts coal, iron ore, copper ore, tin ore, and silicon only; ore of
        any other item id is still mined and buffered but never tallied.
    """
    h, w = state.map.shape
    active = state.ent_y >= 0
    is_miner = (state.ent_type == Machine.MINER) & active

    # Gather grid data at miner positions (clipped for safety).
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
    # Clamp to free output space, miners only. Other kinds already have
    # mine_amt=0 and are allowed to hold more than MINER_OUTPUT_CAP.
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

    # Scatter updates back to grid (block_resources, map).
    new_resources = state.block_resources.at[ey, ex].add(
        jnp.where(mined, -mine_amt, jnp.int16(0)),
    )
    is_depleted = (new_resources[ey, ex] <= 0) & mined
    # Add a delta rather than setting the value: free slots all clip to tile
    # (0, 0) and would put a second, stale write on that index, and duplicate
    # scatter indices resolve in an unspecified order. Every lane that is not
    # depleting contributes zero.
    to_dirt = jnp.int8(int(BlockType.DIRT)) - state.map[ey, ex]
    new_map = state.map.at[ey, ex].add(
        jnp.where(is_depleted, to_dirt, jnp.int8(0)),
    )

    # Track mined items.
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

    # --- Push buffer to adjacent entity in facing direction ---
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

        # Gather: each entity checks if a miner behind it (opposite
        # of d) is pushing to it.
        _, _, up_eidx, _, up_diff, up_safe = _lookup_neighbor(
            ey, ex, -dy, -dx, h, w, state.tile_entity, n
        )
        # Gate on the receiver being active: inactive slots all clip to
        # tile (0, 0), so without this they would each pull in a push
        # aimed at the real entity on (0, 0)'s neighbour -- duplicating
        # the item and leaking it into slots later reused by placement.
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

    Machines do not reach into each other. An arm is what connects two of
    them, and it is the only way to unload a machine that does not push on
    its own, such as an assembler or a pallet.

    An arm bridges the two tiles it sits between. Each step it takes one item
    off the machine behind it, meaning the tile opposite its facing, and gives
    that item to the machine in front. One item per arm per step, both ends
    required: no source, no room in front, and nothing moves.

    Which slot an item comes from depends on the machine behind. A finished
    craft parks in ``ent_asm_out`` and blocks the next one, so arms drain
    that slot first and keep an assembler or furnace cycling. Machines that
    only store, such as miners, pallets, and belts, are drained from
    ``ent_buf`` instead.

    Which slot it lands in depends on the machine in front. An assembler,
    furnace, or science lab takes delivery into ``ent_asm_in``, slot 0 when
    that is empty or already holds the same item, otherwise slot 1. An input
    slot fills to the machine's ``MACHINE_MAX_STACK``, the same limit the
    buffer route uses, so an arm feeding a machine that is not consuming
    stalls rather than piling up without bound. Everything else receives into
    ``ent_buf``, also up to ``MACHINE_MAX_STACK``.

    An arm and a player deposit are the two ways to give a science lab a pack
    it will consume. ``factoriax.engine.step.run_labs`` reads ``ent_asm_in``
    and nothing else, and both routes land there. A belt aimed at a lab is not
    refused, but it delivers into the lab's ``ent_buf``, where the items sit
    unconsumed until an arm or a player takes them back out.

    An arm never draws from its own ``ent_buf``, but that buffer is not
    sealed. ``MACHINE_MAX_STACK`` gives an arm room for one item and a belt
    facing an arm pushes into it. The next arm along treats a loaded arm as an
    ordinary source and drains it like any other machine.

    Parameters
    ----------
    state
        State to read.
    params
        Unused. Present so every pass in this module takes the same pair and
        :func:`update_all_machines` can call them in a row.

    Returns
    -------
    EnvState
        New state with ``ent_buf_type``, ``ent_buf_count``,
        ``ent_asm_in_type``, ``ent_asm_in_count``, ``ent_asm_out_type``, and
        ``ent_asm_out_count`` updated.
    """
    h, w = state.map.shape
    active = state.ent_y >= 0
    is_arm = (state.ent_type == Machine.ARM) & active
    # Assemblers, furnaces, and science labs take a delivery into
    # ``ent_asm_in``. One table decides that everywhere; see
    # ``MACHINE_HAS_INPUT_SLOTS``.
    self_has_input_slots = MACHINE_HAS_INPUT_SLOTS[state.ent_type.astype(jnp.int32)]
    self_max_stack = MACHINE_MAX_STACK[state.ent_type.astype(jnp.int32)]

    ey = jnp.clip(state.ent_y, 0, h - 1)
    ex = jnp.clip(state.ent_x, 0, w - 1)

    buf_type = state.ent_buf_type
    buf_count = state.ent_buf_count
    out_type = state.ent_asm_out_type
    out_count = state.ent_asm_out_count
    # Per-slot scalars for asm_in updates. Stacked once at return so the
    # direction loop does not allocate a jnp.stack per iteration.
    in_t0 = state.ent_asm_in_type[..., 0]
    in_c0 = state.ent_asm_in_count[..., 0]
    in_t1 = state.ent_asm_in_type[..., 1]
    in_c1 = state.ent_asm_in_count[..., 1]

    n = buf_type.shape[0] - 1
    for d in range(1, 5):
        dy, dx = _DY[d], _DX[d]
        facing_d = (state.ent_direction == d) & is_arm

        # Source = behind (opposite of facing). Read asm_out first
        # (assembler/furnace recipe output) so adjacent arms can
        # drain the output slot; fall back to buf for everything
        # else (miners, pallets, belts).
        # rcv_* (gather dest side) looks at the same tile as src_*, so
        # we reuse those values below instead of computing them twice.
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

        # Destination = in front (facing direction). Combiner
        # destinations route the push to ``ent_asm_in`` (slot 0
        # first if empty/type-match, else slot 1); everything else
        # writes to ``ent_buf``.
        # giv_* (gather source side) looks at the same tile as dst_*.
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
        # An empty destination is not automatically a willing one: a machine
        # whose MACHINE_MAX_STACK is 0 reads as empty and holds nothing.
        dst_buf_accepts = (dst_max > 0) & (dst_empty | (dst_same & dst_space))

        # Input-slot receptivity: slot 0 first if empty or holding the same
        # item with room left, else slot 1. Capacity comes from the same
        # MACHINE_MAX_STACK the buffer route uses, so an input slot fills up
        # rather than growing without bound.
        dst_in_t0 = in_t0[dst_safe]
        dst_in_c0 = in_c0[dst_safe]
        dst_in_t1 = in_t1[dst_safe]
        dst_in_c1 = in_c1[dst_safe]
        dst_s0_ok = (dst_in_c0 == 0) | ((dst_in_t0 == src_bt) & (dst_in_c0 < dst_max))
        dst_s1_ok = (dst_in_c1 == 0) | ((dst_in_t1 == src_bt) & (dst_in_c1 < dst_max))
        dst_slot_accepts = (dst_max > 0) & (dst_s0_ok | dst_s1_ok)
        dst_accepts = jnp.where(dst_has_input_slots, dst_slot_accepts, dst_buf_accepts)

        can_xfer = facing_d & src_has & dst_valid & dst_diff & dst_accepts

        # Gather destination side: entity at (ey-dy, ex-dx) = src_* tile.
        # Gate on the receiver being active: free slots all clip to tile
        # (0, 0), so without this each one would take a copy of a transfer
        # aimed at (0, 0)'s neighbour and mint an item from nothing.
        receiving = active & can_xfer[src_safe] & (src_eidx >= 0) & src_diff
        rcv_bt = src_bt[src_safe]

        # ent_buf track: machines with input slots receive there instead.
        receives_buf = receiving & ~self_has_input_slots
        buf_type = jnp.where(receives_buf, rcv_bt, buf_type)
        buf_count = jnp.where(receives_buf, buf_count + jnp.int16(1), buf_count)

        # ent_asm_in track: pick slot 0 first if empty or holding the same
        # item with room left, else slot 1. Per-slot scalar updates avoid
        # the per-iteration ``jnp.stack`` allocator pressure.
        receives_slots = receiving & self_has_input_slots
        self_s0_ok = (in_c0 == 0) | ((in_t0 == rcv_bt) & (in_c0 < self_max_stack))
        self_s1_ok = (in_c1 == 0) | ((in_t1 == rcv_bt) & (in_c1 < self_max_stack))
        to_s0 = receives_slots & self_s0_ok
        to_s1 = receives_slots & ~self_s0_ok & self_s1_ok
        in_t0 = jnp.where(to_s0, rcv_bt, in_t0)
        in_c0 = jnp.where(to_s0, in_c0 + jnp.int16(1), in_c0)
        in_t1 = jnp.where(to_s1, rcv_bt, in_t1)
        in_c1 = jnp.where(to_s1, in_c1 + jnp.int16(1), in_c1)

        # Gather source side: entity at (ey+dy, ex+dx) = dst_* tile.
        # Decrement asm_out first if it has stuff; otherwise decrement buf.
        # Same active gate, mirrored: an ungated free slot pays for a
        # transfer it never made and goes negative.
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

    This is where one item becomes another. A miner only ever yields what was
    already in the ground; a recipe run here is the only way to get an item
    that no tile contains.

    A machine holds two input slots, a countdown, and one output slot. Give it
    the inputs a recipe wants and it starts, counts down while it works, and
    parks the result in its output slot. Assemblers and furnaces run this same
    cycle and differ only in which recipes they accept, which
    ``params.recipe_table.machine_type`` fixes per recipe.

    Two things decide whether the cycle keeps turning, and both are the
    caller's problem rather than this function's. Inputs have to arrive, and
    the output has to leave.

    Inputs arrive one of two ways: an arm delivering into ``ent_asm_in``, or
    the pull at the start of this pass. The pull takes one item from each
    adjacent conveyor belt whose facing points at the machine. Only belts feed
    this way. A pallet, miner, splitter, or arm standing beside a machine is
    left alone, so storage put down next to an assembler is not quietly
    drained, and a belt running past sideways is not read as a feed.

    The output never leaves on its own. Nothing in this pass empties
    ``ent_asm_out``, so a machine that has finished a craft stays stopped
    until a player, an arm, or a belt takes the result away. That is what
    makes pallets the only way to buffer a chain.

    The rest is timing. Inputs are consumed on the step a craft starts, not
    on the step it finishes. The countdown is the recipe's ``ticks``, held in
    ``ent_power``, and the output appears on the step that begins with
    ``ent_power == 1``, which is the step after the one that decremented it
    there. A recipe of ``T`` ticks therefore takes ``T + 1`` calls, counting
    the call that starts it. A recipe takes one or two inputs, and the two
    slots are matched in either order, so it does not matter which slot an arm
    or belt happened to fill.

    Parameters
    ----------
    state
        State to read.
    params
        Supplies ``recipe_table``. The recipe count sets the length of an
        unrolled match loop, so a table of a different size retraces under
        ``jit`` while a retuned table of the same size does not.

    Returns
    -------
    EnvState
        New state with ``ent_power``, ``ent_buf_type``, ``ent_buf_count``,
        ``ent_asm_in_type``, ``ent_asm_in_count``, ``ent_asm_out_type``, and
        ``ent_asm_out_count`` updated.
    """
    table = params.recipe_table
    h, w = state.map.shape
    active = state.ent_y >= 0
    # Assemblers and furnaces share the same engine shape and code path;
    # they only differ in which recipes they're allowed to match in
    # Phase 3 (via params.recipe_table.machine_type). A science lab is not
    # here: it has the same input slots but runs no recipe.
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

    # --- Phase 0: Directional pull from facing-belt neighbours ---
    #
    # Only a CONVEYOR_BELT aimed at the machine feeds it. A pallet next door
    # would otherwise be drained without pushing, and a belt running past
    # sideways would count as a feed.
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

        # Gather: each entity checks if an assembler or furnace on the
        # opposite side (direction d) is pulling from it. Gate on the payer
        # being active: free slots all clip to tile (0, 0), so without this
        # each pays for a pull aimed at (0, 0)'s neighbour and goes negative.
        _, _, asm_eidx, _, asm_diff, asm_safe = _lookup_neighbor(
            ey, ex, -dy, -dx, h, w, state.tile_entity, n
        )
        taken = active & tk[asm_safe] & (asm_eidx >= 0) & asm_diff
        buf_type, buf_count = _subtract_buffer(taken, buf_type, buf_count, jnp.int16(1))

    # --- Phase 1: Complete crafts (power == 1) ---
    completing = is_assembler_or_furnace & (state.ent_power == 1)
    out_empty = state.ent_asm_out_count == 0
    can_complete = completing & (state.ent_asm_out_type != 0) & out_empty

    # Look up the per-recipe output count via the output-type reverse
    # index. ``can_complete`` already requires ``ent_asm_out_type != 0``
    # so when the where-mask fires the recipe index is guaranteed valid;
    # the clip is defensive for the lanes that mask out.
    # Sized from the table this call was handed, not the base book: scenarios
    # ship their own. Shapes are static, so a new count just recompiles.
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

    # --- Phase 2: Progress (power > 1) ---
    progressing = is_assembler_or_furnace & (new_power > 1)
    new_power = jnp.where(progressing, new_power - jnp.int16(1), new_power)

    # --- Phase 3: Start new crafts ---
    # Recipes are 1- or 2-input, with 1-input recipes padded to
    # (EMPTY, 0) in the second slot. The matcher checks both slot
    # orderings so inputs can land in either slot. Each recipe is
    # gated to its owning machine type via params.recipe_table.machine_type.
    # ``asm_out_count == 0`` is part of the idle gate: without a
    # downstream buffer to drain to (no Phase 4), a stuck output
    # must be withdrawn before the machine can start a new cycle.
    idle = is_assembler_or_furnace & (new_power == 0) & (new_out_count == 0)
    matched = jnp.int32(-1)
    for r in range(num_recipes):
        # 1-input recipes pad the unused slot with (EMPTY, 0) in the
        # table, so the match naturally requires the corresponding
        # slot on the machine to also be empty.
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

    # NOTE: no Phase 4. The output slot holds the recipe output until
    # a withdraw (player, arm, or downstream belt/pallet) pulls it
    # out. Pallets are the only buffering primitive.

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

    Arms move items one tile. Belts are how items cross a map. A chain of them
    is what connects a mine on one side to an assembler on the other.

    Every machine in the network moves items in the direction it faces, one
    tile per step, and each has room for only a few items. That is what makes
    a belt line behave like a line: when the far end stops taking items, the
    tile behind it fills, then the one behind that, and the stall travels
    backwards to the source.

    Three kinds share the network, and the difference between them is only
    what shape of junction they make.

    ``CONVEYOR_BELT``
        The plain link. Pushes its whole buffer into the tile it faces, as far
        as the destination has room.

    ``SPLITTER``
        A fork. Outputs to the two tiles perpendicular to its facing, one item
        per side. It fires only while holding at least 2, so an even split
        stays even under steady flow instead of favouring whichever side came
        first. The two sides commit separately, so a splitter with one blocked
        side still feeds the other rather than stalling on both.

    ``CROSSING``
        An overpass. Two streams pass through at right angles and never mix,
        because each axis has its own buffer and its own direction. A push
        aimed at the output face of an axis is refused. Each axis holds 2
        items and moves 1 per step, so an axis holding 1 accepts and forwards
        in the same step. An axis already holding 2 forwards but refuses,
        because a sender tests the destination against the count it had at the
        start of the step, before the destination pushed.

    One kind of neighbour refuses delivery. A belt or splitter will not push
    into an assembler or furnace, which take items only through their own pull
    or through an arm. A belt line ending at one of those therefore holds its
    items until the machine reaches out and takes them.

    A science lab is not on that list, and the asymmetry is worth knowing
    before wiring one up. A belt pushes into a lab's ``ent_buf`` and the lab
    accepts, but ``factoriax.engine.step.run_labs`` reads ``ent_asm_in``, so
    nothing consumes what lands there. Feed a lab with an arm or a player
    deposit. A belt run into one fills it with packs it will never spend.

    Under the hood, a crossing stores its two streams in the slot pair a
    an assembler uses for inputs: ``ent_asm_in[:, CROSSING_VERT_SLOT]`` for
    the vertical stream and ``ent_asm_in[:, CROSSING_HORIZ_SLOT]`` for the
    horizontal one. Its two axis directions are packed into the single
    ``ent_direction`` byte and unpacked by ``CROSSING_AXIS_DIRS``.

    Parameters
    ----------
    state
        State to read.
    params
        Unused. Present so every pass in this module takes the same pair and
        :func:`update_all_machines` can call them in a row.

    Returns
    -------
    EnvState
        New state with ``ent_buf_type``, ``ent_buf_count``,
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

    # Splitter output sides, decoded from its facing. Indexing the (5, 2)
    # lookup yields a (N, 2) per-entity table of the two perpendicular
    # directions this splitter pushes to.
    splitter_outputs = SPLITTER_PERP_OUTPUTS[dir_index]

    # Crossing per-axis output directions, unpacked from the single
    # ent_direction byte. Indexing the (5, 2) lookup gives (N, 2).
    crossing_axes = CROSSING_AXIS_DIRS[dir_index]
    crossing_vert_dir = crossing_axes[:, 0]  # output direction of vertical axis
    crossing_horiz_dir = crossing_axes[:, 1]  # output direction of horizontal axis

    # A splitter fires both perpendicular outputs when it holds a pair;
    # each side commits independently in the d-loop below (see docstring).
    splitter_ready = is_splitter & (buf_count >= 2)

    n = buf_type.shape[0] - 1
    for d in range(1, 5):
        dy, dx = _DY[d], _DX[d]

        # One crossing axis fires per iteration, the one whose output
        # direction is d. ``axis`` is the slot it both pushes from and
        # receives into.
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

        # Combined pusher mask. The three sub-masks are pairwise disjoint
        # because their underlying machine types are disjoint.
        pusher_d = belt_pusher_d | splitter_pusher_d | crossing_pusher_d
        is_axis_pusher = crossing_pusher_d
        is_buf_pusher = belt_pusher_d | splitter_pusher_d

        # Per-entity source: for crossings, the axis slot; for belts and
        # splitters, ent_buf.
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
        # Refuse pushes into an assembler or a furnace. Each takes items
        # through its own pull or through an arm, so a belt aimed at one
        # backs up instead. A science lab is deliberately not in this set;
        # see the module docstring and ``ISSUES.md``.
        dn_is_assembler_or_furnace = (dn_type == Machine.ASSEMBLER) | (
            dn_type == Machine.FURNACE
        )
        # A crossing accepts only on the input face of the axis, which is the
        # face opposite its output: a belt pushing DOWN enters a downward
        # axis from the north.
        dn_axis_dir = crossing_dst_axis_dir[dn_safe]
        crossing_accepts_d = dn_is_crossing & (dn_axis_dir == d)

        # Read the destination's buffer track. For non-crossing dests we
        # use ent_buf; for crossings we use the axis slot picked above.
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

        # Crossing destinations enforce wrong-side rejection: a push in
        # direction d must align with the destination's input direction
        # for the relevant axis.
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

        # Belts push everything that fits; splitters and crossings push
        # exactly one per axis-fire.
        push_target = jnp.where(
            is_axis_pusher | splitter_pusher_d, jnp.int16(1), src_count
        )
        xfer = jnp.where(can_push, push_target, jnp.int16(0))
        xfer = jnp.minimum(xfer, dn_max - dn_bc)

        # Gather: each entity checks if a pusher in direction d (located
        # at -d) is targeting it. Receivers split into two tracks: the
        # ent_buf track for belt/splitter/pallet receivers, and the
        # ent_asm_in[*, axis] track for crossing receivers (the axis is
        # the same constant the iteration's pushers use).
        _, _, up_eidx, _, up_diff, up_safe = _lookup_neighbor(
            ey, ex, -dy, -dx, h, w, state.tile_entity, n
        )
        # Gate on the receiver being active: inactive slots all clip to
        # tile (0, 0), so without this a belt or splitter pushing toward
        # (0, 0)'s neighbour would duplicate its item into every inactive
        # slot via the buffer track below (~is_crossing is true for them).
        incoming = active & can_push[up_safe] & (up_eidx >= 0) & up_diff
        in_type = src_type[up_safe]
        in_xfer = xfer[up_safe]

        # Buf-track gather: receivers that are not crossings. An assembler
        # or furnace cannot appear here; can_push already excluded them.
        receives_buf = incoming & ~is_crossing
        buf_type = jnp.where(receives_buf, in_type, buf_type)
        buf_count = jnp.where(receives_buf, buf_count + in_xfer, buf_count)

        # Axis-track gather: crossing receivers, into the iteration's
        # axis slot.
        receives_axis = incoming & is_crossing
        axis_slot_type_after = jnp.where(receives_axis, in_type, axis_slot_type)
        axis_slot_count_after = jnp.where(
            receives_axis, axis_slot_count + in_xfer, axis_slot_count
        )

        # Scatter, subtracting from the post-gather values so an entity that
        # received and pushed in the same iteration ends at
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
