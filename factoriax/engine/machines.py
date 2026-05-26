"""Machine update logic using entity-based processing.

Machine state lives in fixed-size entity arrays instead of grid arrays.
Each function iterates over entity slots (MAX_M) rather than the full
map (H×W), making cost proportional to machine count not map size.

Neighbor lookups use ``tile_entity[y, x]`` to find the entity index at
a grid position, then gather that entity's state.
"""

import jax.numpy as jnp

from factoriax.engine.belts import (
    CROSSING_AXIS_DIRS,
    CROSSING_HORIZ_SLOT,
    CROSSING_VERT_SLOT,
)
from factoriax.engine.constants import (
    BlockType,
    ItemType,
    Machine,
)
from factoriax.engine.recipes import NUM_RECIPES
from factoriax.engine.state import EnvParams, EnvState
from factoriax.engine.tables import BLOCK_TO_ITEM_ARRAY

_DY: tuple[int, ...] = (0, 0, 0, -1, 1)
_DX: tuple[int, ...] = (0, -1, 1, 0, 0)

# Miner output slot holds at most one mining cycle's worth. Combined
# with ``miner_mining_rate`` this means a miner tops up in one tick
# and idles until something (arm, belt, player) drains it — matching
# the "no buffering in output slots" rule. Pallets remain the only
# large-capacity storage.
MINER_OUTPUT_CAP: int = 3


def update_all_machines(
    state: EnvState,
    params: EnvParams,
) -> EnvState:
    """Update all machines for one step.

    Args:
        state: Current environment state.
        params: Environment parameters.

    Returns:
        Updated state.
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
    """Extract ore from the tile a miner is standing on.

    Miners mine the tile **directly underneath them** — they read
    ``state.block_resources`` at their own ``(ent_y, ent_x)``, never
    at an adjacent tile. To start a node, place the miner ON an
    ore-patch tile (typically the south-edge tile, with its facing
    direction pointing at an adjacent pallet to receive the push).

    Per tick, the miner extracts ``params.miner_mining_rate`` ore,
    capped by the tile's remaining ``block_resources`` and the
    miner's ``MINER_OUTPUT_CAP`` buffer slot. The miner stops once
    its tile is depleted, even if other tiles of the same patch
    still hold ore — covering a full patch needs one miner per
    tile, or moving the miner.

    Args:
        state: Current environment state.
        params: Environment parameters.

    Returns:
        Updated state.
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
    # Buf-space clamp: only meaningful for miners (non-miners have
    # mine_amt=0 already, and their buf may legitimately exceed
    # MINER_OUTPUT_CAP — e.g. pallets go up to 1000).
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
    new_map = state.map.at[ey, ex].set(
        jnp.where(is_depleted, jnp.int8(int(BlockType.DIRT)), state.map[ey, ex]),
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
    for d in range(1, 5):
        dy, dx = _DY[d], _DX[d]
        facing_d = (state.ent_direction == d) & is_miner

        dn_y = jnp.clip(ey + dy, 0, h - 1)
        dn_x = jnp.clip(ex + dx, 0, w - 1)
        dn_eidx = state.tile_entity[dn_y, dn_x]
        dn_valid = dn_eidx >= 0
        dn_diff = (dn_y != ey) | (dn_x != ex)
        dn_safe = jnp.clip(dn_eidx, 0, new_buf_type.shape[0] - 1)

        dn_bc = new_buf_count[dn_safe]
        dn_bt = new_buf_type[dn_safe]
        dn_max = params.machine_config.max_stack[
            state.ent_type[dn_safe].astype(jnp.int32)
        ]
        dn_empty = dn_bc == 0
        dn_same = dn_bt == new_buf_type
        has_buf = new_buf_count > 0

        can_push = (
            facing_d
            & has_buf
            & dn_valid
            & dn_diff
            & (dn_empty | (dn_same & (dn_bc < dn_max)))
        )
        xfer = jnp.where(can_push, new_buf_count, jnp.int16(0))
        xfer = jnp.minimum(xfer, dn_max - dn_bc)

        # Gather: each entity checks if a miner behind it (opposite
        # of d) is pushing to it.
        up_y = jnp.clip(ey - dy, 0, h - 1)
        up_x = jnp.clip(ex - dx, 0, w - 1)
        up_diff = (up_y != ey) | (up_x != ex)
        up_eidx = state.tile_entity[up_y, up_x]
        up_safe = jnp.clip(up_eidx, 0, new_buf_type.shape[0] - 1)
        # Gate on the receiver being active: inactive slots all clip to
        # tile (0, 0), so without this they would each pull in a push
        # aimed at the real entity on (0, 0)'s neighbour -- duplicating
        # the item and leaking it into slots later reused by placement.
        incoming = active & can_push[up_safe] & (up_eidx >= 0) & up_diff
        in_type = new_buf_type[up_safe]
        in_xfer = xfer[up_safe]

        new_buf_type = jnp.where(incoming, in_type, new_buf_type)
        new_buf_count = jnp.where(
            incoming,
            new_buf_count + in_xfer,
            new_buf_count,
        )

        # Subtract sent items from source.
        remaining = new_buf_count - xfer
        new_buf_type = jnp.where(
            can_push & (remaining == 0),
            jnp.int8(0),
            new_buf_type,
        )
        new_buf_count = jnp.where(can_push, remaining, new_buf_count)

    return state.replace(
        map=new_map,
        block_resources=new_resources,
        ent_buf_type=new_buf_type,
        ent_buf_count=new_buf_count,
        items_mined=state.items_mined + mined_flat,
    )


def run_arms(state: EnvState, params: EnvParams) -> EnvState:
    """Transfer one item from source (behind) to destination (in front).

    Arms perform instant pass-through: no internal buffer. Each tick
    an arm looks at the entity behind it (opposite of facing), takes
    one item from it, and deposits it into the entity it faces (if
    that entity has space).

    The source slot picks ``ent_asm_out`` over ``ent_buf``: when an
    assembler/furnace finishes a recipe its output sits in
    ``ent_asm_out``, and an adjacent arm pulls from there to free
    the slot for the next cycle. Buffer machines (miner, pallet,
    belt) keep their items in ``ent_buf``, so the same arm can
    drain those too. Pre-fix, arms only saw ``ent_buf`` — recipe
    outputs got stuck in ``asm_out`` with no automated way out, and
    the docstring intent at the bottom of ``run_assemblers`` ("a
    withdraw (player, arm, or downstream belt/pallet) pulls it
    out") didn't match the code.

    Args:
        state: Current environment state.
        params: Environment parameters (supplies the per-machine
            buffer cap via ``params.machine_config.max_stack``).

    Returns:
        Updated state.
    """
    h, w = state.map.shape
    active = state.ent_y >= 0
    is_arm = (state.ent_type == Machine.ARM) & active
    self_is_combiner = (state.ent_type == Machine.ASSEMBLER) | (
        state.ent_type == Machine.FURNACE
    )

    ey = jnp.clip(state.ent_y, 0, h - 1)
    ex = jnp.clip(state.ent_x, 0, w - 1)

    buf_type = state.ent_buf_type
    buf_count = state.ent_buf_count
    out_type = state.ent_asm_out_type
    out_count = state.ent_asm_out_count
    # Per-slot scalars for asm_in updates — stack only at function
    # return so per-iteration jnp.stack allocations don't dominate.
    in_t0 = state.ent_asm_in_type[..., 0]
    in_c0 = state.ent_asm_in_count[..., 0]
    in_t1 = state.ent_asm_in_type[..., 1]
    in_c1 = state.ent_asm_in_count[..., 1]

    for d in range(1, 5):
        dy, dx = _DY[d], _DX[d]
        facing_d = (state.ent_direction == d) & is_arm

        # Source = behind (opposite of facing). Read asm_out first
        # (assembler/furnace recipe output) so adjacent arms can
        # drain the output slot; fall back to buf for everything
        # else (miners, pallets, belts).
        src_y = jnp.clip(ey - dy, 0, h - 1)
        src_x = jnp.clip(ex - dx, 0, w - 1)
        src_eidx = state.tile_entity[src_y, src_x]
        src_valid = src_eidx >= 0
        src_diff = (src_y != ey) | (src_x != ex)
        src_safe = jnp.clip(src_eidx, 0, buf_type.shape[0] - 1)

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
        dst_y = jnp.clip(ey + dy, 0, h - 1)
        dst_x = jnp.clip(ex + dx, 0, w - 1)
        dst_eidx = state.tile_entity[dst_y, dst_x]
        dst_valid = dst_eidx >= 0
        dst_diff = (dst_y != ey) | (dst_x != ex)
        dst_safe = jnp.clip(dst_eidx, 0, buf_type.shape[0] - 1)

        dst_type = state.ent_type[dst_safe]
        dst_is_combiner = (dst_type == Machine.ASSEMBLER) | (
            dst_type == Machine.FURNACE
        )

        dst_bc = buf_count[dst_safe]
        dst_bt = buf_type[dst_safe]
        dst_max = params.machine_config.max_stack[dst_type.astype(jnp.int32)]
        dst_empty = dst_bc == 0
        dst_same = dst_bt == src_bt
        dst_space = dst_bc < dst_max
        dst_buf_accepts = dst_empty | (dst_same & dst_space)

        # Combiner-destination receptivity: slot 0 first if empty
        # or matches type, else slot 1.
        dst_in_t0 = in_t0[dst_safe]
        dst_in_c0 = in_c0[dst_safe]
        dst_in_t1 = in_t1[dst_safe]
        dst_in_c1 = in_c1[dst_safe]
        dst_combiner_s0_ok = (dst_in_c0 == 0) | (dst_in_t0 == src_bt)
        dst_combiner_s1_ok = (dst_in_c1 == 0) | (dst_in_t1 == src_bt)
        dst_combiner_accepts = dst_combiner_s0_ok | dst_combiner_s1_ok
        dst_accepts = jnp.where(dst_is_combiner, dst_combiner_accepts, dst_buf_accepts)

        can_xfer = facing_d & src_has & dst_valid & dst_diff & dst_accepts

        # Gather: each entity checks if an arm behind it (opposite
        # of d) is transferring to it, and if an arm in front of it
        # is taking from it.
        # -- Destination side: look at tile (ey - dy, ex - dx). If an
        #    arm there faces d and can_xfer, this entity receives.
        rcv_y = jnp.clip(ey - dy, 0, h - 1)
        rcv_x = jnp.clip(ex - dx, 0, w - 1)
        rcv_diff = (rcv_y != ey) | (rcv_x != ex)
        rcv_eidx = state.tile_entity[rcv_y, rcv_x]
        rcv_safe = jnp.clip(rcv_eidx, 0, buf_type.shape[0] - 1)
        receiving = can_xfer[rcv_safe] & (rcv_eidx >= 0) & rcv_diff
        rcv_bt = src_bt[rcv_safe]

        # ent_buf-track: combiners receive into ``ent_asm_in`` instead.
        receives_buf = receiving & ~self_is_combiner
        buf_type = jnp.where(receives_buf, rcv_bt, buf_type)
        buf_count = jnp.where(
            receives_buf,
            buf_count + jnp.int16(1),
            buf_count,
        )

        # ent_asm_in track: pick slot 0 first if empty or matches the
        # incoming type, else slot 1. Per-slot scalar updates avoid
        # the per-iteration ``jnp.stack`` allocator pressure.
        receives_combiner = receiving & self_is_combiner
        self_s0_ok = (in_c0 == 0) | (in_t0 == rcv_bt)
        self_s1_ok = (in_c1 == 0) | (in_t1 == rcv_bt)
        to_s0 = receives_combiner & self_s0_ok
        to_s1 = receives_combiner & ~self_s0_ok & self_s1_ok
        in_t0 = jnp.where(to_s0, rcv_bt, in_t0)
        in_c0 = jnp.where(to_s0, in_c0 + jnp.int16(1), in_c0)
        in_t1 = jnp.where(to_s1, rcv_bt, in_t1)
        in_c1 = jnp.where(to_s1, in_c1 + jnp.int16(1), in_c1)

        # -- Source side: look at tile (ey + dy, ex + dx). If an arm
        #    there faces d and can_xfer, this entity loses 1 item.
        #    Decrement asm_out first if it has stuff (matches the
        #    arm's read decision via ``src_use_out``); otherwise
        #    decrement buf.
        giv_y = jnp.clip(ey + dy, 0, h - 1)
        giv_x = jnp.clip(ex + dx, 0, w - 1)
        giv_diff = (giv_y != ey) | (giv_x != ex)
        giv_eidx = state.tile_entity[giv_y, giv_x]
        giv_safe = jnp.clip(giv_eidx, 0, buf_type.shape[0] - 1)
        giving = can_xfer[giv_safe] & (giv_eidx >= 0) & giv_diff

        gave_out = giving & (out_count > 0)
        gave_buf = giving & ~(out_count > 0)

        new_out_c = out_count - jnp.where(gave_out, jnp.int16(1), jnp.int16(0))
        out_type = jnp.where(
            gave_out & (new_out_c == 0),
            jnp.int8(0),
            out_type,
        )
        out_count = jnp.where(gave_out, new_out_c, out_count)

        new_buf_c = buf_count - jnp.where(gave_buf, jnp.int16(1), jnp.int16(0))
        buf_type = jnp.where(
            gave_buf & (new_buf_c == 0),
            jnp.int8(0),
            buf_type,
        )
        buf_count = jnp.where(gave_buf, new_buf_c, buf_count)

    return state.replace(
        ent_buf_type=buf_type,
        ent_buf_count=buf_count,
        ent_asm_in_type=jnp.stack([in_t0, in_t1], axis=-1),
        ent_asm_in_count=jnp.stack([in_c0, in_c1], axis=-1),
        ent_asm_out_type=out_type,
        ent_asm_out_count=out_count,
    )


def run_assemblers(state: EnvState, params: EnvParams) -> EnvState:
    """Run assemblers: pull inputs, craft, push output to buffer.

    Entity-based: iterates over entity slots. Neighbor lookups use
    ``tile_entity`` grid to find adjacent entities. Recipe identity
    and balance numbers are read via ``params.recipe_table`` so a
    tuned :class:`~factoriax.engine.state.EnvParams` re-uses the cached XLA
    trace (shape stable) but applies the user's balance overlay.

    Args:
        state: Current environment state.
        params: Environment parameters (supplies the recipe table).

    Returns:
        Updated state.
    """
    table = params.recipe_table
    h, w = state.map.shape
    active = state.ent_y >= 0
    # Assemblers and furnaces share the same engine shape and code path;
    # they only differ in which recipes they're allowed to match in
    # Phase 3 (via params.recipe_table.machine_type).
    is_combiner = (
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
    # Combiners pull only from a neighbour that is a CONVEYOR_BELT
    # whose direction points *at* the machine (``nb_dir ==
    # opposite(d)``). Pallets, miners, splitters, and arms are
    # excluded — pallets in particular don't push, so they shouldn't
    # passively feed adjacent machines (this kills the F+A drain on
    # incidentally-adjacent ent_buf storage). Belts whose direction
    # is parallel/perpendicular to the scan are also excluded — only
    # a belt aimed at the machine counts as a feed.
    opposite_dir = {1: 2, 2: 1, 3: 4, 4: 3}
    for d in range(1, 5):
        dy, dx = _DY[d], _DX[d]
        ny = jnp.clip(ey + dy, 0, h - 1)
        nx = jnp.clip(ex + dx, 0, w - 1)

        nb_eidx = state.tile_entity[ny, nx]
        nb_valid = nb_eidx >= 0
        nb_diff = (ny != ey) | (nx != ex)
        nb_safe = jnp.clip(nb_eidx, 0, buf_type.shape[0] - 1)

        nb_bt = buf_type[nb_safe]
        nb_bc = buf_count[nb_safe]
        nb_type = state.ent_type[nb_safe]
        nb_dir = state.ent_direction[nb_safe]
        nb_is_belt = nb_type == Machine.CONVEYOR_BELT
        nb_facing_self = nb_dir == jnp.int8(opposite_dir[d])
        nb_eligible = nb_valid & nb_diff & (nb_bc > 0) & nb_is_belt & nb_facing_self

        s0_ok = (in_c0 == 0) | (in_t0 == nb_bt)
        s1_ok = (in_c1 == 0) | (in_t1 == nb_bt)
        tk0 = is_combiner & nb_eligible & s0_ok
        tk1 = is_combiner & nb_eligible & ~tk0 & s1_ok
        tk = tk0 | tk1

        in_t0 = jnp.where(tk0, nb_bt, in_t0)
        in_c0 = jnp.where(tk0, in_c0 + jnp.int16(1), in_c0)
        in_t1 = jnp.where(tk1, nb_bt, in_t1)
        in_c1 = jnp.where(tk1, in_c1 + jnp.int16(1), in_c1)

        # Gather: each entity checks if a combiner on the opposite
        # side (direction d) is pulling from it.
        asm_y = jnp.clip(ey - dy, 0, h - 1)
        asm_x = jnp.clip(ex - dx, 0, w - 1)
        asm_diff = (asm_y != ey) | (asm_x != ex)
        asm_eidx = state.tile_entity[asm_y, asm_x]
        asm_safe = jnp.clip(asm_eidx, 0, buf_type.shape[0] - 1)
        taken = tk[asm_safe] & (asm_eidx >= 0) & asm_diff

        new_c = buf_count - jnp.where(taken, jnp.int16(1), jnp.int16(0))
        buf_type = jnp.where(
            taken & (new_c == 0),
            jnp.int8(0),
            buf_type,
        )
        buf_count = jnp.where(taken, new_c, buf_count)

    # --- Phase 1: Complete crafts (power == 1) ---
    completing = is_combiner & (state.ent_power == 1)
    out_empty = state.ent_asm_out_count == 0
    can_complete = completing & (state.ent_asm_out_type != 0) & out_empty

    # Look up the per-recipe output count via the output-type reverse
    # index. ``can_complete`` already requires ``ent_asm_out_type != 0``
    # so when the where-mask fires the recipe index is guaranteed valid;
    # the clip is defensive for the lanes that mask out.
    completing_ridx = table.output_to_recipe[state.ent_asm_out_type.astype(jnp.int32)]
    safe_completing_ridx = jnp.clip(completing_ridx, 0, NUM_RECIPES - 1)
    yield_count = table.output_counts[safe_completing_ridx].astype(jnp.int16)

    new_out_count = jnp.where(
        can_complete,
        yield_count,
        state.ent_asm_out_count,
    )
    new_out_type = state.ent_asm_out_type
    new_power = jnp.where(completing, jnp.int16(0), state.ent_power)

    # --- Phase 2: Progress (power > 1) ---
    progressing = is_combiner & (new_power > 1)
    new_power = jnp.where(progressing, new_power - jnp.int16(1), new_power)

    # --- Phase 3: Start new crafts ---
    # Recipes are 1- or 2-input, with 1-input recipes padded to
    # (EMPTY, 0) in the second slot. The matcher checks both slot
    # orderings so inputs can land in either slot. Each recipe is
    # gated to its owning machine type via params.recipe_table.machine_type.
    # ``asm_out_count == 0`` is part of the idle gate: without a
    # downstream buffer to drain to (no Phase 4), a stuck output
    # must be withdrawn before the machine can start a new cycle.
    idle = is_combiner & (new_power == 0) & (new_out_count == 0)
    matched = jnp.int32(-1)
    for r in range(NUM_RECIPES):
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
    ridx = jnp.clip(matched, 0, NUM_RECIPES - 1)
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
    """Advance the belt network one tick — belts, splitters, crossings.

    Three tile types share this pass:

    * **CONVEYOR_BELT** — pushes its buffer in the single direction it
      faces (1 item every tick, capped by destination space).
    * **SPLITTER** — buffers up to 2 items in ``ent_buf``. When the
      buffer holds a pair (>= 2), the splitter dispatches one push per
      perpendicular output side; each push commits independently using
      the d-loop's standard receptivity check, so a half-blocked
      splitter still drains 1 to its receptive side instead of
      stalling on both. With buffer of 1 the splitter holds — the
      pair-firing semantic keeps the even-split contract honest under
      symmetric flow, and the in-loop ``has_item`` guard prevents
      overdraw if the second push would race the first.
    * **CROSSING** — two independent per-axis flows. Slot
      ``ent_asm_in[idx, 0]`` is the vertical buffer (UP/DOWN flow),
      slot ``ent_asm_in[idx, 1]`` is the horizontal buffer (LEFT/RIGHT
      flow). Streams cannot mix because they live in disjoint slots
      and each axis only fires in its own direction. A push from a
      belt or splitter onto a crossing's *output* side is rejected
      (the crossing has well-defined input and output sides per axis).
      Per-axis cap is 2 — one cell of slack so a saturated chain
      fed at one tile per tick can also drain at one tile per tick
      without a separate look-ahead pass: the gather puts incoming +
      old (= 2) into the slot, the scatter subtracts the outgoing
      (= 1), leaving 1 in steady state.

    Folding all three tile types into a single 4-iteration
    scatter-gather loop shares the receptivity work (one pass instead
    of three). The branching per entity is constant-time and remains
    XLA-friendly because every branch is a ``jnp.where`` over masks.

    Args:
        state: Current environment state.
        params: Environment parameters (supplies the per-machine
            buffer cap via ``params.machine_config.max_stack``).

    Returns:
        Updated state.
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

    # Splitter group masks: vertical-facing (UP/DOWN) splitters output
    # LEFT/RIGHT; horizontal-facing (LEFT/RIGHT) output UP/DOWN.
    # Direction values: LEFT=1, RIGHT=2, UP=3, DOWN=4 (Direction enum).
    is_vert_split = is_splitter & (
        (state.ent_direction == 3) | (state.ent_direction == 4)
    )
    is_horiz_split = is_splitter & (
        (state.ent_direction == 1) | (state.ent_direction == 2)
    )

    # Crossing per-axis output directions — decoded from the packed
    # ent_direction (1..4 maps the four flow combinations). Indexing
    # the (5, 2) lookup yields a (N, 2) per-entity table.
    crossing_axes = CROSSING_AXIS_DIRS[state.ent_direction.astype(jnp.int32)]
    crossing_vert_dir = crossing_axes[:, 0]  # output direction of vertical axis
    crossing_horiz_dir = crossing_axes[:, 1]  # output direction of horizontal axis

    has_pair = buf_count >= 2
    # Pair-gated firing. A splitter attempts to fire *both* perpendicular
    # outputs whenever its buffer holds a pair (>= 2 items). Each output
    # is dispatched as an independent push in the d-loop below, where the
    # standard per-iteration receptivity check (``dn_empty | (dn_same &
    # dn_space)``) decides whether that side actually commits — so a
    # blocked downstream on one side simply skips that iteration while
    # the other side still pushes one item, instead of stalling the
    # whole splitter. Buffer of 1 holds: the pair-firing semantic
    # preserves the even-split contract under symmetric flow, and the
    # second-iteration ``has_item`` check ensures we never overdraw the
    # buffer when the second side is also receptive.
    splitter_pushers_vert = is_vert_split & has_pair
    splitter_pushers_horiz = is_horiz_split & has_pair

    # Index 0 (direction NONE) is never read — the d-loop below ranges
    # 1..4 — but a zeros placeholder keeps the tuple typed as Array
    # rather than ``Array | None`` so downstream ``|`` ops type-check.
    _no_pusher = jnp.zeros_like(splitter_pushers_vert)
    splitter_pushers_by_dir: tuple[jnp.ndarray, ...] = (
        _no_pusher,  # 0 — unused
        splitter_pushers_vert,  # LEFT  — vert-facing splitters output here
        splitter_pushers_vert,  # RIGHT
        splitter_pushers_horiz,  # UP   — horiz-facing splitters output here
        splitter_pushers_horiz,  # DOWN
    )

    for d in range(1, 5):
        dy, dx = _DY[d], _DX[d]

        # Crossings push: vert axis fires when d == vert_axis_dir; horiz
        # axis fires when d == horiz_axis_dir. The two axes are
        # independent — a single crossing can fire in two iterations
        # (one per axis), but never twice in the same iteration.
        # ``axis`` is the slot that this iteration's crossing pushes
        # *and receives* read/write from.
        if d in (3, 4):  # UP, DOWN — vertical axis
            axis = CROSSING_VERT_SLOT
            crossing_pusher_d = is_crossing & (crossing_vert_dir == d)
            crossing_dst_axis_dir = crossing_vert_dir  # used dest-side
        else:  # LEFT, RIGHT — horizontal axis
            axis = CROSSING_HORIZ_SLOT
            crossing_pusher_d = is_crossing & (crossing_horiz_dir == d)
            crossing_dst_axis_dir = crossing_horiz_dir

        belt_pusher_d = (state.ent_direction == d) & is_belt
        splitter_pusher_d = splitter_pushers_by_dir[d]

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
        dn_y = jnp.clip(ey + dy, 0, h - 1)
        dn_x = jnp.clip(ex + dx, 0, w - 1)
        dn_eidx = state.tile_entity[dn_y, dn_x]
        dn_valid = dn_eidx >= 0
        dn_diff = (dn_y != ey) | (dn_x != ex)
        dn_safe = jnp.clip(dn_eidx, 0, buf_type.shape[0] - 1)

        dn_type = state.ent_type[dn_safe]
        dn_is_crossing = dn_type == Machine.CROSSING
        # Reject pushes into combiner destinations. Combiners receive
        # inputs only via Phase 0's directional pull (from facing
        # belts) or via an arm pushing into ``ent_asm_in`` — never via
        # a belt's blind push into ``ent_buf`` (where items would
        # accumulate uncontrolled). Belts whose terminus faces a
        # combiner back-pressure: items hold on the belt and Phase 0
        # picks them up next tick.
        dn_is_combiner = (dn_type == Machine.ASSEMBLER) | (dn_type == Machine.FURNACE)
        # Crossing destination only accepts pushes that align with the
        # *input direction* for the relevant axis. The input direction is
        # the same as the axis output direction (a flow N→S takes inputs
        # from the N side and outputs to the S side; a belt pushing DOWN
        # is correctly entering the N face).
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
        dn_max = params.machine_config.max_stack[dn_type.astype(jnp.int32)]

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
            & ~dn_is_combiner
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
        up_y = jnp.clip(ey - dy, 0, h - 1)
        up_x = jnp.clip(ex - dx, 0, w - 1)
        up_diff = (up_y != ey) | (up_x != ex)
        up_eidx = state.tile_entity[up_y, up_x]
        up_safe = jnp.clip(up_eidx, 0, buf_type.shape[0] - 1)
        # Gate on the receiver being active: inactive slots all clip to
        # tile (0, 0), so without this a belt or splitter pushing toward
        # (0, 0)'s neighbour would duplicate its item into every inactive
        # slot via the buffer track below (~is_crossing is true for them).
        incoming = active & can_push[up_safe] & (up_eidx >= 0) & up_diff
        in_type = src_type[up_safe]
        in_xfer = xfer[up_safe]

        # Buf-track gather: receivers that are not crossings.
        # Combiner receivers are unreachable here because can_push
        # already excluded ``dn_is_combiner`` — combiners are fed only
        # by Phase 0's directional pull and by arms pushing into
        # ``ent_asm_in`` (see ``run_arms``).
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

        # Scatter (subtract from source). The post-gather buf/axis values
        # are what we subtract from — same chain semantics as the
        # original belt loop (an entity that both received and pushed
        # in the same iteration ends with old + in_xfer - xfer).
        new_buf_c = buf_count - xfer
        buf_type = jnp.where(
            can_push & is_buf_pusher & (new_buf_c == 0), jnp.int8(0), buf_type
        )
        buf_count = jnp.where(can_push & is_buf_pusher, new_buf_c, buf_count)

        new_axis_c = axis_slot_count_after - xfer
        axis_slot_type_after = jnp.where(
            can_push & is_axis_pusher & (new_axis_c == 0),
            jnp.int8(0),
            axis_slot_type_after,
        )
        axis_slot_count_after = jnp.where(
            can_push & is_axis_pusher, new_axis_c, axis_slot_count_after
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
