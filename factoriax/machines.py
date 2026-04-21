"""Machine update logic using entity-based processing.

Machine state lives in fixed-size entity arrays instead of grid arrays.
Each function iterates over entity slots (MAX_M) rather than the full
map (H×W), making cost proportional to machine count not map size.

Neighbor lookups use ``tile_entity[y, x]`` to find the entity index at
a grid position, then gather that entity's state.
"""

import jax.numpy as jnp

from factoriax.constants import (
    BLOCK_TO_ITEM_ARRAY,
    MACHINE_MAX_STACK,
    BlockType,
    ItemType,
    MachineType,
)
from factoriax.recipes import (
    NUM_RECIPES,
    RECIPE_MACHINE_TYPE,
    RECIPE_OUTPUTS,
    RECIPE_TICKS,
    RECIPES,
)
from factoriax.state import EnvParams, EnvState

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
    state = run_assemblers(state)
    state = run_conveyor_belts(state)
    state = run_arms(state)
    return state


def run_miners(
    state: EnvState,
    params: EnvParams,
) -> EnvState:
    """Extract ore into miner buffers.

    Reads block_resources from the grid at each miner's position.
    Writes to entity buffer arrays.

    Args:
        state: Current environment state.
        params: Environment parameters.

    Returns:
        Updated state.
    """
    h, w = state.map.shape
    active = state.ent_y >= 0
    is_miner = (state.ent_type == MachineType.MINER) & active

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
        dn_max = MACHINE_MAX_STACK[state.ent_type[dn_safe].astype(jnp.int32)]
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
        incoming = can_push[up_safe] & (up_eidx >= 0) & up_diff
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


def run_arms(state: EnvState) -> EnvState:
    """Transfer one item from source (behind) to destination (in front).

    Arms perform instant pass-through: no internal buffer. Each tick
    an arm looks at the entity behind it (opposite of facing), takes
    one item from its buffer, and deposits it into the entity it
    faces (if that entity has space).

    Args:
        state: Current environment state.

    Returns:
        Updated state.
    """
    h, w = state.map.shape
    active = state.ent_y >= 0
    is_arm = (state.ent_type == MachineType.ARM) & active

    ey = jnp.clip(state.ent_y, 0, h - 1)
    ex = jnp.clip(state.ent_x, 0, w - 1)

    buf_type = state.ent_buf_type
    buf_count = state.ent_buf_count

    for d in range(1, 5):
        dy, dx = _DY[d], _DX[d]
        facing_d = (state.ent_direction == d) & is_arm

        # Source = behind (opposite of facing).
        src_y = jnp.clip(ey - dy, 0, h - 1)
        src_x = jnp.clip(ex - dx, 0, w - 1)
        src_eidx = state.tile_entity[src_y, src_x]
        src_valid = src_eidx >= 0
        src_diff = (src_y != ey) | (src_x != ex)
        src_safe = jnp.clip(src_eidx, 0, buf_type.shape[0] - 1)

        src_bt = buf_type[src_safe]
        src_bc = buf_count[src_safe]
        src_has = src_valid & src_diff & (src_bc > 0)

        # Destination = in front (facing direction).
        dst_y = jnp.clip(ey + dy, 0, h - 1)
        dst_x = jnp.clip(ex + dx, 0, w - 1)
        dst_eidx = state.tile_entity[dst_y, dst_x]
        dst_valid = dst_eidx >= 0
        dst_diff = (dst_y != ey) | (dst_x != ex)
        dst_safe = jnp.clip(dst_eidx, 0, buf_type.shape[0] - 1)

        dst_bc = buf_count[dst_safe]
        dst_bt = buf_type[dst_safe]
        dst_max = MACHINE_MAX_STACK[state.ent_type[dst_safe].astype(jnp.int32)]
        dst_empty = dst_bc == 0
        dst_same = dst_bt == src_bt
        dst_space = dst_bc < dst_max

        can_xfer = (
            facing_d
            & src_has
            & dst_valid
            & dst_diff
            & (dst_empty | (dst_same & dst_space))
        )

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

        buf_type = jnp.where(receiving, rcv_bt, buf_type)
        buf_count = jnp.where(
            receiving,
            buf_count + jnp.int16(1),
            buf_count,
        )

        # -- Source side: look at tile (ey + dy, ex + dx). If an arm
        #    there faces d and can_xfer, this entity loses 1 item.
        giv_y = jnp.clip(ey + dy, 0, h - 1)
        giv_x = jnp.clip(ex + dx, 0, w - 1)
        giv_diff = (giv_y != ey) | (giv_x != ex)
        giv_eidx = state.tile_entity[giv_y, giv_x]
        giv_safe = jnp.clip(giv_eidx, 0, buf_type.shape[0] - 1)
        giving = can_xfer[giv_safe] & (giv_eidx >= 0) & giv_diff

        new_c = buf_count - jnp.where(giving, jnp.int16(1), jnp.int16(0))
        buf_type = jnp.where(giving & (new_c == 0), jnp.int8(0), buf_type)
        buf_count = jnp.where(giving, new_c, buf_count)

    return state.replace(ent_buf_type=buf_type, ent_buf_count=buf_count)


def run_assemblers(state: EnvState) -> EnvState:
    """Run assemblers: pull inputs, craft, push output to buffer.

    Entity-based: iterates over entity slots. Neighbor lookups use
    ``tile_entity`` grid to find adjacent entities.

    Args:
        state: Current environment state.

    Returns:
        Updated state.
    """
    h, w = state.map.shape
    active = state.ent_y >= 0
    # Assemblers and furnaces share the same engine shape and code path;
    # they only differ in which recipes they're allowed to match in
    # Phase 3 (via RECIPE_MACHINE_TYPE).
    is_combiner = (
        (state.ent_type == MachineType.ASSEMBLER)
        | (state.ent_type == MachineType.FURNACE)
    ) & active

    ey = jnp.clip(state.ent_y, 0, h - 1)
    ex = jnp.clip(state.ent_x, 0, w - 1)

    buf_type = state.ent_buf_type
    buf_count = state.ent_buf_count
    in_t0 = state.ent_asm_in_type[..., 0]
    in_c0 = state.ent_asm_in_count[..., 0]
    in_t1 = state.ent_asm_in_type[..., 1]
    in_c1 = state.ent_asm_in_count[..., 1]

    # --- Phase 0: Pull inputs from adjacent tiles ---
    for d in range(1, 5):
        dy, dx = _DY[d], _DX[d]
        ny = jnp.clip(ey + dy, 0, h - 1)
        nx = jnp.clip(ex + dx, 0, w - 1)

        # Look up neighbor entity via grid.
        nb_eidx = state.tile_entity[ny, nx]
        nb_valid = nb_eidx >= 0
        nb_diff = (ny != ey) | (nx != ex)
        nb_safe = jnp.clip(nb_eidx, 0, buf_type.shape[0] - 1)

        nb_bt = buf_type[nb_safe]
        nb_bc = buf_count[nb_safe]
        nb_has = nb_valid & nb_diff & (nb_bc > 0)

        s0_ok = (in_c0 == 0) | (in_t0 == nb_bt)
        s1_ok = (in_c1 == 0) | (in_t1 == nb_bt)
        tk0 = is_combiner & nb_has & s0_ok
        tk1 = is_combiner & nb_has & ~tk0 & s1_ok
        tk = tk0 | tk1

        in_t0 = jnp.where(tk0, nb_bt, in_t0)
        in_c0 = jnp.where(tk0, in_c0 + jnp.int16(1), in_c0)
        in_t1 = jnp.where(tk1, nb_bt, in_t1)
        in_c1 = jnp.where(tk1, in_c1 + jnp.int16(1), in_c1)

        # Gather: each entity checks if an assembler on the opposite
        # side (direction d) is taking from it.
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

    new_out_count = jnp.where(
        can_complete,
        jnp.int16(1),
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
    # gated to its owning machine type via RECIPE_MACHINE_TYPE.
    # ``asm_out_count == 0`` is part of the idle gate: without a
    # downstream buffer to drain to (no Phase 4), a stuck output
    # must be withdrawn before the machine can start a new cycle.
    idle = is_combiner & (new_power == 0) & (new_out_count == 0)
    matched = jnp.int32(-1)
    for r in range(NUM_RECIPES):
        (rt_a, ra_a), (rt_b, ra_b) = RECIPES[r]["inputs"]
        rmt = RECIPE_MACHINE_TYPE[r]
        o1 = (in_t0 == rt_a) & (in_c0 >= ra_a) & (in_t1 == rt_b) & (in_c1 >= ra_b)
        o2 = (in_t0 == rt_b) & (in_c0 >= ra_b) & (in_t1 == rt_a) & (in_c1 >= ra_a)
        type_ok = state.ent_type == rmt
        matched = jnp.where((o1 | o2) & idle & type_ok, jnp.int32(r), matched)

    can_start = matched >= 0
    ridx = jnp.clip(matched, 0, NUM_RECIPES - 1)
    craft_t = RECIPE_TICKS[ridx].astype(jnp.int16)
    out_item = RECIPE_OUTPUTS[ridx].astype(jnp.int8)

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


def run_conveyor_belts(state: EnvState) -> EnvState:
    """Push items along belts into the entity they face.

    Each belt pushes its buffer into the downstream entity (belt,
    pallet, or any machine with buffer space). This replaces the
    old pull-from-upstream model so belts naturally feed into pallets.

    Args:
        state: Current environment state.

    Returns:
        Updated state.
    """
    h, w = state.map.shape
    active = state.ent_y >= 0
    is_belt = (state.ent_type == MachineType.CONVEYOR_BELT) & active

    ey = jnp.clip(state.ent_y, 0, h - 1)
    ex = jnp.clip(state.ent_x, 0, w - 1)

    buf_type = state.ent_buf_type
    buf_count = state.ent_buf_count

    for d in range(1, 5):
        dy, dx = _DY[d], _DX[d]
        facing_d = (state.ent_direction == d) & is_belt

        dn_y = jnp.clip(ey + dy, 0, h - 1)
        dn_x = jnp.clip(ex + dx, 0, w - 1)

        dn_eidx = state.tile_entity[dn_y, dn_x]
        dn_valid = dn_eidx >= 0
        dn_diff = (dn_y != ey) | (dn_x != ex)  # not pushing to self
        dn_safe = jnp.clip(dn_eidx, 0, buf_type.shape[0] - 1)

        dn_bt = buf_type[dn_safe]
        dn_bc = buf_count[dn_safe]
        dn_empty = dn_bc == 0
        dn_same = dn_bt == buf_type
        dn_max = MACHINE_MAX_STACK[state.ent_type[dn_safe].astype(jnp.int32)]
        dn_space = dn_bc < dn_max

        has_item = buf_count > 0
        can_push = (
            facing_d & has_item & dn_valid & dn_diff & (dn_empty | (dn_same & dn_space))
        )

        xfer = jnp.where(can_push, buf_count, jnp.int16(0))
        xfer = jnp.minimum(xfer, dn_max - dn_bc)

        # Gather: for each entity, check if a pusher targets it.
        # Look at the tile opposite to direction d; if an entity there
        # is pushing (facing d), this entity is the destination.
        up_y = jnp.clip(ey - dy, 0, h - 1)
        up_x = jnp.clip(ex - dx, 0, w - 1)
        up_diff = (up_y != ey) | (up_x != ex)
        up_eidx = state.tile_entity[up_y, up_x]
        up_safe = jnp.clip(up_eidx, 0, buf_type.shape[0] - 1)
        incoming = can_push[up_safe] & (up_eidx >= 0) & up_diff
        in_type = buf_type[up_safe]
        in_xfer = xfer[up_safe]

        buf_type = jnp.where(incoming, in_type, buf_type)
        buf_count = jnp.where(incoming, buf_count + in_xfer, buf_count)

        # Subtract sent items from source.
        new_c = buf_count - xfer
        buf_type = jnp.where(
            can_push & (new_c == 0),
            jnp.int8(0),
            buf_type,
        )
        buf_count = jnp.where(can_push, new_c, buf_count)

    return state.replace(ent_buf_type=buf_type, ent_buf_count=buf_count)
