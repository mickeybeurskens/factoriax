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
    BlockType,
    ItemType,
    MachineType,
)
from factoriax.recipes import (
    NUM_RECIPES,
    RECIPE_OUTPUTS,
    RECIPE_TICKS,
    RECIPES,
)
from factoriax.state import EnvParams, EnvState

_DY: tuple[int, ...] = (0, 0, 0, -1, 1)
_DX: tuple[int, ...] = (0, -1, 1, 0, 0)


def update_all_machines(
    state: EnvState, params: EnvParams,
) -> EnvState:
    """Update all machines for one step.

    Args:
        state: Current environment state.
        params: Environment parameters.

    Returns:
        Updated state.
    """
    state = refuel_machines(state, params)
    state = run_miners(state, params)
    state = run_assemblers(state)
    state = run_conveyor_belts(state)
    return state


def refuel_machines(
    state: EnvState, params: EnvParams,
) -> EnvState:
    """Consume coal from miner fuel to restore power.

    Args:
        state: Current environment state.
        params: Environment parameters.

    Returns:
        Updated state.
    """
    active = state.ent_y >= 0
    is_miner = (state.ent_type == MachineType.MINER) & active
    needs_power = state.ent_power <= 0
    has_fuel = state.ent_fuel > 0
    should_refuel = is_miner & needs_power & has_fuel

    amt = should_refuel.astype(jnp.int16)
    new_fuel = state.ent_fuel - amt
    new_power = state.ent_power + amt * jnp.int16(params.power_per_coal)

    return state.replace(ent_fuel=new_fuel, ent_power=new_power)


def run_miners(
    state: EnvState, params: EnvParams,
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
    has_power = state.ent_power > 0

    # Gather grid data at miner positions (clipped for safety).
    ey = jnp.clip(state.ent_y, 0, h - 1)
    ex = jnp.clip(state.ent_x, 0, w - 1)
    resources = state.block_resources[ey, ex]
    block_item = BLOCK_TO_ITEM_ARRAY[state.map[ey, ex].astype(jnp.int32)]

    has_resources = resources > 0
    buf_empty = state.ent_buf_count == 0
    buf_same = state.ent_buf_type == block_item.astype(jnp.int8)
    buf_ok = buf_empty | buf_same
    has_space = state.ent_buf_count < jnp.int16(64)

    can_mine = is_miner & has_power & has_resources & buf_ok & has_space
    mine_amt = jnp.where(
        can_mine, jnp.int16(params.miner_mining_rate), jnp.int16(0),
    )
    mine_amt = jnp.minimum(mine_amt, resources)
    mine_amt = jnp.minimum(mine_amt, jnp.int16(64) - state.ent_buf_count)
    mined = mine_amt > 0

    # Update entity state.
    new_buf_type = jnp.where(
        mined, block_item.astype(jnp.int8), state.ent_buf_type,
    )
    new_buf_count = state.ent_buf_count + mine_amt
    new_power = state.ent_power - mined.astype(jnp.int16)

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
        int(ItemType.COAL), int(ItemType.IRON_ORE),
        int(ItemType.COPPER_ORE), int(ItemType.TIN_ORE),
        int(ItemType.SILICON),
    ):
        mined_flat = mined_flat.at[item_id].set(
            jnp.sum(jnp.where(
                (block_item == item_id) & mined,
                mine_amt.astype(jnp.int32), 0,
            )),
        )

    return state.replace(
        map=new_map,
        block_resources=new_resources,
        ent_buf_type=new_buf_type,
        ent_buf_count=new_buf_count,
        ent_power=new_power,
        items_mined=state.items_mined + mined_flat,
    )


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
    is_asm = (state.ent_type == MachineType.ASSEMBLER) & active

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
        # Clip for safe gathering (invalid slots read entity 0, masked out).
        nb_safe = jnp.clip(nb_eidx, 0, buf_type.shape[0] - 1)

        nb_bt = buf_type[nb_safe]
        nb_bc = buf_count[nb_safe]
        nb_has = nb_valid & (nb_bc > 0)

        s0_ok = (in_c0 == 0) | (in_t0 == nb_bt)
        s1_ok = (in_c1 == 0) | (in_t1 == nb_bt)
        tk0 = is_asm & nb_has & s0_ok
        tk1 = is_asm & nb_has & ~tk0 & s1_ok
        tk = tk0 | tk1

        in_t0 = jnp.where(tk0, nb_bt, in_t0)
        in_c0 = jnp.where(tk0, in_c0 + jnp.int16(1), in_c0)
        in_t1 = jnp.where(tk1, nb_bt, in_t1)
        in_c1 = jnp.where(tk1, in_c1 + jnp.int16(1), in_c1)

        # Clear taken item from neighbor entity.
        new_nb_c = jnp.where(tk, nb_bc - jnp.int16(1), nb_bc)
        buf_count = buf_count.at[nb_safe].set(
            jnp.where(tk, new_nb_c, buf_count[nb_safe]),
        )
        buf_type = buf_type.at[nb_safe].set(
            jnp.where(tk & (new_nb_c == 0), jnp.int8(0), buf_type[nb_safe]),
        )

    # --- Phase 1: Complete crafts (power == 1) ---
    completing = is_asm & (state.ent_power == 1)
    out_empty = state.ent_asm_out_count == 0
    can_complete = completing & (state.ent_asm_out_type != 0) & out_empty

    new_out_count = jnp.where(
        can_complete, jnp.int16(1), state.ent_asm_out_count,
    )
    new_out_type = state.ent_asm_out_type
    new_power = jnp.where(completing, jnp.int16(0), state.ent_power)

    # --- Phase 2: Progress (power > 1) ---
    progressing = is_asm & (new_power > 1)
    new_power = jnp.where(progressing, new_power - jnp.int16(1), new_power)

    # --- Phase 3: Start new crafts ---
    idle = is_asm & (new_power == 0)
    matched = jnp.int32(-1)
    for r in range(NUM_RECIPES):
        inputs = RECIPES[r]["inputs"]
        rt_a, ra_a = inputs[0]
        rt_b = inputs[1][0] if len(inputs) > 1 else 0
        ra_b = inputs[1][1] if len(inputs) > 1 else 0
        if rt_b == 0:
            m = ((in_t0 == rt_a) & (in_c0 >= ra_a)) | (
                (in_t1 == rt_a) & (in_c1 >= ra_a)
            )
        else:
            o1 = (
                (in_t0 == rt_a) & (in_c0 >= ra_a)
                & (in_t1 == rt_b) & (in_c1 >= ra_b)
            )
            o2 = (
                (in_t0 == rt_b) & (in_c0 >= ra_b)
                & (in_t1 == rt_a) & (in_c1 >= ra_a)
            )
            m = o1 | o2
        matched = jnp.where(m & idle, jnp.int32(r), matched)

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

    # --- Phase 4: Push output to buffer ---
    can_push = is_asm & (new_out_count > 0) & (buf_count == 0)
    buf_type = jnp.where(can_push, new_out_type, buf_type)
    buf_count = jnp.where(can_push, new_out_count, buf_count)
    new_out_type = jnp.where(can_push, jnp.int8(0), new_out_type)
    new_out_count = jnp.where(can_push, jnp.int16(0), new_out_count)

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
    """Move items along belts using entity-based neighbor lookups.

    Each belt looks up its upstream neighbor via ``tile_entity``.

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

        # Upstream = opposite of facing direction.
        up_y = jnp.clip(ey - dy, 0, h - 1)
        up_x = jnp.clip(ex - dx, 0, w - 1)

        up_eidx = state.tile_entity[up_y, up_x]
        up_valid = up_eidx >= 0
        up_safe = jnp.clip(up_eidx, 0, buf_type.shape[0] - 1)

        up_bt = buf_type[up_safe]
        up_bc = buf_count[up_safe]

        can_pull = facing_d & up_valid & (buf_count == 0) & (up_bc > 0)

        buf_type = jnp.where(can_pull, up_bt, buf_type)
        buf_count = jnp.where(can_pull, up_bc, buf_count)

        # Clear upstream.
        buf_type = buf_type.at[up_safe].set(
            jnp.where(can_pull, jnp.int8(0), buf_type[up_safe]),
        )
        buf_count = buf_count.at[up_safe].set(
            jnp.where(can_pull, jnp.int16(0), buf_count[up_safe]),
        )

    return state.replace(ent_buf_type=buf_type, ent_buf_count=buf_count)
