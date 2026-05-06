"""Game logic for player movement and environment stepping.

Uses compound actions: placement, deposit, withdraw, and craft actions
name the specific item type. No slot cursors or recipe selection needed.
"""

import jax
import jax.numpy as jnp

from factoriax.constants import (
    BLOCK_TO_ITEM_ARRAY,
    CRAFT_BASE,
    DEPOSIT_BASE,
    DIRECTIONS,
    MINEABLE_BLOCKS,
    NUM_ITEM_TYPES,
    NUM_SCIENCE_PACK_TYPES,
    PLACE_ACTION_TO_ITEM,
    PLACE_BASE,
    PLAYER_MAX_STACK,
    ROTATE_ACTION_TO_DIR,
    ROTATE_BASE,
    SCIENCE_PACK_INDEX,
    Action,
    BlockType,
    Direction,
    ItemType,
    MachineType,
)
from factoriax.crafting import craft_recipe
from factoriax.machines import update_all_machines
from factoriax.placement import (
    get_tile_in_front,
    pickup_machine,
    place_machine,
    set_machine_direction,
)
from factoriax.recipes import NUM_RECIPES
from factoriax.state import EnvParams, EnvState


def is_position_in_bounds(
    position: jax.Array,
    map_width: int,
    map_height: int,
) -> jax.Array:
    """Check if a position is within map bounds.

    Args:
        position: (x, y) array.
        map_width: Map width.
        map_height: Map height.

    Returns:
        Boolean scalar.
    """
    return (
        (position[0] >= 0)
        & (position[0] < map_width)
        & (position[1] >= 0)
        & (position[1] < map_height)
    )


def get_block_at(state: EnvState, position: jax.Array) -> jax.Array:
    """Get the block type at a position, or OUT_OF_BOUNDS.

    Args:
        state: Current environment state.
        position: (x, y) coordinates.

    Returns:
        Block type at position.
    """
    map_height, map_width = state.map.shape
    in_bounds = is_position_in_bounds(position, map_width, map_height)
    clipped_x = jnp.clip(position[0], 0, map_width - 1)
    clipped_y = jnp.clip(position[1], 0, map_height - 1)
    return jnp.where(
        in_bounds,
        state.map[clipped_y, clipped_x],
        jnp.int8(BlockType.OUT_OF_BOUNDS),
    )


def is_position_walkable(
    state: EnvState,
    position: jax.Array,
) -> jax.Array:
    """Check if a position can be walked on.

    Args:
        state: Current environment state.
        position: (x, y) coordinates.

    Returns:
        Boolean: walkable.
    """
    block = get_block_at(state, position)
    is_solid = jnp.any(
        block
        == jnp.array(
            [
                BlockType.WATER,
                BlockType.OUT_OF_BOUNDS,
            ]
        )
    )
    map_height, map_width = state.map.shape
    in_bounds = is_position_in_bounds(position, map_width, map_height)
    clipped_x = jnp.clip(position[0], 0, map_width - 1)
    clipped_y = jnp.clip(position[1], 0, map_height - 1)
    mt = state.machine_types[clipped_y, clipped_x]
    has_blocking = in_bounds & (
        (mt != MachineType.NONE) & (mt != MachineType.CONVEYOR_BELT)
    )
    return ~is_solid & ~has_blocking


def move_player(
    state: EnvState,
    action: int | jax.Array,
    player_idx: int | jax.Array,
) -> EnvState:
    """Move or face a player based on the action.

    Args:
        state: Current environment state.
        action: Movement or facing action.
        player_idx: Player index.

    Returns:
        Updated state.
    """
    pos = state.player_positions[player_idx]
    current_dir = state.player_directions[player_idx]

    # Movement directions.
    move_dir = jnp.int32(0)
    move_dir = jnp.where(action == Action.UP, Direction.UP, move_dir)
    move_dir = jnp.where(action == Action.DOWN, Direction.DOWN, move_dir)
    move_dir = jnp.where(action == Action.LEFT, Direction.LEFT, move_dir)
    move_dir = jnp.where(action == Action.RIGHT, Direction.RIGHT, move_dir)

    is_move = move_dir > 0
    offset = DIRECTIONS[move_dir]
    new_pos = pos + offset.astype(jnp.int16)
    can_move = is_move & is_position_walkable(state, new_pos)
    final_pos = jnp.where(can_move, new_pos, pos)

    # Facing: movement sets facing, FACE_* sets facing without moving.
    new_facing = jnp.where(is_move, move_dir, current_dir)
    new_facing = jnp.where(
        action == Action.FACE_UP,
        Direction.UP,
        new_facing,
    )
    new_facing = jnp.where(
        action == Action.FACE_DOWN,
        Direction.DOWN,
        new_facing,
    )
    new_facing = jnp.where(
        action == Action.FACE_LEFT,
        Direction.LEFT,
        new_facing,
    )
    new_facing = jnp.where(
        action == Action.FACE_RIGHT,
        Direction.RIGHT,
        new_facing,
    )

    return state.replace(
        player_positions=state.player_positions.at[player_idx].set(
            final_pos,
        ),
        player_directions=state.player_directions.at[player_idx].set(
            new_facing.astype(jnp.int8),
        ),
    )


def mine_block(
    state: EnvState,
    player_idx: int | jax.Array,
) -> EnvState:
    """Mine the block at the player's position.

    Args:
        state: Current environment state.
        player_idx: Player index.

    Returns:
        Updated state.
    """
    pos = state.player_positions[player_idx]
    px, py = pos[0], pos[1]
    block_type = state.map[py, px]

    is_mineable = jnp.any(block_type == MINEABLE_BLOCKS)
    has_resources = state.block_resources[py, px] > 0
    item_type = BLOCK_TO_ITEM_ARRAY[block_type.astype(jnp.int32)]

    current_count = state.player_inventory[player_idx, item_type]
    max_stack = PLAYER_MAX_STACK[item_type]
    has_space = current_count < max_stack
    can_mine = is_mineable & has_resources & has_space

    new_resources = state.block_resources[py, px] - 1
    is_depleted = new_resources <= 0
    mine_one = jnp.where(can_mine, jnp.int16(1), jnp.int16(0))

    new_inv = state.player_inventory.at[player_idx, item_type].add(mine_one)
    new_block_resources = state.block_resources.at[py, px].set(
        jnp.where(can_mine, new_resources, state.block_resources[py, px]),
    )
    new_map = state.map.at[py, px].set(
        jnp.where(
            can_mine & is_depleted,
            jnp.int8(BlockType.DIRT),
            state.map[py, px],
        ),
    )
    new_items_mined = state.items_mined.at[item_type].add(
        jnp.where(can_mine, 1, 0),
    )

    return state.replace(
        map=new_map,
        player_inventory=new_inv,
        block_resources=new_block_resources,
        items_mined=new_items_mined,
    )


def deposit_to_adjacent(
    state: EnvState,
    player_idx: int | jax.Array,
    item_type: int | jax.Array,
) -> EnvState:
    """Deposit an item into the machine in front of the player.

    For assemblers: deposits into asm_in slots.
    For buffer machines: deposits into buffer.

    Args:
        state: Current environment state.
        player_idx: Player index.
        item_type: ItemType to deposit.

    Returns:
        Updated state.
    """
    item_type = jnp.int32(item_type)
    tx, ty = get_tile_in_front(state, player_idx)
    map_h, map_w = state.map.shape
    in_bounds = (tx >= 0) & (tx < map_w) & (ty >= 0) & (ty < map_h)
    sx = jnp.clip(tx, 0, map_w - 1)
    sy = jnp.clip(ty, 0, map_h - 1)

    mt = jnp.where(in_bounds, state.machine_types[sy, sx], MachineType.NONE)
    has_machine = mt != MachineType.NONE
    player_count = state.player_inventory[player_idx, item_type]
    has_item = player_count > 0

    # Entity lookup for the target tile.
    max_e = state.ent_y.shape[0]
    eidx_raw = state.tile_entity[sy, sx]
    eidx = jnp.clip(eidx_raw, 0, max_e - 1)

    # Assemblers and furnaces share the 2-input-slot shape.
    is_combiner = (mt == MachineType.ASSEMBLER) | (mt == MachineType.FURNACE)

    # Deposit to combiner input slot.
    in_t0 = state.ent_asm_in_type[eidx, 0]
    in_c0 = state.ent_asm_in_count[eidx, 0]
    in_t1 = state.ent_asm_in_type[eidx, 1]
    in_c1 = state.ent_asm_in_count[eidx, 1]

    slot0_ok = (in_c0 == 0) | (in_t0 == item_type)
    slot1_ok = (in_c1 == 0) | (in_t1 == item_type)
    use_s0 = is_combiner & slot0_ok
    use_s1 = is_combiner & ~use_s0 & slot1_ok

    can_deposit_asm = in_bounds & has_item & (use_s0 | use_s1)

    # Deposit to buffer machine (non-combiner, non-miner).
    is_miner = mt == MachineType.MINER
    buf_empty = state.ent_buf_count[eidx] == 0
    buf_same = state.ent_buf_type[eidx] == item_type
    buf_space = state.ent_buf_count[eidx] < jnp.int16(64)
    is_buf = ~is_combiner & ~is_miner & has_machine
    can_deposit_buf = in_bounds & has_item & is_buf & (buf_empty | buf_same) & buf_space

    can_deposit = can_deposit_asm | can_deposit_buf
    transfer = jnp.where(can_deposit, jnp.int16(1), jnp.int16(0))

    # Apply.
    new_player_inv = state.player_inventory.at[player_idx, item_type].add(
        -transfer,
    )

    new_buf_type = jnp.where(
        can_deposit_buf,
        state.ent_buf_type.at[eidx].set(item_type.astype(jnp.int8)),
        state.ent_buf_type,
    )
    new_buf_count = jnp.where(
        can_deposit_buf,
        state.ent_buf_count.at[eidx].add(transfer),
        state.ent_buf_count,
    )

    new_asm_in_type = state.ent_asm_in_type
    new_asm_in_count = state.ent_asm_in_count
    new_asm_in_type = jnp.where(
        use_s0 & has_item,
        new_asm_in_type.at[eidx, 0].set(item_type.astype(jnp.int8)),
        new_asm_in_type,
    )
    new_asm_in_count = jnp.where(
        use_s0 & has_item,
        new_asm_in_count.at[eidx, 0].add(transfer),
        new_asm_in_count,
    )
    new_asm_in_type = jnp.where(
        use_s1 & has_item,
        new_asm_in_type.at[eidx, 1].set(item_type.astype(jnp.int8)),
        new_asm_in_type,
    )
    new_asm_in_count = jnp.where(
        use_s1 & has_item,
        new_asm_in_count.at[eidx, 1].add(transfer),
        new_asm_in_count,
    )

    return state.replace(
        player_inventory=new_player_inv,
        ent_buf_type=new_buf_type,
        ent_buf_count=new_buf_count,
        ent_asm_in_type=new_asm_in_type,
        ent_asm_in_count=new_asm_in_count,
    )


def withdraw_from_adjacent(
    state: EnvState,
    player_idx: int | jax.Array,
) -> EnvState:
    """Withdraw from the output slot of the machine in front of the player.

    Each machine exposes exactly one output slot at a time
    (``ent_asm_out`` for combiners mid-cycle, otherwise ``ent_buf``).
    A single WITHDRAW action pulls **as many items as can fit** —
    the min of what's in the slot and the player's remaining
    inventory space. That keeps agents from burning 100 ticks
    emptying a 100-ore pallet one item at a time.
    """
    tx, ty = get_tile_in_front(state, player_idx)
    map_h, map_w = state.map.shape
    in_bounds = (tx >= 0) & (tx < map_w) & (ty >= 0) & (ty < map_h)
    sx = jnp.clip(tx, 0, map_w - 1)
    sy = jnp.clip(ty, 0, map_h - 1)

    mt = jnp.where(in_bounds, state.machine_types[sy, sx], MachineType.NONE)
    is_machine = in_bounds & (mt != MachineType.NONE)

    max_e = state.ent_y.shape[0]
    eidx_raw = state.tile_entity[sy, sx]
    eidx = jnp.clip(eidx_raw, 0, max_e - 1)

    # Prefer ``asm_out`` when populated (combiner mid-cycle); otherwise
    # read from ``buf``. The two slots are mutually exclusive in
    # practice — combiners only use asm_out, buffer machines only buf.
    out_has = state.ent_asm_out_count[eidx] > 0
    buf_has = state.ent_buf_count[eidx] > 0
    use_asm = is_machine & out_has
    use_buf = is_machine & ~out_has & buf_has

    item_type = jnp.where(
        use_asm,
        state.ent_asm_out_type[eidx],
        state.ent_buf_type[eidx],
    ).astype(jnp.int32)

    player_count = state.player_inventory[player_idx, item_type]
    player_max = PLAYER_MAX_STACK[item_type]
    player_space = jnp.maximum(player_max - player_count, jnp.int32(0))

    available = jnp.where(
        use_asm,
        state.ent_asm_out_count[eidx].astype(jnp.int32),
        jnp.where(
            use_buf,
            state.ent_buf_count[eidx].astype(jnp.int32),
            jnp.int32(0),
        ),
    )

    # Transfer = min(available, player_space). Clamped non-negative
    # so a full inventory yields a no-op instead of a reverse move.
    transfer32 = jnp.minimum(available, player_space)
    transfer = jnp.where(
        is_machine,
        transfer32.astype(jnp.int16),
        jnp.int16(0),
    )
    can_withdraw_asm = use_asm & (transfer > 0)
    can_withdraw_buf = use_buf & (transfer > 0)

    new_player_inv = state.player_inventory.at[player_idx, item_type].add(
        transfer.astype(state.player_inventory.dtype),
    )

    # Decrement only the targeted entity's count.
    new_asm_out_count = jnp.where(
        can_withdraw_asm,
        state.ent_asm_out_count.at[eidx].add(-transfer),
        state.ent_asm_out_count,
    )
    # Type clear must also be scoped to the targeted entity — an
    # elementwise ``new_count == 0`` would clobber the recipe-output
    # marker on every other assembler whose cycle is mid-flight
    # (Phase 3 sets out_type at cycle start, before Phase 1 writes
    # count), killing those cycles on the next tick.
    clear_asm_type = can_withdraw_asm & (new_asm_out_count[eidx] == 0)
    new_asm_out_type = jnp.where(
        clear_asm_type,
        state.ent_asm_out_type.at[eidx].set(jnp.int8(0)),
        state.ent_asm_out_type,
    )

    new_buf_count = jnp.where(
        can_withdraw_buf,
        state.ent_buf_count.at[eidx].add(-transfer),
        state.ent_buf_count,
    )
    clear_buf_type = can_withdraw_buf & (new_buf_count[eidx] == 0)
    new_buf_type = jnp.where(
        clear_buf_type,
        state.ent_buf_type.at[eidx].set(jnp.int8(0)),
        state.ent_buf_type,
    )

    return state.replace(
        player_inventory=new_player_inv,
        ent_buf_type=new_buf_type,
        ent_buf_count=new_buf_count,
        ent_asm_out_type=new_asm_out_type,
        ent_asm_out_count=new_asm_out_count,
    )


def run_labs(state: EnvState) -> EnvState:
    """Consume every science pack sitting in any SCIENCE_LAB input slot.

    Greedy: whatever's in a lab's two input slots this tick is fully
    consumed. The per-type delta goes into ``science_consumed_step``
    for wrappers (e.g. :class:`ScienceTallyWrapper`) to integrate.

    Vectorised over all entities: one mask, one gather, one
    :func:`jax.ops.segment_sum`. No scatter — touches the
    scatter-free pattern the rest of the engine is converging toward.

    Args:
        state: Current environment state.

    Returns:
        State with lab input slots zeroed (where they held packs) and
        ``science_consumed_step`` populated with the summed counts.
    """
    is_lab = (state.ent_type == MachineType.SCIENCE_LAB) & (state.ent_y >= 0)
    # Shape (E, 2) after broadcasting.
    lab_mask = is_lab[:, None]
    slot_types = state.ent_asm_in_type
    slot_counts = state.ent_asm_in_count
    pack_idx = SCIENCE_PACK_INDEX[slot_types]  # (E, 2), -1 for non-packs.
    slot_is_pack = lab_mask & (pack_idx >= 0)
    counts_to_consume = jnp.where(slot_is_pack, slot_counts, 0).astype(jnp.int32)
    flat_idx = jnp.where(slot_is_pack, pack_idx, 0).reshape(-1)
    flat_cnt = counts_to_consume.reshape(-1)
    delta = jax.ops.segment_sum(flat_cnt, flat_idx, num_segments=NUM_SCIENCE_PACK_TYPES)
    new_in_types = jnp.where(slot_is_pack, jnp.int8(0), slot_types)
    new_in_counts = jnp.where(slot_is_pack, jnp.int16(0), slot_counts)
    return state.replace(
        ent_asm_in_type=new_in_types,
        ent_asm_in_count=new_in_counts,
        science_consumed_step=delta.astype(jnp.int32),
    )


def _handle_player_action(
    state: EnvState,
    params: EnvParams,
    action: int | jax.Array,
    player_idx: int | jax.Array,
) -> EnvState:
    """Handle a single player action.

    Args:
        state: Current environment state.
        params: Environment parameters (supplies the recipe table).
        action: Action to take.
        player_idx: Index of the player.

    Returns:
        Updated environment state.
    """
    # Pre-compute all derived action parameters (cheap indexing).
    recipe_idx = jnp.clip(action - CRAFT_BASE, 0, NUM_RECIPES - 1)
    place_item = PLACE_ACTION_TO_ITEM[
        jnp.clip(action - PLACE_BASE, 0, len(PLACE_ACTION_TO_ITEM) - 1)
    ]
    rotate_dir = ROTATE_ACTION_TO_DIR[jnp.clip(action - ROTATE_BASE, 0, 3)]
    deposit_item = jnp.clip(
        action - DEPOSIT_BASE + int(ItemType.COAL),
        0,
        NUM_ITEM_TYPES - 1,
    )

    # Map action to handler category (0-7).
    cat = jnp.int32(0)  # default: movement
    cat = jnp.where(action == Action.MINE, 1, cat)
    cat = jnp.where(
        (action >= Action.CRAFT_IRON_PLATE) & (action <= Action.CRAFT_CROSSING),
        2,
        cat,
    )
    cat = jnp.where(
        (action >= Action.PLACE_MINER) & (action <= Action.PLACE_CROSSING),
        3,
        cat,
    )
    cat = jnp.where(action == Action.PICKUP, 4, cat)
    cat = jnp.where(
        (action >= Action.ROTATE_LEFT) & (action <= Action.ROTATE_DOWN),
        5,
        cat,
    )
    cat = jnp.where(
        (action >= Action.DEPOSIT_COAL) & (action <= Action.DEPOSIT_CROSSING),
        6,
        cat,
    )
    cat = jnp.where(action == Action.WITHDRAW, 7, cat)

    # Single-dispatch: only the matching handler executes at runtime.
    return jax.lax.switch(
        cat,
        [
            lambda s: move_player(s, action, player_idx),
            lambda s: mine_block(s, player_idx),
            lambda s: craft_recipe(s, params, player_idx, recipe_idx),
            lambda s: place_machine(s, params, player_idx, place_item),
            lambda s: pickup_machine(s, player_idx),
            lambda s: set_machine_direction(s, player_idx, rotate_dir),
            lambda s: deposit_to_adjacent(s, player_idx, deposit_item),
            lambda s: withdraw_from_adjacent(s, player_idx),
        ],
        state,
    )


def factoriax_step(
    rng: jax.Array,
    state: EnvState,
    action: int | jax.Array,
    params: EnvParams,
) -> EnvState:
    """Execute one step of the environment.

    Args:
        rng: JAX random key (unused, kept for API compat).
        state: Current environment state.
        action: Action to take.
        params: Environment parameters.

    Returns:
        Updated environment state.
    """
    player_idx = state.selected_player
    # Reset the per-step science-lab delta before the action runs. Any
    # consumption this tick is written by ``run_labs`` below.
    state = state.replace(
        science_consumed_step=jnp.zeros(NUM_SCIENCE_PACK_TYPES, dtype=jnp.int32),
    )
    state = _handle_player_action(state, params, action, player_idx)
    state = update_all_machines(state, params)
    state = run_labs(state)
    return state.replace(timestep=state.timestep + 1)


def is_game_over(
    state: EnvState,
    params: EnvParams,
) -> jax.Array:
    """Check if the episode has ended.

    Args:
        state: Current environment state.
        params: Environment parameters.

    Returns:
        Boolean indicating game over.
    """
    return jnp.bool_(state.timestep >= params.max_timesteps)
