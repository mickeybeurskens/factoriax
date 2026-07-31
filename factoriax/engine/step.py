"""Advance the environment one step.

A step is one player action, and then one run of the whole factory.
:func:`factoriax_step` holds the order. The action lands first, then every
machine on the map moves, then the science labs consume their packs, then the
clock advances. An agent therefore sees the result of its action only in the
next observation.

The actions are compound. There is no cursor that selects a slot and no
separate key that acts on it. Each action names its own target: one action for
each placeable item, one for each craftable item, and one for each depositable
item. The action space is therefore wide, and each action means something on
its own. No field in the state has to hold a selection.

Every player action here targets one tile, the tile that the player faces. Each
one uses a mask and not a branch. An action that cannot happen writes the state
back unchanged and reports nothing.

A position is ``(x, y)``, the column first. Every grid lookup is ``[y, x]``.

The player transfers here obey the same per-machine capacities as the machine
passes in :mod:`factoriax.engine.machines`. Both read ``MACHINE_MAX_STACK``. A
player cannot put more into a belt than a miner can.
"""

import jax
import jax.numpy as jnp

from factoriax.engine.actions import (
    CRAFT_ACTION_TO_ITEM,
    DEPOSIT_ACTION_TO_ITEM,
    PLACE_ACTION_TO_ITEM,
)
from factoriax.engine.constants import (
    CRAFT_BASE,
    DEPOSIT_BASE,
    NUM_SCIENCE_PACK_TYPES,
    PLACE_BASE,
    ROTATE_BASE,
    Action,
    BlockType,
    Direction,
    Machine,
)
from factoriax.engine.crafting import craft_recipe
from factoriax.engine.machines import update_all_machines
from factoriax.engine.placement import (
    apply_repair,
    get_tile_in_front,
    pickup_machine,
    place_machine,
    set_machine_direction,
)
from factoriax.engine.state import EnvParams, EnvState
from factoriax.engine.tables import (
    BLOCK_TO_ITEM_ARRAY,
    DIRECTIONS,
    MACHINE_HAS_INPUT_SLOTS,
    MACHINE_MAX_STACK,
    MINEABLE_BLOCKS,
    PLAYER_MAX_STACK,
    SCIENCE_PACK_INDEX,
    SOLID_BLOCKS,
)

# ROTATE_* offset in 0..3 -> Direction value. The PLACE/CRAFT/DEPOSIT
# offset->item tables come from factoriax.engine.actions, in the imports above.
ROTATE_ACTION_TO_DIR = jnp.array(
    [Direction.LEFT, Direction.RIGHT, Direction.UP, Direction.DOWN],
    dtype=jnp.int32,
)


def is_position_in_bounds(
    position: jax.Array,
    map_width: int,
    map_height: int,
) -> jax.Array:
    """Report whether a position falls on the map.

    Parameters
    ----------
    position
        Tile to test as ``(x, y)``, column first.
    map_width
        Map width in tiles.
    map_height
        Map height in tiles.

    Returns
    -------
    jax.Array
        Scalar bool. The value is False for a negative coordinate, and for a
        coordinate at the far edge or past it.
    """
    return (
        (position[0] >= 0)
        & (position[0] < map_width)
        & (position[1] >= 0)
        & (position[1] < map_height)
    )


def get_block_at(state: EnvState, position: jax.Array) -> jax.Array:
    """Read the terrain block at a position.

    A position outside the map returns ``BlockType.OUT_OF_BOUNDS``, and the
    function raises nothing. A caller can therefore read past an edge with no
    test first. The function clips the index before the read, so nothing
    indexes outside the array.

    Parameters
    ----------
    state
        State to read.
    position
        Tile to read as ``(x, y)``, column first.

    Returns
    -------
    jax.Array
        Scalar block id, or ``BlockType.OUT_OF_BOUNDS`` outside the map.
    """
    map_height, map_width = state.map.shape
    in_bounds = is_position_in_bounds(position, map_width, map_height)
    clipped_x = jnp.clip(position[0], 0, map_width - 1)
    clipped_y = jnp.clip(position[1], 0, map_height - 1)
    return jnp.asarray(
        jnp.where(
            in_bounds,
            state.map[clipped_y, clipped_x],
            jnp.int8(BlockType.OUT_OF_BOUNDS),
        )
    )


def is_position_walkable(
    state: EnvState,
    position: jax.Array,
) -> jax.Array:
    """Report whether a player can stand on a position.

    Two things block a tile: a block in ``SOLID_BLOCKS``, such as stone or
    water, and a machine. A conveyor belt is the one exception, and a player
    can walk over it. A belt line therefore does not close a player out of its
    own factory.

    The function ignores the other players, so two players can stand on one
    tile.

    A position outside the map fails, because ``OUT_OF_BOUNDS`` is a solid
    block.

    Parameters
    ----------
    state
        State to read.
    position
        Tile to test as ``(x, y)``, column first.

    Returns
    -------
    jax.Array
        Scalar bool.
    """
    block = get_block_at(state, position)
    is_solid = jnp.any(block == SOLID_BLOCKS)
    map_height, map_width = state.map.shape
    in_bounds = is_position_in_bounds(position, map_width, map_height)
    clipped_x = jnp.clip(position[0], 0, map_width - 1)
    clipped_y = jnp.clip(position[1], 0, map_height - 1)
    mt = state.machine_types[clipped_y, clipped_x]
    has_blocking = in_bounds & ((mt != Machine.NONE) & (mt != Machine.CONVEYOR_BELT))
    return ~is_solid & ~has_blocking


def move_player(
    state: EnvState,
    action: int | jax.Array,
    player_idx: int | jax.Array,
) -> EnvState:
    """Move a player one tile, or turn them without moving.

    The facing matters more than the position, because every other player
    action targets the tile in front. Two actions set it. A movement action
    turns the player, and then moves them when the target tile is walkable. A
    ``FACE_`` action turns them and does not move them.

    A movement action into a blocked tile still turns the player, and this is
    deliberate. One action can therefore always aim at a wall, a machine, or an
    ore patch, and the next action can act on it.

    An action that is neither a movement nor a ``FACE_`` leaves the position
    and the facing as they were. A caller can therefore call this function for
    every action.

    Parameters
    ----------
    state
        State to read.
    action
        Action id. Only the four movement values and the four ``FACE_`` values
        have an effect.
    player_idx
        Player to move.

    Returns
    -------
    EnvState
        New state, with ``player_positions`` and ``player_directions`` updated
        for that player.
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

    # Facing: a movement sets the facing, and FACE_* sets it with no move.
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
    params: EnvParams,
) -> EnvState:
    """Mine the block on the tile in front of the player.

    The function targets the tile in the facing direction of the player. It
    does nothing in four cases: the tile is outside the map, no player can mine
    that block, the tile holds no ore left, or the inventory of the player has
    no room for the yield. An action that works extracts up to
    ``params.player_mining_yield`` units. The ore left in the tile and the
    stack space in the inventory both limit that amount.

    This function is the only way to get ore with no miner on the map, and it
    is how a run starts. A player must craft the first miner from ore that they
    mined by hand.

    A tile with no ore left becomes ``BlockType.DIRT``, the same as a tile
    under a miner.

    Parameters
    ----------
    state
        State to read.
    player_idx
        Player that mines the block.
    params
        Supplies ``player_mining_yield``, the amount for one action before the
        tile limit and the inventory limit apply.

    Returns
    -------
    EnvState
        New state, with ``map``, ``player_inventory``, ``block_resources``, and
        ``items_mined`` updated. The state is unchanged when the action mined
        nothing. This function adds every mined item to ``items_mined``, and
        not only the five ore types. :func:`factoriax.engine.machines.run_miners`
        counts the five ore types only.
    """
    tx, ty = get_tile_in_front(state, player_idx)
    h, w = state.map.shape
    in_bounds = (tx >= 0) & (tx < w) & (ty >= 0) & (ty < h)
    sx = jnp.clip(tx, 0, w - 1)
    sy = jnp.clip(ty, 0, h - 1)
    block_type = state.map[sy, sx]

    is_mineable = jnp.any(block_type == MINEABLE_BLOCKS) & in_bounds
    available = state.block_resources[sy, sx]
    has_resources = available > 0
    item_type = BLOCK_TO_ITEM_ARRAY[block_type.astype(jnp.int32)]

    current_count = state.player_inventory[player_idx, item_type]
    max_stack = PLAYER_MAX_STACK[item_type].astype(jnp.int16)
    space = jnp.maximum(max_stack - current_count, jnp.int16(0))
    has_space = space > 0

    desired = jnp.asarray(params.player_mining_yield, dtype=jnp.int16)
    extracted_raw = jnp.minimum(jnp.minimum(desired, space), available)
    can_mine = is_mineable & has_resources & has_space
    extracted = jnp.where(can_mine, extracted_raw, jnp.int16(0))

    new_resources = available - extracted
    is_depleted = new_resources <= 0

    new_inv = state.player_inventory.at[player_idx, item_type].add(extracted)
    new_block_resources = state.block_resources.at[sy, sx].set(
        jnp.where(can_mine, new_resources, available),
    )
    new_map = state.map.at[sy, sx].set(
        jnp.where(
            can_mine & is_depleted,
            jnp.int8(BlockType.DIRT),
            state.map[sy, sx],
        ),
    )
    new_items_mined = state.items_mined.at[item_type].add(
        extracted.astype(state.items_mined.dtype),
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
    """Hand one item to the machine in front of the player.

    One action moves one item, for every machine kind.
    :func:`withdraw_from_adjacent` is different: it empties a whole slot in one
    action.

    The machine decides where the item lands. An assembler, a furnace, or a
    science lab takes it into ``ent_asm_in``. It uses slot 0 when that slot is
    empty or already holds the same item, and slot 1 in every other case. Every
    other machine takes it into ``ent_buf``, and only when the buffer is empty
    or already holds the same item.

    A miner is the one machine that always refuses a deposit. Its buffer is an
    output.

    Both routes obey ``MACHINE_MAX_STACK``, the same table that the machine
    passes read. A belt therefore fills to 3, and a rocket refuses every item.
    A refused deposit leaves the item in the inventory of the player.

    Parameters
    ----------
    state
        State to read.
    player_idx
        Player that deposits the item.
    item_type
        Item to hand over. The action itself names it, and there is no
        selection cursor.

    Returns
    -------
    EnvState
        New state, with ``player_inventory``, ``ent_buf_type``,
        ``ent_buf_count``, ``ent_asm_in_type``, and ``ent_asm_in_count``
        updated. The state is unchanged when the function refuses the deposit.
    """
    item_type_arr = jnp.int32(item_type)
    tx, ty = get_tile_in_front(state, player_idx)
    map_h, map_w = state.map.shape
    in_bounds = (tx >= 0) & (tx < map_w) & (ty >= 0) & (ty < map_h)
    sx = jnp.clip(tx, 0, map_w - 1)
    sy = jnp.clip(ty, 0, map_h - 1)

    mt = jnp.where(in_bounds, state.machine_types[sy, sx], Machine.NONE)
    has_machine = mt != Machine.NONE
    player_count = state.player_inventory[player_idx, item_type_arr]
    has_item = player_count > 0

    # Find the entity on the target tile.
    max_e = state.ent_y.shape[0]
    eidx_raw = state.tile_entity[sy, sx]
    eidx = jnp.clip(eidx_raw, 0, max_e - 1)

    # Assemblers, furnaces, and science labs take a delivery into
    # ``ent_asm_in`` and not into ``ent_buf``. ``MACHINE_HAS_INPUT_SLOTS`` is
    # the one definition of that set. A list of the three kinds written out
    # here is what let this path and the belt pass disagree in the past.
    max_stack = MACHINE_MAX_STACK[mt.astype(jnp.int32)]
    has_input_slots = MACHINE_HAS_INPUT_SLOTS[mt.astype(jnp.int32)]

    # Deposit into one of the input slots.
    in_t0 = state.ent_asm_in_type[eidx, 0]
    in_c0 = state.ent_asm_in_count[eidx, 0]
    in_t1 = state.ent_asm_in_type[eidx, 1]
    in_c1 = state.ent_asm_in_count[eidx, 1]

    slot0_ok = (in_c0 == 0) | ((in_t0 == item_type_arr) & (in_c0 < max_stack))
    slot1_ok = (in_c1 == 0) | ((in_t1 == item_type_arr) & (in_c1 < max_stack))
    use_s0 = has_input_slots & (max_stack > 0) & slot0_ok
    use_s1 = has_input_slots & (max_stack > 0) & ~use_s0 & slot1_ok

    can_deposit_asm = in_bounds & has_item & (use_s0 | use_s1)

    # Deposit into the buffer of a machine. The capacity comes from
    # MACHINE_MAX_STACK, so a belt takes 3 and a rocket takes none. Every
    # machine transfer in factoriax.engine.machines uses the same limits.
    is_miner = mt == Machine.MINER
    buf_empty = state.ent_buf_count[eidx] == 0
    buf_same = state.ent_buf_type[eidx] == item_type_arr
    buf_space = state.ent_buf_count[eidx] < max_stack
    is_buf = ~has_input_slots & ~is_miner & has_machine & (max_stack > 0)
    can_deposit_buf = in_bounds & has_item & is_buf & (buf_empty | buf_same) & buf_space

    can_deposit = can_deposit_asm | can_deposit_buf
    transfer = jnp.where(can_deposit, jnp.int16(1), jnp.int16(0))

    # Write the result.
    new_player_inv = state.player_inventory.at[player_idx, item_type_arr].add(
        -transfer,
    )

    new_buf_type = jnp.where(
        can_deposit_buf,
        state.ent_buf_type.at[eidx].set(item_type_arr.astype(jnp.int8)),
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
        new_asm_in_type.at[eidx, 0].set(item_type_arr.astype(jnp.int8)),
        new_asm_in_type,
    )
    new_asm_in_count = jnp.where(
        use_s0 & has_item,
        new_asm_in_count.at[eidx, 0].add(transfer),
        new_asm_in_count,
    )
    new_asm_in_type = jnp.where(
        use_s1 & has_item,
        new_asm_in_type.at[eidx, 1].set(item_type_arr.astype(jnp.int8)),
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

    A machine exposes exactly one output slot at a time. That slot is
    ``ent_asm_out`` for an assembler or a furnace inside a cycle, and
    ``ent_buf`` in every other case. One WITHDRAW action takes as many items as
    fit: the smaller of the count in the slot and the free inventory space of
    the player. An agent therefore does not spend 100 steps to empty a pallet
    of 100 ore, one item at a time.

    The contents decide which slot the function reads, not the machine kind. It
    reads ``ent_asm_out`` when that slot holds items, and ``ent_buf`` in every
    other case. In practice the two never hold items at the same time, because
    an assembler and a furnace write only ``asm_out``, and a buffer machine
    writes only ``buf``.

    An empty slot loses its item type as well as its count, because the rest of
    the engine reads a cleared type as free to accept any item.

    Parameters
    ----------
    state
        State to read.
    player_idx
        Player that withdraws the items.

    Returns
    -------
    EnvState
        New state, with ``player_inventory``, ``ent_buf_type``,
        ``ent_buf_count``, ``ent_asm_out_type``, and ``ent_asm_out_count``
        updated. The state is unchanged when the tile holds no machine, when
        the machine holds nothing, or when the player has no room for the item.
    """
    tx, ty = get_tile_in_front(state, player_idx)
    map_h, map_w = state.map.shape
    in_bounds = (tx >= 0) & (tx < map_w) & (ty >= 0) & (ty < map_h)
    sx = jnp.clip(tx, 0, map_w - 1)
    sy = jnp.clip(ty, 0, map_h - 1)

    mt = jnp.where(in_bounds, state.machine_types[sy, sx], Machine.NONE)
    is_machine = in_bounds & (mt != Machine.NONE)

    max_e = state.ent_y.shape[0]
    eidx_raw = state.tile_entity[sy, sx]
    eidx = jnp.clip(eidx_raw, 0, max_e - 1)

    # Read ``asm_out`` when it holds items, which happens for an assembler or a
    # furnace inside a cycle. Read ``buf`` in every other case. The two slots
    # never hold items at the same time: an assembler and a furnace use
    # asm_out, and a buffer machine uses buf.
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

    # Transfer = min(available, player_space), clamped to zero or more, so a
    # full inventory does nothing and never moves items back.
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

    # Subtract from the count of the target entity only.
    new_asm_out_count = jnp.where(
        can_withdraw_asm,
        state.ent_asm_out_count.at[eidx].add(-transfer),
        state.ent_asm_out_count,
    )
    # The type clear must also touch the target entity only. An elementwise
    # ``new_count == 0`` overwrites the recipe-output marker on every other
    # assembler inside a cycle. Phase 3 sets out_type at the start of a cycle,
    # before Phase 1 writes the count, so those cycles stop on the next tick.
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

    A lab is a sink. It has no countdown and no output. It consumes everything
    in its two input slots at this point in the step. The function records the
    amount for each type in ``science_consumed_step``, and the reward functions
    and the achievements read that field. ``factoriax_step`` sets the field to
    zero at the start of every step, so it is a per-step delta and not a total.

    The function consumes science packs only. An item that is not a pack stays
    in the input slot of the lab and blocks that slot.

    The function reads ``ent_asm_in`` only. It never touches the ``ent_buf`` of
    a lab, and that is why a belt that points at a lab is a dead end. See
    :func:`factoriax.engine.machines.run_conveyor_belts`.

    Parameters
    ----------
    state
        State to read.

    Returns
    -------
    EnvState
        New state, with the consumed slots of every lab cleared in
        ``ent_asm_in_type`` and ``ent_asm_in_count``, and
        ``science_consumed_step`` set to the totals of this step for each type.
        The totals are zero when no lab held a pack.
    """
    is_lab = (state.ent_type == Machine.SCIENCE_LAB) & (state.ent_y >= 0)
    # Shape (E, 2) after the broadcast.
    lab_mask = is_lab[:, None]
    slot_types = state.ent_asm_in_type
    slot_counts = state.ent_asm_in_count
    pack_idx = SCIENCE_PACK_INDEX[slot_types]  # (E, 2), -1 for other items.
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
    """Route one action to the function that carries it out.

    The function sorts every action into one of nine categories, and runs
    exactly that one handler through :func:`jax.lax.switch`. A step therefore
    costs one handler, and not all nine.

    The three wide action blocks, ``PLACE_``, ``CRAFT_``, and ``DEPOSIT_``, are
    half-open ranges. Each range starts at its base offset and takes its length
    from its lookup table. A new item therefore makes a range longer and needs
    no new branch. An action that matches no category falls through to
    movement, and movement does nothing for an action that is not a movement.

    ``CRAFT_`` dispatches through the item, not through the recipe index. The
    action names an output item, and the active recipe table gives the row that
    produces that item. A scenario can therefore change the order of its
    recipes, or replace them, and the meaning of an action stays the same. A
    table with no recipe for that item gives -1, and
    :func:`factoriax.engine.crafting.craft_recipe` then does nothing.

    Parameters
    ----------
    state
        State to read.
    params
        Supplies ``recipe_table`` for the craft dispatch. The function also
        passes it to the handlers that take it.
    action
        Action id.
    player_idx
        Player that acts.

    Returns
    -------
    EnvState
        New state, after the one matching handler ran.
    """
    # Compute every derived action parameter first. These are cheap index
    # reads. The CRAFT dispatch goes action -> output item, which is fixed,
    # and then output item -> recipe row in the active table. The row is -1
    # when this table has no recipe for that item. The route through the item
    # keeps the order of the recipe list out of the dispatch.
    craft_offset = jnp.clip(action - CRAFT_BASE, 0, len(CRAFT_ACTION_TO_ITEM) - 1)
    craft_item = CRAFT_ACTION_TO_ITEM[craft_offset]
    recipe_idx = params.recipe_table.output_to_recipe[craft_item]
    place_item = PLACE_ACTION_TO_ITEM[
        jnp.clip(action - PLACE_BASE, 0, len(PLACE_ACTION_TO_ITEM) - 1)
    ]
    rotate_dir = ROTATE_ACTION_TO_DIR[jnp.clip(action - ROTATE_BASE, 0, 3)]
    deposit_item = DEPOSIT_ACTION_TO_ITEM[
        jnp.clip(action - DEPOSIT_BASE, 0, len(DEPOSIT_ACTION_TO_ITEM) - 1)
    ]

    # Map the action to a handler category in 0 to 8. The three parametric
    # blocks are half-open ranges that start at their *_BASE offset and take
    # their length from the block table. The fixed interactions match on the
    # member value.
    cat = jnp.int32(0)  # the default is movement
    cat = jnp.where(action == Action.MINE, 1, cat)
    cat = jnp.where(
        (action >= CRAFT_BASE) & (action < CRAFT_BASE + len(CRAFT_ACTION_TO_ITEM)),
        2,
        cat,
    )
    cat = jnp.where(
        (action >= PLACE_BASE) & (action < PLACE_BASE + len(PLACE_ACTION_TO_ITEM)),
        3,
        cat,
    )
    cat = jnp.where(action == Action.PICKUP, 4, cat)
    cat = jnp.where(
        (action >= ROTATE_BASE) & (action < ROTATE_BASE + len(ROTATE_ACTION_TO_DIR)),
        5,
        cat,
    )
    cat = jnp.where(
        (action >= DEPOSIT_BASE)
        & (action < DEPOSIT_BASE + len(DEPOSIT_ACTION_TO_ITEM)),
        6,
        cat,
    )
    cat = jnp.where(action == Action.WITHDRAW, 7, cat)
    cat = jnp.where(action == Action.REPAIR, 8, cat)

    # Single dispatch: only the matching handler runs.
    new_state: EnvState = jax.lax.switch(
        cat,
        [
            lambda s: move_player(s, action, player_idx),
            lambda s: mine_block(s, player_idx, params),
            lambda s: craft_recipe(s, params, player_idx, recipe_idx),
            lambda s: place_machine(s, params, player_idx, place_item),
            lambda s: pickup_machine(s, params, player_idx),
            lambda s: set_machine_direction(s, player_idx, rotate_dir),
            lambda s: deposit_to_adjacent(s, player_idx, deposit_item),
            lambda s: withdraw_from_adjacent(s, player_idx),
            lambda s: apply_repair(s, params, player_idx),
        ],
        state,
    )
    return new_state


def factoriax_step(
    rng: jax.Array,
    state: EnvState,
    action: int | jax.Array,
    params: EnvParams,
) -> EnvState:
    """Apply one action, run the factory, and advance the clock.

    The order is fixed and is part of the contract. The player acts first. Then
    :func:`factoriax.engine.machines.update_all_machines` runs every machine.
    Then :func:`run_labs` consumes the packs in the science labs. Then
    ``timestep`` increases by 1.

    The player acts first, so a player deposit reaches a machine in time for
    that machine to use it in the same step. The labs run last, so a pack that
    an arm delivered in this step is also consumed in this step.

    Only ``state.selected_player`` acts. Every other player keeps its position.
    There is no per-player action vector here.

    Parameters
    ----------
    rng
        The function does not read this argument. Nothing in a step is random.
        It is present so that the signature matches the gymnax-style step API
        around it.
    state
        State to advance.
    action
        Action for the selected player.
    params
        The function passes this to the action handlers and to the machine
        passes.

    Returns
    -------
    EnvState
        New state, one step later, with ``timestep`` one higher.
        ``science_consumed_step`` holds the totals of this step, not the totals
        of the step before.
    """
    player_idx = state.selected_player
    # Set the per-step science-lab delta to zero before the action runs.
    # ``run_labs`` below writes the amount that this tick consumes.
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
    """Report whether the episode has run out of time.

    Time is the only end condition in the base engine. There is no state in
    which a player loses, and no goal that stops a run early. A scenario that
    wants one adds it in a wrapper.

    Parameters
    ----------
    state
        State to test.
    params
        Supplies ``max_timesteps``.

    Returns
    -------
    jax.Array
        Scalar bool. True once ``timestep`` reaches ``max_timesteps``.
    """
    return jnp.asarray(state.timestep >= params.max_timesteps, dtype=jnp.bool_)
