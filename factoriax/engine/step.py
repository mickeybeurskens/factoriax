"""Advance the environment one step.

A step is one player action followed by the whole factory running once.
:func:`factoriax_step` is the order: the action lands first, then every
machine on the map moves, then science labs drain, then the clock ticks.
An agent therefore sees the consequences of its action only on the next
observation.

Actions are compound. Rather than a cursor that selects a slot and a
separate key that acts on it, each action names its own target: there is one
action per placeable item, one per craftable item, and one per depositable
item. That makes the action space wide and every action meaningful on its
own, so nothing in the state has to remember what was selected.

Every player action here targets one tile, the one the player faces, and
each is masked rather than branched. An action that cannot happen writes the
state back unchanged and reports nothing.

Positions are ``(x, y)``, column first, while every grid lookup is ``[y, x]``.

The player-facing transfers here obey the same per-machine capacities as the
machine passes in :mod:`factoriax.engine.machines`, both reading
``MACHINE_MAX_STACK``. A player cannot put more into a belt than a miner
could.
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

# ROTATE_* offset (0..3) -> Direction value. (The PLACE/CRAFT/DEPOSIT
# offset->item tables are imported from factoriax.engine.actions, above.)
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
        Scalar bool. False for any negative coordinate or one at or past
        the far edge.
    """
    return (
        (position[0] >= 0)
        & (position[0] < map_width)
        & (position[1] >= 0)
        & (position[1] < map_height)
    )


def get_block_at(state: EnvState, position: jax.Array) -> jax.Array:
    """Read the terrain block at a position.

    Off-map positions get ``BlockType.OUT_OF_BOUNDS`` rather than raising,
    so a caller can look past an edge without checking first. The lookup is
    clipped before it happens, so nothing indexes out of range.

    Parameters
    ----------
    state
        State to read.
    position
        Tile to read as ``(x, y)``, column first.

    Returns
    -------
    jax.Array
        Scalar block id, or ``BlockType.OUT_OF_BOUNDS`` off the map.
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
    water, and a machine. Conveyor belts are the one exception and can be
    walked over, so a belt line does not wall a player off from its own
    factory.

    Other players are not considered, so two players can occupy one tile.

    Off-map positions are refused, because ``OUT_OF_BOUNDS`` is a solid
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

    Facing matters more than position, because every other player action
    targets the tile in front. Two ways to set it: a movement action turns
    the player and then moves them if the target is walkable, and a
    ``FACE_`` action turns them without moving.

    A movement action into a blocked tile still turns the player. That is
    deliberate: it means one action can always aim at a wall, a machine, or
    an ore patch in order to act on it next.

    Any action that is neither a movement nor a ``FACE_`` leaves the player
    where they are, facing as they were, so this is safe to call
    unconditionally.

    Parameters
    ----------
    state
        State to read.
    action
        Action id. Only the four movement and four ``FACE_`` values have
        an effect.
    player_idx
        Which player to move.

    Returns
    -------
    EnvState
        New state with ``player_positions`` and ``player_directions``
        updated for that player.
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
    params: EnvParams,
) -> EnvState:
    """Mine the block on the tile in front of the player.

    Targets the tile in the player's facing direction. NOOP when the
    target is out of bounds, not a mineable block, depleted, or when
    the player's inventory has no space for any of the resulting yield.
    Each successful action extracts up to ``params.player_mining_yield``
    units, capped by remaining tile resources and remaining inventory
    stack space.

    This is the only way to get ore without building a miner, and it is how
    a run bootstraps: the first miner has to be crafted from ore mined by
    hand.

    A tile emptied of resources turns to ``BlockType.DIRT``, the same as
    under a miner.

    Parameters
    ----------
    state
        State to read.
    player_idx
        Which player is mining.
    params
        Supplies ``player_mining_yield``, the amount per action before the
        tile and inventory caps apply.

    Returns
    -------
    EnvState
        New state with ``map``, ``player_inventory``, ``block_resources``,
        and ``items_mined`` updated. Unchanged when nothing was mined.
        Unlike :func:`factoriax.engine.machines.run_miners`, this tallies
        every mined item into ``items_mined``, not only the five ore types.
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

    One item per action, whatever the machine. That is the asymmetry with
    :func:`withdraw_from_adjacent`, which empties a slot in one action.

    Where the item lands depends on the machine. An assembler, furnace, or
    science lab takes it into ``ent_asm_in``, slot 0 when that is empty or
    already holds the same item and slot 1 otherwise. Every other machine
    takes it into ``ent_buf``, provided the buffer is empty or already holds
    the same item.

    A miner is the one machine that refuses a deposit outright. Its buffer
    is an output.

    Both routes honour ``MACHINE_MAX_STACK``, the same table the machine
    passes read, so a belt fills to 3 and a rocket refuses every item. A
    refused deposit leaves the item in the player's inventory.

    Parameters
    ----------
    state
        State to read.
    player_idx
        Which player is depositing.
    item_type
        Item to hand over. The action itself names it; there is no
        selection cursor.

    Returns
    -------
    EnvState
        New state with ``player_inventory``, ``ent_buf_type``,
        ``ent_buf_count``, ``ent_asm_in_type``, and ``ent_asm_in_count``
        updated. Unchanged when the deposit is refused.
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

    # Entity lookup for the target tile.
    max_e = state.ent_y.shape[0]
    eidx_raw = state.tile_entity[sy, sx]
    eidx = jnp.clip(eidx_raw, 0, max_e - 1)

    # Assemblers, furnaces, and science labs take a delivery into
    # ``ent_asm_in`` rather than ``ent_buf``. ``MACHINE_HAS_INPUT_SLOTS`` is
    # the one definition of that set; spelling the three kinds out here is
    # how this path and the belt pass drifted apart in the first place.
    max_stack = MACHINE_MAX_STACK[mt.astype(jnp.int32)]
    has_input_slots = MACHINE_HAS_INPUT_SLOTS[mt.astype(jnp.int32)]

    # Deposit into an input slot.
    in_t0 = state.ent_asm_in_type[eidx, 0]
    in_c0 = state.ent_asm_in_count[eidx, 0]
    in_t1 = state.ent_asm_in_type[eidx, 1]
    in_c1 = state.ent_asm_in_count[eidx, 1]

    slot0_ok = (in_c0 == 0) | ((in_t0 == item_type_arr) & (in_c0 < max_stack))
    slot1_ok = (in_c1 == 0) | ((in_t1 == item_type_arr) & (in_c1 < max_stack))
    use_s0 = has_input_slots & (max_stack > 0) & slot0_ok
    use_s1 = has_input_slots & (max_stack > 0) & ~use_s0 & slot1_ok

    can_deposit_asm = in_bounds & has_item & (use_s0 | use_s1)

    # Deposit into a buffer machine. Capacity comes from MACHINE_MAX_STACK,
    # so a belt takes 3 and a rocket takes none, matching what every
    # machine-driven transfer in factoriax.engine.machines enforces.
    is_miner = mt == Machine.MINER
    buf_empty = state.ent_buf_count[eidx] == 0
    buf_same = state.ent_buf_type[eidx] == item_type_arr
    buf_space = state.ent_buf_count[eidx] < max_stack
    is_buf = ~has_input_slots & ~is_miner & has_machine & (max_stack > 0)
    can_deposit_buf = in_bounds & has_item & is_buf & (buf_empty | buf_same) & buf_space

    can_deposit = can_deposit_asm | can_deposit_buf
    transfer = jnp.where(can_deposit, jnp.int16(1), jnp.int16(0))

    # Apply.
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

    Each machine exposes exactly one output slot at a time
    (``ent_asm_out`` for an assembler or furnace mid-cycle, otherwise
    ``ent_buf``).
    A single WITHDRAW action pulls **as many items as can fit** —
    the min of what's in the slot and the player's remaining
    inventory space. That keeps agents from burning 100 ticks
    emptying a 100-ore pallet one item at a time.

    Which slot is read is decided by what is in them, not by the machine
    kind: ``ent_asm_out`` when it holds anything, ``ent_buf`` otherwise. In
    practice the two never both hold items, because an assembler or furnace
    writes only
    ``asm_out`` and buffer machines only ``buf``.

    Emptying a slot clears its item type as well as its count, since the
    rest of the engine reads a cleared type as free to accept anything.

    Parameters
    ----------
    state
        State to read.
    player_idx
        Which player is withdrawing.

    Returns
    -------
    EnvState
        New state with ``player_inventory``, ``ent_buf_type``,
        ``ent_buf_count``, ``ent_asm_out_type``, and ``ent_asm_out_count``
        updated. Unchanged when the tile holds no machine, the machine
        holds nothing, or the player has no room for the item.
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

    # Prefer ``asm_out`` when populated (assembler or furnace mid-cycle);
    # read from ``buf``. The two slots are mutually exclusive in
    # practice: assemblers and furnaces only use asm_out, buffer machines
    # only buf.
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

    A lab is a sink. It has no countdown and no output: whatever sits in its
    two input slots at this point in the step is consumed outright, and the
    per-type amount is recorded in ``science_consumed_step`` for the reward
    functions and achievements to read. ``factoriax_step`` zeroes that field
    at the top of every step, so it is a per-step delta and not a running
    total.

    Only science packs are consumed. A non-pack item delivered into a lab's
    input slot stays there and blocks that slot.

    Only ``ent_asm_in`` is read. A lab's ``ent_buf`` is never touched, which
    is what makes a belt aimed at a lab a dead end. See
    :func:`factoriax.engine.machines.run_conveyor_belts`.

    Parameters
    ----------
    state
        State to read.

    Returns
    -------
    EnvState
        New state with the consumed slots of every lab cleared in
        ``ent_asm_in_type`` and ``ent_asm_in_count``, and
        ``science_consumed_step`` set to this step's per-type totals.
        The totals are zero when no lab held a pack.
    """
    is_lab = (state.ent_type == Machine.SCIENCE_LAB) & (state.ent_y >= 0)
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
    """Route one action to the function that carries it out.

    Sorts every action into one of nine categories and runs exactly that
    handler through :func:`jax.lax.switch`, so the cost of a step is one
    handler rather than all nine.

    The three wide action families, ``PLACE_``, ``CRAFT_``, and
    ``DEPOSIT_``, are half-open ranges anchored at their base offset and
    sized by their lookup table, so adding an item extends a range rather
    than needing a new branch. Anything that matches no category falls
    through to movement, which is a no-op for an action that is not a
    movement.

    ``CRAFT_`` dispatches through the item rather than the recipe index:
    the action names an output item, and the active recipe table is asked
    which of its rows produces it. A scenario can therefore reorder or
    replace its recipes without changing what an action means. A table with
    no recipe for that item yields -1, and
    :func:`factoriax.engine.crafting.craft_recipe` treats that as a no-op.

    Parameters
    ----------
    state
        State to read.
    params
        Supplies ``recipe_table`` for craft dispatch, and is passed through
        to the handlers that take it.
    action
        Action id.
    player_idx
        Which player is acting.

    Returns
    -------
    EnvState
        New state after the one matching handler ran.
    """
    # Pre-compute all derived action parameters (cheap indexing).
    # CRAFT dispatch: action -> output item (fixed) -> recipe row in the
    # active table (-1 when this table has no recipe for that item). Routing
    # through the item keeps recipe-list order out of the dispatch.
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

    # Map action to handler category (0-8). The three parametric families are
    # half-open ranges anchored at their *_BASE offset and sized by the family
    # table; the fixed interactions are matched by member identity.
    cat = jnp.int32(0)  # default: movement
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

    # Single-dispatch: only the matching handler executes at runtime.
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

    The order is fixed and is part of the contract. The player acts first,
    then :func:`factoriax.engine.machines.update_all_machines` runs every
    machine, then :func:`run_labs` drains science labs, then ``timestep``
    increments.

    Acting first means a player deposit reaches a machine in time for that
    machine to use it on the same step. Labs running last means a pack an
    arm delivered this step is consumed this step.

    Only ``state.selected_player`` acts. Other players hold position; there
    is no per-player action vector here.

    Parameters
    ----------
    rng
        Unused. Nothing in a step is random. Kept so the signature matches
        the gymnax-style step API that wraps it.
    state
        State to advance.
    action
        Action for the selected player.
    params
        Passed through to the action handlers and the machine passes.

    Returns
    -------
    EnvState
        New state one step on, with ``timestep`` incremented and
        ``science_consumed_step`` holding this step's totals rather than
        the previous step's.
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
    """Report whether the episode has run out of time.

    Time is the only ending condition in the base engine. There is no
    losing state and no goal that stops a run early; a scenario that wants
    one adds it in a wrapper.

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
