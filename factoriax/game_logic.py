"""Game logic for player movement and environment stepping.

Uses the pouch inventory model: compound actions name specific item
types for placement, deposit, and withdraw. No slot cursors or
navigation actions exist in the environment.
"""

import jax
import jax.numpy as jnp
from jax import lax

from factoriax.biters import update_biters, update_scent_field
from factoriax.constants import (
    BLOCK_TO_ITEM_ARRAY,
    CRAFT_BASE,
    DEFAULT_MACHINE_MAX_HEALTH,
    DEPOSIT_BASE,
    DIRECTIONS,
    MACHINE_MAX_STACK,
    MACHINE_MAX_TYPES,
    MACHINE_TO_RECIPE,
    MINEABLE_BLOCKS,
    NUM_ITEM_TYPES,
    PLACE_ACTION_TO_ITEM,
    PLACE_BASE,
    PLAYER_MAX_STACK,
    RESEARCH_ACTION_TO_PACK,
    RESEARCH_COST,
    SCIENCE_PACK_TO_TECH,
    SOLID_BLOCKS,
    TURN_LEFT_MAP,
    TURN_RIGHT_MAP,
    WITHDRAW_BASE,
    Action,
    BlockType,
    Direction,
    ItemType,
    MachineType,
)
from factoriax.crafting import (
    start_crafting,
    update_crafting,
)
from factoriax.inventory import (
    can_add_to_machine,
    remove_from_pouch,
)
from factoriax.machines import update_all_machines
from factoriax.placement import get_tile_in_front, pickup_machine, place_machine
from factoriax.recipes import (
    ASSEMBLER_RECIPE_INPUT_COUNTS,
    ASSEMBLER_RECIPE_INPUT_ITEMS,
    ASSEMBLER_RECIPE_OUTPUTS,
    MAX_RECIPE_INPUTS,
    RECIPE_INPUT_COUNTS,
    RECIPE_INPUT_ITEMS,
)
from factoriax.state import EnvParams, EnvState


def is_position_in_bounds(
    position: jax.Array, map_width: int, map_height: int,
) -> jax.Array:
    """Check if a position is within map boundaries.

    Args:
        position: (x, y) coordinates to check.
        map_width: Width of the map.
        map_height: Height of the map.

    Returns:
        Boolean indicating whether position is in bounds.
    """
    x, y = position[0], position[1]
    return (x >= 0) & (x < map_width) & (y >= 0) & (y < map_height)


def get_block_at(state: EnvState, position: jax.Array) -> jax.Array:
    """Get the block type at a position.

    Args:
        state: Current environment state.
        position: (x, y) coordinates to query.

    Returns:
        Block type at the position, or OUT_OF_BOUNDS.
    """
    map_height, map_width = state.map.shape
    in_bounds = is_position_in_bounds(position, map_width, map_height)
    block: jax.Array = lax.cond(
        in_bounds,
        lambda: state.map[position[1], position[0]],
        lambda: jnp.int32(BlockType.OUT_OF_BOUNDS),
    )
    return block


def is_position_walkable(
    state: EnvState, position: jax.Array,
) -> jax.Array:
    """Check if a position can be walked on.

    Args:
        state: Current environment state.
        position: (x, y) coordinates to check.

    Returns:
        Boolean indicating whether position is walkable.
    """
    block = get_block_at(state, position)
    is_solid = jnp.any(block == SOLID_BLOCKS)

    map_height, map_width = state.map.shape
    in_bounds = is_position_in_bounds(position, map_width, map_height)
    has_blocking_machine = lax.cond(
        in_bounds,
        lambda: (
            (state.machine_types[position[1], position[0]]
             != MachineType.NONE)
            & (state.machine_types[position[1], position[0]]
               != MachineType.CONVEYOR_BELT)
        ),
        lambda: jnp.bool_(False),
    )

    return ~is_solid & ~has_blocking_machine


def move_player(
    state: EnvState,
    action: int | jax.Array,
    player_idx: int | jax.Array,
) -> EnvState:
    """Move or turn a player.

    Args:
        state: Current environment state.
        action: Action to take (from Action enum).
        player_idx: Index of the player to move.

    Returns:
        Updated environment state.
    """
    current_position = state.player_positions[player_idx]
    facing = state.player_directions[player_idx]

    act_to_dir = jnp.array(
        [0, Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT],
        dtype=jnp.int32,
    )
    safe_act = jnp.clip(action, 0, act_to_dir.shape[0] - 1)
    offset = DIRECTIONS[act_to_dir[safe_act]]

    is_move = (action >= Action.UP) & (action <= Action.RIGHT)
    target = current_position + offset
    can_move = is_position_walkable(state, target) & is_move
    final_position = jnp.where(can_move, target, current_position)

    face_to_dir = jnp.array(
        [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT],
        dtype=jnp.int32,
    )
    is_face = (action >= Action.FACE_UP) & (action <= Action.FACE_RIGHT)
    face_idx = jnp.clip(action - Action.FACE_UP, 0, 3)

    new_facing = jnp.where(
        is_face,
        face_to_dir[face_idx],
        jnp.where(
            action == Action.TURN_LEFT,
            TURN_LEFT_MAP[facing],
            jnp.where(
                action == Action.TURN_RIGHT,
                TURN_RIGHT_MAP[facing],
                facing,
            ),
        ),
    )

    return state.replace(
        player_positions=state.player_positions.at[player_idx].set(
            final_position,
        ),
        player_directions=state.player_directions.at[player_idx].set(
            new_facing,
        ),
    )


def mine_block(
    state: EnvState, player_idx: int | jax.Array,
) -> EnvState:
    """Attempt to mine the block at a player's current position.

    Adds the mined item directly to the player's pouch inventory.

    Args:
        state: Current environment state.
        player_idx: Index of the player performing the mining.

    Returns:
        Updated environment state.
    """
    player_pos = state.player_positions[player_idx]
    px, py = player_pos[0], player_pos[1]
    block_type = state.map[py, px]

    is_mineable = jnp.any(block_type == MINEABLE_BLOCKS)
    has_resources = state.block_resources[py, px] > 0
    item_type = BLOCK_TO_ITEM_ARRAY[block_type]

    player_inv = state.player_inventory[player_idx]
    current_count = player_inv[item_type]
    max_stack = PLAYER_MAX_STACK[item_type]
    has_space = current_count < max_stack

    can_mine = is_mineable & has_resources & has_space

    new_inv = lax.cond(
        can_mine,
        lambda: state.player_inventory.at[player_idx, item_type].set(
            current_count + 1,
        ),
        lambda: state.player_inventory,
    )

    new_resources = state.block_resources[py, px] - 1
    is_depleted = new_resources <= 0

    new_block_resources = lax.cond(
        can_mine,
        lambda: state.block_resources.at[py, px].set(new_resources),
        lambda: state.block_resources,
    )

    new_map = lax.cond(
        can_mine & is_depleted,
        lambda: state.map.at[py, px].set(jnp.int32(BlockType.DIRT)),
        lambda: state.map,
    )

    new_items_mined = lax.cond(
        can_mine,
        lambda: state.items_mined.at[item_type].add(1),
        lambda: state.items_mined,
    )

    return state.replace(
        map=new_map,
        player_inventory=new_inv,
        block_resources=new_block_resources,
        items_mined=new_items_mined,
    )


# Clockwise direction cycle for rotate_adjacent.
_NEXT_DIR = jnp.zeros(len(Direction) + 1, dtype=jnp.int32)
_NEXT_DIR = _NEXT_DIR.at[Direction.DOWN].set(Direction.RIGHT)
_NEXT_DIR = _NEXT_DIR.at[Direction.RIGHT].set(Direction.UP)
_NEXT_DIR = _NEXT_DIR.at[Direction.UP].set(Direction.LEFT)
_NEXT_DIR = _NEXT_DIR.at[Direction.LEFT].set(Direction.DOWN)


def rotate_adjacent(
    state: EnvState, player_idx: int | jax.Array,
) -> EnvState:
    """Rotate the machine in front of the player one step clockwise.

    Args:
        state: Current environment state.
        player_idx: Index of the acting player.

    Returns:
        Updated state with the machine's direction advanced.
    """
    target_x, target_y = get_tile_in_front(state, player_idx)
    map_h, map_w = state.map.shape
    in_bounds = is_position_in_bounds(
        jnp.array([target_x, target_y]), map_w, map_h,
    )

    machine_type = jnp.where(
        in_bounds,
        state.machine_types[target_y, target_x],
        MachineType.NONE,
    )
    has_machine = machine_type != MachineType.NONE

    current_dir = jnp.where(
        in_bounds,
        state.machine_direction[target_y, target_x],
        jnp.int32(0),
    )
    new_dir = _NEXT_DIR[current_dir]

    def do_rotate(s: EnvState) -> EnvState:
        return s.replace(
            machine_direction=s.machine_direction.at[
                target_y, target_x
            ].set(new_dir),
        )

    return lax.cond(
        in_bounds & has_machine, do_rotate, lambda s: s, state,
    )


def _is_valid_deposit_item(
    machine_type: jax.Array,
    item_type: jax.Array,
    recipe: jax.Array,
) -> jax.Array:
    """Check if an item type can be deposited into a machine.

    Miners accept COAL as fuel. Assemblers accept recipe inputs.
    Chests accept anything. Belts/arms accept anything (one type).

    Args:
        machine_type: MachineType of the target machine.
        item_type: ItemType to deposit.
        recipe: Machine's selected recipe index.

    Returns:
        Boolean: True if the item type is a valid deposit.
    """
    # Miner: only coal (fuel).
    miner_ok = item_type == int(ItemType.COAL)

    # Assembler: must match one of the recipe's input items.
    asm_inputs = ASSEMBLER_RECIPE_INPUT_ITEMS[recipe]
    asm_counts = ASSEMBLER_RECIPE_INPUT_COUNTS[recipe]
    asm_ok = jnp.any(
        (item_type == asm_inputs) & (asm_counts > 0),
    )

    # Chest/Belt/Arm: accept anything.
    storage_ok = jnp.bool_(True)

    is_miner = machine_type == int(MachineType.MINER)
    is_asm = machine_type == int(MachineType.ASSEMBLER)

    return jnp.where(is_miner, miner_ok, jnp.where(is_asm, asm_ok, storage_ok))


def _is_valid_withdraw_item(
    machine_type: jax.Array,
    item_type: jax.Array,
    recipe: jax.Array,
) -> jax.Array:
    """Check if an item type can be withdrawn from a machine.

    Miners: withdraw any item (ore output or leftover fuel).
    Assemblers: withdraw recipe output. Chests: withdraw anything.
    Belts/Arms: withdraw anything.

    Args:
        machine_type: MachineType of the target machine.
        item_type: ItemType to withdraw.
        recipe: Machine's selected recipe index.

    Returns:
        Boolean: True if withdrawal is valid.
    """
    # Assembler: only the recipe's output item.
    asm_output = ASSEMBLER_RECIPE_OUTPUTS[recipe]
    asm_ok = item_type == asm_output

    is_asm = machine_type == int(MachineType.ASSEMBLER)

    # Everything else: withdraw anything that's present.
    return jnp.where(is_asm, asm_ok, jnp.bool_(True))


def deposit_to_adjacent(
    state: EnvState,
    player_idx: int | jax.Array,
    item_type: int | jax.Array,
) -> EnvState:
    """Deposit items of a specific type into the machine in front.

    The item type comes from the compound action (e.g., DEPOSIT_IRON).
    Transfers up to the machine's capacity from the player's pouch.

    Args:
        state: Current environment state.
        player_idx: Index of the acting player.
        item_type: ItemType to deposit.

    Returns:
        Updated state with items transferred, or unchanged if invalid.
    """
    item_type = jnp.int32(item_type)
    target_x, target_y = get_tile_in_front(state, player_idx)
    map_h, map_w = state.map.shape
    in_bounds = is_position_in_bounds(
        jnp.array([target_x, target_y]), map_w, map_h,
    )

    machine_type = jnp.where(
        in_bounds,
        state.machine_types[target_y, target_x],
        MachineType.NONE,
    )
    has_machine = machine_type != MachineType.NONE

    player_count = state.player_inventory[player_idx, item_type]
    has_item = player_count > 0

    recipe = jnp.where(
        in_bounds,
        state.machine_selected_recipe[target_y, target_x],
        jnp.int32(0),
    )

    valid_item = _is_valid_deposit_item(machine_type, item_type, recipe)

    machine_inv = jnp.where(
        in_bounds,
        state.machine_inventory[target_y, target_x],
        jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.int16),
    )

    max_types = MACHINE_MAX_TYPES[machine_type]
    max_stack = MACHINE_MAX_STACK[machine_type]

    can_add = can_add_to_machine(
        machine_inv, item_type, jnp.int32(1), max_types, max_stack,
    )

    can_deposit = (
        in_bounds & has_machine & has_item & valid_item & can_add
    )

    def do_deposit(s: EnvState) -> EnvState:
        m_inv = s.machine_inventory[target_y, target_x].astype(
            jnp.int32,
        )
        m_count = m_inv[item_type]
        space = max_stack - m_count
        transfer = jnp.minimum(player_count, space)

        new_m_inv = m_inv.at[item_type].set(m_count + transfer)
        new_p_count = player_count - transfer

        return s.replace(
            machine_inventory=s.machine_inventory.at[
                target_y, target_x
            ].set(new_m_inv.astype(jnp.int16)),
            player_inventory=s.player_inventory.at[
                player_idx, item_type
            ].set(new_p_count),
        )

    return lax.cond(can_deposit, do_deposit, lambda s: s, state)


def withdraw_from_adjacent(
    state: EnvState,
    player_idx: int | jax.Array,
    item_type: int | jax.Array,
) -> EnvState:
    """Withdraw items of a specific type from the machine in front.

    The item type comes from the compound action (e.g., WITHDRAW_IRON).
    Transfers the full stack from the machine to the player's pouch.

    Args:
        state: Current environment state.
        player_idx: Index of the acting player.
        item_type: ItemType to withdraw.

    Returns:
        Updated state with items transferred, or unchanged if invalid.
    """
    item_type = jnp.int32(item_type)
    target_x, target_y = get_tile_in_front(state, player_idx)
    map_h, map_w = state.map.shape
    in_bounds = is_position_in_bounds(
        jnp.array([target_x, target_y]), map_w, map_h,
    )

    machine_type = jnp.where(
        in_bounds,
        state.machine_types[target_y, target_x],
        MachineType.NONE,
    )
    has_machine = machine_type != MachineType.NONE

    machine_inv = jnp.where(
        in_bounds,
        state.machine_inventory[target_y, target_x],
        jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.int16),
    )
    machine_count = machine_inv[item_type].astype(jnp.int32)
    has_items = machine_count > 0

    recipe = jnp.where(
        in_bounds,
        state.machine_selected_recipe[target_y, target_x],
        jnp.int32(0),
    )
    valid_item = _is_valid_withdraw_item(
        machine_type, item_type, recipe,
    )

    player_count = state.player_inventory[player_idx, item_type]
    player_max = PLAYER_MAX_STACK[item_type]
    has_space = player_count < player_max

    can_withdraw = (
        in_bounds & has_machine & has_items & valid_item & has_space
    )

    def do_withdraw(s: EnvState) -> EnvState:
        space = player_max - player_count
        transfer = jnp.minimum(machine_count, space)

        new_m_count = machine_count - transfer
        new_m_inv = s.machine_inventory[target_y, target_x].astype(
            jnp.int32,
        )
        new_m_inv = new_m_inv.at[item_type].set(new_m_count)

        return s.replace(
            machine_inventory=s.machine_inventory.at[
                target_y, target_x
            ].set(new_m_inv.astype(jnp.int16)),
            player_inventory=s.player_inventory.at[
                player_idx, item_type
            ].set(player_count + transfer),
        )

    return lax.cond(can_withdraw, do_withdraw, lambda s: s, state)


def apply_research(
    state: EnvState,
    player_idx: int | jax.Array,
    science_pack_type: int | jax.Array,
) -> EnvState:
    """Consume one science pack to advance research.

    The science pack type comes from the compound action
    (e.g., RESEARCH_BASIC maps to BASIC_SCIENCE_PACK).

    Args:
        state: Current environment state.
        player_idx: Index of the player.
        science_pack_type: ItemType of the science pack to consume.

    Returns:
        Updated state (unchanged if invalid).
    """
    science_pack_type = jnp.int32(science_pack_type)
    count = state.player_inventory[player_idx, science_pack_type]
    has_item = count > 0

    tech_idx = SCIENCE_PACK_TO_TECH[science_pack_type]
    already_unlocked = state.research_unlocked[tech_idx]
    can_research = has_item & ~already_unlocked

    new_inv = state.player_inventory.at[player_idx, science_pack_type].set(
        jnp.where(can_research, count - 1, count),
    )

    new_progress = state.research_progress.at[tech_idx].add(
        jnp.where(can_research, 1, 0),
    )
    new_unlocked = state.research_unlocked.at[tech_idx].set(
        state.research_unlocked[tech_idx]
        | (new_progress[tech_idx] >= RESEARCH_COST),
    )

    return state.replace(
        player_inventory=new_inv,
        research_progress=new_progress,
        research_unlocked=new_unlocked,
    )


def repair_machine(
    state: EnvState, player_idx: int | jax.Array,
) -> EnvState:
    """Repair the machine on the tile in front of the player.

    Consumes the recipe cost from the player's pouch and restores
    health to full.

    Args:
        state: Current environment state.
        player_idx: Index of the player.

    Returns:
        Updated state (unchanged if repair is not possible).
    """
    tx, ty = get_tile_in_front(state, player_idx)
    map_h, map_w = state.map.shape
    in_bounds = (tx >= 0) & (tx < map_w) & (ty >= 0) & (ty < map_h)

    machine_type = jnp.where(
        in_bounds, state.machine_types[ty, tx], MachineType.NONE,
    )
    has_machine = machine_type != MachineType.NONE
    current_health = jnp.where(
        in_bounds, state.machine_health[ty, tx], 0,
    )
    needs_repair = current_health < DEFAULT_MACHINE_MAX_HEALTH

    recipe_idx = jnp.clip(
        MACHINE_TO_RECIPE[machine_type], 0, MAX_RECIPE_INPUTS,
    )
    is_repairable = MACHINE_TO_RECIPE[machine_type] >= 0

    inv = state.player_inventory[player_idx]
    recipe_items = RECIPE_INPUT_ITEMS[recipe_idx]
    recipe_counts = RECIPE_INPUT_COUNTS[recipe_idx]

    def _has_ingredient(idx: jax.Array) -> jax.Array:
        needed_item = recipe_items[idx]
        needed_count = recipe_counts[idx]
        is_needed = (needed_item != int(ItemType.EMPTY)) & (
            needed_count > 0
        )
        has_it = inv[needed_item] >= needed_count
        return ~is_needed | has_it

    can_afford = jnp.all(
        jax.vmap(_has_ingredient)(jnp.arange(MAX_RECIPE_INPUTS)),
    )

    can_repair = (
        has_machine & needs_repair & is_repairable
        & can_afford & in_bounds
    )

    def do_repair(s: EnvState) -> EnvState:
        p_inv = s.player_inventory[player_idx]

        def consume_one(
            inv: jax.Array, idx: jax.Array,
        ) -> tuple[jax.Array, None]:
            needed_item = recipe_items[idx]
            needed_count = recipe_counts[idx]
            is_needed = (needed_item != int(ItemType.EMPTY)) & (
                needed_count > 0
            )
            new_inv, _ = remove_from_pouch(
                inv, needed_item, needed_count,
            )
            return jnp.where(is_needed, new_inv, inv), None

        p_inv, _ = lax.scan(
            consume_one, p_inv, jnp.arange(MAX_RECIPE_INPUTS),
        )

        return s.replace(
            player_inventory=s.player_inventory.at[player_idx].set(
                p_inv,
            ),
            machine_health=s.machine_health.at[ty, tx].set(
                DEFAULT_MACHINE_MAX_HEALTH,
            ),
        )

    return lax.cond(can_repair, do_repair, lambda s: s, state)


def _handle_player_action(
    state: EnvState,
    action: int | jax.Array,
    player_idx: int | jax.Array,
) -> EnvState:
    """Handle a single player action.

    Dispatches compound actions to the appropriate handler using
    arithmetic on contiguous action ranges.

    Args:
        state: Current environment state.
        action: Action to take.
        player_idx: Index of the player.

    Returns:
        Updated environment state.
    """
    is_mine = action == Action.MINE
    is_pickup = action == Action.PICKUP
    is_rotate = action == Action.ROTATE
    is_repair = action == Action.REPAIR

    # Placement: PLACE_MINER .. PLACE_ROCKET.
    is_place = (action >= Action.PLACE_MINER) & (
        action <= Action.PLACE_ROCKET
    )
    place_item = PLACE_ACTION_TO_ITEM[
        jnp.clip(action - PLACE_BASE, 0, len(PLACE_ACTION_TO_ITEM) - 1)
    ]

    # Crafting: CRAFT_MINER .. CRAFT_ASSEMBLER.
    is_craft = (action >= Action.CRAFT_MINER) & (
        action <= Action.CRAFT_ASSEMBLER
    )
    recipe_idx = jnp.clip(action - CRAFT_BASE, 0, 4)

    # Research: RESEARCH_BASIC .. RESEARCH_ADVANCED.
    is_research = (action >= Action.RESEARCH_BASIC) & (
        action <= Action.RESEARCH_ADVANCED
    )
    research_pack = RESEARCH_ACTION_TO_PACK[
        jnp.clip(
            action - Action.RESEARCH_BASIC, 0,
            len(RESEARCH_ACTION_TO_PACK) - 1,
        )
    ]

    # Deposit: DEPOSIT_COAL .. DEPOSIT_ADVANCED_SCIENCE.
    is_deposit = (action >= Action.DEPOSIT_COAL) & (
        action <= Action.DEPOSIT_ADVANCED_SCIENCE
    )
    deposit_item = jnp.clip(
        action - DEPOSIT_BASE + int(ItemType.COAL), 0,
        NUM_ITEM_TYPES - 1,
    )

    # Withdraw: WITHDRAW_COAL .. WITHDRAW_ADVANCED_SCIENCE.
    is_withdraw = (action >= Action.WITHDRAW_COAL) & (
        action <= Action.WITHDRAW_ADVANCED_SCIENCE
    )
    withdraw_item = jnp.clip(
        action - WITHDRAW_BASE + int(ItemType.COAL), 0,
        NUM_ITEM_TYPES - 1,
    )

    # Dispatch.
    state = lax.cond(
        is_mine,
        lambda s: mine_block(s, player_idx),
        lambda s: s, state,
    )
    state = lax.cond(
        is_craft,
        lambda s: start_crafting(s, player_idx, recipe_idx),
        lambda s: s, state,
    )
    state = lax.cond(
        is_place,
        lambda s: place_machine(s, player_idx, place_item),
        lambda s: s, state,
    )
    state = lax.cond(
        is_pickup,
        lambda s: pickup_machine(s, player_idx),
        lambda s: s, state,
    )
    state = lax.cond(
        is_deposit,
        lambda s: deposit_to_adjacent(s, player_idx, deposit_item),
        lambda s: s, state,
    )
    state = lax.cond(
        is_withdraw,
        lambda s: withdraw_from_adjacent(
            s, player_idx, withdraw_item,
        ),
        lambda s: s, state,
    )
    state = lax.cond(
        is_rotate,
        lambda s: rotate_adjacent(s, player_idx),
        lambda s: s, state,
    )
    state = lax.cond(
        is_research,
        lambda s: apply_research(s, player_idx, research_pack),
        lambda s: s, state,
    )
    state = lax.cond(
        is_repair,
        lambda s: repair_machine(s, player_idx),
        lambda s: s, state,
    )

    is_movement = ~(
        is_mine | is_craft | is_place | is_pickup
        | is_deposit | is_withdraw | is_rotate
        | is_research | is_repair
    )
    state = lax.cond(
        is_movement,
        lambda s: move_player(s, action, player_idx),
        lambda s: s, state,
    )

    return state


def factoriax_step(
    rng: jax.Array,
    state: EnvState,
    action: int | jax.Array,
    params: EnvParams,
) -> EnvState:
    """Execute one step of the environment.

    Processes in order:
    1. Selected player action
    2. Crafting progress for all players
    3. All machine updates
    4. Scent field and biter updates
    5. Timestep increment

    Args:
        rng: JAX random key.
        state: Current environment state.
        action: Action to take for the selected player.
        params: Environment parameters.

    Returns:
        Updated environment state.
    """
    player_idx = state.selected_player
    state = _handle_player_action(state, action, player_idx)
    state = update_crafting(state)
    state = update_all_machines(state, params)
    state = update_scent_field(state, params)
    state = update_biters(state, params, rng)
    return state.replace(timestep=state.timestep + 1)


def is_game_over(
    state: EnvState, params: EnvParams,
) -> jax.Array:
    """Check if the episode has ended.

    Args:
        state: Current environment state.
        params: Environment parameters.

    Returns:
        Boolean indicating whether the game is over.
    """
    return jnp.bool_(state.timestep >= params.max_timesteps)
