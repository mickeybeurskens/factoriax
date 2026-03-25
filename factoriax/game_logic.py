"""Game logic for player movement and environment stepping."""

import jax
import jax.numpy as jnp
from jax import lax

from factoriax.achievements import check_achievements
from factoriax.constants import (
    BLOCK_TO_ITEM_ARRAY,
    DIRECTIONS,
    MACHINE_NUM_SLOTS,
    MACHINE_SLOT_ROLES,
    MAX_MACHINE_INVENTORY_SLOTS,
    MAX_MACHINE_STACK_SIZE,
    MAX_STACK_SIZE,
    MINEABLE_BLOCKS,
    SOLID_BLOCKS,
    Action,
    BlockType,
    ItemType,
    MachineType,
    SlotRole,
)
from factoriax.crafting import cycle_recipe, cycle_slot, start_crafting, update_crafting
from factoriax.inventory import find_best_slot
from factoriax.machines import update_all_machines
from factoriax.placement import pickup_machine, place_machine
from factoriax.recipes import (
    ASSEMBLER_RECIPE_INPUT_COUNTS,
    ASSEMBLER_RECIPE_INPUT_ITEMS,
    MAX_ASSEMBLER_STACK_SIZE,
)
from factoriax.state import EnvParams, EnvState


def is_position_in_bounds(
    position: jax.Array, map_width: int, map_height: int
) -> jax.Array:
    """Check if a position is within map boundaries.

    Args:
        position: (x, y) coordinates to check
        map_width: Width of the map
        map_height: Height of the map

    Returns:
        Boolean indicating whether position is in bounds
    """
    x, y = position[0], position[1]
    return (x >= 0) & (x < map_width) & (y >= 0) & (y < map_height)


def get_block_at(state: EnvState, position: jax.Array) -> jax.Array:
    """Get the block type at a position, returning OUT_OF_BOUNDS if outside map.

    Args:
        state: Current environment state
        position: (x, y) coordinates to query

    Returns:
        Block type at the position
    """
    map_height, map_width = state.map.shape
    in_bounds = is_position_in_bounds(position, map_width, map_height)
    block: jax.Array = lax.cond(
        in_bounds,
        lambda: state.map[position[1], position[0]],
        lambda: jnp.int32(BlockType.OUT_OF_BOUNDS),
    )
    return block


def is_position_walkable(state: EnvState, position: jax.Array) -> jax.Array:
    """Check if a position can be walked on.

    A position is walkable if it's in bounds, not a solid block (water, etc),
    and does not have a blocking machine on it. Conveyor belts are not
    blocking because players can walk over them.

    Args:
        state: Current environment state
        position: (x, y) coordinates to check

    Returns:
        Boolean indicating whether position is walkable
    """
    block = get_block_at(state, position)
    is_solid = jnp.any(block == SOLID_BLOCKS)

    map_height, map_width = state.map.shape
    in_bounds = is_position_in_bounds(position, map_width, map_height)
    has_blocking_machine = lax.cond(
        in_bounds,
        lambda: (
            (state.machine_types[position[1], position[0]] != MachineType.NONE)
            & (
                state.machine_types[position[1], position[0]]
                != MachineType.CONVEYOR_BELT
            )
        ),
        lambda: jnp.bool_(False),
    )

    return ~is_solid & ~has_blocking_machine


def move_player(
    state: EnvState, action: int | jax.Array, player_idx: int | jax.Array
) -> EnvState:
    """Attempt to move a player in the specified direction.

    The player will face the direction of movement regardless of whether
    the move succeeds. Movement only succeeds if the target tile is walkable.

    Args:
        state: Current environment state
        action: Action to take (from Action enum)
        player_idx: Index of the player to move

    Returns:
        Updated environment state with new player position and direction
    """
    current_position = state.player_positions[player_idx]
    current_direction = state.player_directions[player_idx]

    direction = DIRECTIONS[action]
    new_position = current_position + direction
    can_move = is_position_walkable(state, new_position)
    final_position = jnp.where(can_move, new_position, current_position)
    new_direction = lax.cond(
        action == Action.NOOP,
        lambda: current_direction,
        lambda: jnp.int32(action),
    )

    new_positions = state.player_positions.at[player_idx].set(final_position)
    new_directions = state.player_directions.at[player_idx].set(new_direction)

    return state.replace(  # type: ignore[attr-defined, no-any-return]
        player_positions=new_positions,
        player_directions=new_directions,
    )


def mine_block(state: EnvState, player_idx: int | jax.Array) -> EnvState:
    """Attempt to mine the block at a player's current position.

    Mining succeeds if:
    - The block is mineable (coal, iron, or copper)
    - The block has remaining resources
    - There is inventory space (existing stack with room or empty slot)

    On success, one resource is extracted and added to inventory. The block
    becomes dirt only when all resources are depleted.

    Args:
        state: Current environment state
        player_idx: Index of the player performing the mining

    Returns:
        Updated environment state with updated resources and inventory
    """
    player_pos = state.player_positions[player_idx]
    px, py = player_pos[0], player_pos[1]
    block_type = state.map[py, px]

    is_mineable = jnp.any(block_type == MINEABLE_BLOCKS)
    has_resources = state.block_resources[py, px] > 0
    item_type = BLOCK_TO_ITEM_ARRAY[block_type]

    player_inv_items = state.inventory_items[player_idx]
    player_inv_counts = state.inventory_counts[player_idx]

    slot_idx, can_add_to_inventory = find_best_slot(
        player_inv_items, player_inv_counts, item_type, MAX_STACK_SIZE
    )
    can_mine = is_mineable & has_resources & can_add_to_inventory

    new_inventory_items = lax.cond(
        can_mine,
        lambda: state.inventory_items.at[player_idx, slot_idx].set(item_type),
        lambda: state.inventory_items,
    )

    new_inventory_counts = lax.cond(
        can_mine,
        lambda: state.inventory_counts.at[player_idx, slot_idx].set(
            player_inv_counts[slot_idx] + 1
        ),
        lambda: state.inventory_counts,
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

    return state.replace(  # type: ignore[attr-defined, no-any-return]
        map=new_map,
        inventory_items=new_inventory_items,
        inventory_counts=new_inventory_counts,
        block_resources=new_block_resources,
        items_mined=new_items_mined,
    )


_SLOT_ROLES_JAX = jnp.array(MACHINE_SLOT_ROLES, dtype=jnp.int32)
_MACHINE_NUM_SLOTS_JAX = jnp.array(MACHINE_NUM_SLOTS, dtype=jnp.int32)

# Slot scan order for withdraw: OUTPUT first, then STORAGE, then INPUT.
# Built as an argsort over a priority array keyed by SlotRole.
_WITHDRAW_PRIORITY = jnp.array(
    [3, 2, 0, 1],  # NONE=3 (last), INPUT=2, OUTPUT=0 (first), STORAGE=1
    dtype=jnp.int32,
)


def _resolve_adjacent_machine(
    state: EnvState, player_idx: int | jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Resolve the machine on the tile in front of the player.

    Shared setup for deposit and withdraw: finds the target tile,
    checks bounds, and reads the machine's slot contents.

    Args:
        state: Current environment state.
        player_idx: Index of the acting player.

    Returns:
        Tuple of ``(target_x, target_y, in_bounds, machine_type,
        slot_items, slot_counts, num_slots)``.
    """
    from factoriax.placement import get_tile_in_front

    target_x, target_y = get_tile_in_front(state, player_idx)
    map_h, map_w = state.map.shape

    in_bounds = is_position_in_bounds(
        jnp.array([target_x, target_y]), map_w, map_h
    )

    machine_type = jnp.where(
        in_bounds, state.machine_types[target_y, target_x], MachineType.NONE
    )

    slot_items = jnp.where(
        in_bounds,
        state.machine_inventory_items[target_y, target_x],
        jnp.zeros(MAX_MACHINE_INVENTORY_SLOTS, dtype=jnp.int32),
    )
    slot_counts = jnp.where(
        in_bounds,
        state.machine_inventory_counts[target_y, target_x],
        jnp.zeros(MAX_MACHINE_INVENTORY_SLOTS, dtype=jnp.int16),
    )
    num_slots = _MACHINE_NUM_SLOTS_JAX[machine_type]

    return (
        target_x, target_y, in_bounds, machine_type,
        slot_items, slot_counts, num_slots,
    )


def _find_deposit_slot(
    machine_type: jax.Array,
    slot_items: jax.Array,
    slot_counts: jax.Array,
    num_slots: jax.Array,
    item: jax.Array,
    recipe: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Find the best machine slot for depositing an item.

    Eligible slots are INPUT and STORAGE roles within range. For
    assembler INPUT slots, the item must match the recipe's expected
    input. Matching slots are preferred over empty ones.

    Args:
        machine_type: Type of the target machine.
        slot_items: Machine slot item types, shape ``(MAX_MACHINE_INVENTORY_SLOTS,)``.
        slot_counts: Machine slot counts, shape ``(MAX_MACHINE_INVENTORY_SLOTS,)``.
        num_slots: Number of active slots for this machine type.
        item: Item type the player wants to deposit.
        recipe: The machine's selected recipe index.

    Returns:
        Tuple of ``(best_slot, has_slot, cap)`` where ``cap`` is the
        per-slot stack limit for the machine type.
    """
    slot_roles = _SLOT_ROLES_JAX[machine_type]
    is_deposit_role = (slot_roles == SlotRole.INPUT) | (slot_roles == SlotRole.STORAGE)
    slot_idx_range = jnp.arange(MAX_MACHINE_INVENTORY_SLOTS)
    in_range = slot_idx_range < num_slots

    cap = jnp.where(
        machine_type == MachineType.ASSEMBLER,
        MAX_ASSEMBLER_STACK_SIZE,
        MAX_MACHINE_STACK_SIZE,
    )

    slot_empty = slot_counts == 0
    slot_matches = slot_items == item
    slot_has_space = (slot_empty | slot_matches) & (slot_counts < cap)

    # Assembler recipe filter for INPUT slots.
    is_asm = machine_type == MachineType.ASSEMBLER
    expected_items = ASSEMBLER_RECIPE_INPUT_ITEMS[recipe]
    expected_counts = ASSEMBLER_RECIPE_INPUT_COUNTS[recipe]
    pad = jnp.zeros(MAX_MACHINE_INVENTORY_SLOTS - 2, dtype=jnp.int32)
    expected_items_full = jnp.concatenate([expected_items, pad])
    expected_counts_full = jnp.concatenate([expected_counts, pad])
    is_input = slot_roles == SlotRole.INPUT
    recipe_ok = (item == expected_items_full) & (expected_counts_full > 0)
    asm_filter = jnp.where(is_asm & is_input, recipe_ok, True)

    slot_usable = is_deposit_role & in_range & slot_has_space & asm_filter

    # Prefer matching slots over empty ones. Non-usable slots get -1.
    priority = jnp.where(
        slot_usable & slot_matches,
        2,
        jnp.where(slot_usable & slot_empty, 1, -1),
    )
    best_slot = jnp.argmax(priority)
    has_slot = jnp.any(slot_usable)

    return best_slot, has_slot, cap


def _find_withdraw_slot(
    machine_type: jax.Array,
    slot_items: jax.Array,
    slot_counts: jax.Array,
    num_slots: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Find the best machine slot to withdraw from.

    Scans slots by priority: OUTPUT first, then STORAGE, then INPUT.
    Within the same role, the lowest slot index wins.

    Args:
        machine_type: Type of the target machine.
        slot_items: Machine slot item types, shape ``(MAX_MACHINE_INVENTORY_SLOTS,)``.
        slot_counts: Machine slot counts, shape ``(MAX_MACHINE_INVENTORY_SLOTS,)``.
        num_slots: Number of active slots for this machine type.

    Returns:
        Tuple of ``(best_slot, has_source, source_item, source_count)``.
    """
    slot_roles = _SLOT_ROLES_JAX[machine_type]
    slot_idx_range = jnp.arange(MAX_MACHINE_INVENTORY_SLOTS)
    in_range = slot_idx_range < num_slots
    has_items = slot_counts > 0
    not_none_role = slot_roles != SlotRole.NONE

    role_priority = _WITHDRAW_PRIORITY[slot_roles]
    eligible = in_range & has_items & not_none_role
    effective_priority = jnp.where(eligible, role_priority, jnp.int32(99))
    score = effective_priority * MAX_MACHINE_INVENTORY_SLOTS + slot_idx_range
    best_slot = jnp.argmin(score)
    has_source = eligible[best_slot]

    source_item = slot_items[best_slot]
    source_count = slot_counts[best_slot].astype(jnp.int32)

    return best_slot, has_source, source_item, source_count


def deposit_to_adjacent(state: EnvState, player_idx: int | jax.Array) -> EnvState:
    """Deposit the player's selected inventory stack into the machine in front.

    Transfers the full stack from the player's currently selected slot into
    the first compatible machine slot. Slot compatibility follows the same
    rules as arm deposits: only INPUT and STORAGE slots are eligible, and
    assembler INPUT slots are filtered by the machine's selected recipe.

    A matching slot (same item type with space) is preferred over an empty
    slot. No-op when:
    - No machine is in front of the player
    - The selected inventory slot is empty
    - No compatible machine slot has space

    Args:
        state: Current environment state.
        player_idx: Index of the acting player.

    Returns:
        Updated state with items transferred, or unchanged if invalid.
    """
    target_x, target_y, in_bounds, machine_type, slot_items, slot_counts, num_slots = (
        _resolve_adjacent_machine(state, player_idx)
    )
    has_machine = machine_type != MachineType.NONE

    selected_slot = state.selected_slots[player_idx]
    item = state.inventory_items[player_idx, selected_slot]
    count = state.inventory_counts[player_idx, selected_slot]
    has_item = (item != ItemType.EMPTY) & (count > 0)

    recipe = jnp.where(
        in_bounds,
        state.machine_selected_recipe[target_y, target_x],
        jnp.int32(0),
    )
    best_slot, has_slot, cap = _find_deposit_slot(
        machine_type, slot_items, slot_counts, num_slots, item, recipe
    )

    can_deposit = in_bounds & has_machine & has_item & has_slot

    space = cap - slot_counts[best_slot]
    transfer = jnp.minimum(count.astype(jnp.int32), space)

    def do_deposit(s: EnvState) -> EnvState:
        new_m_items = s.machine_inventory_items.at[target_y, target_x, best_slot].set(
            item
        )
        new_m_counts = s.machine_inventory_counts.at[target_y, target_x, best_slot].set(
            slot_counts[best_slot] + transfer.astype(jnp.int16)
        )

        remaining = count - transfer.astype(jnp.int32)
        new_inv_items = jnp.where(
            remaining > 0,
            s.inventory_items,
            s.inventory_items.at[player_idx, selected_slot].set(ItemType.EMPTY),
        )
        new_inv_counts = s.inventory_counts.at[player_idx, selected_slot].set(remaining)
        return s.replace(
            machine_inventory_items=new_m_items,
            machine_inventory_counts=new_m_counts,
            inventory_items=new_inv_items,
            inventory_counts=new_inv_counts,
        )

    return lax.cond(can_deposit, do_deposit, lambda s: s, state)


def withdraw_from_adjacent(state: EnvState, player_idx: int | jax.Array) -> EnvState:
    """Withdraw items from the machine in front into the player's inventory.

    Scans machine slots in priority order (OUTPUT, STORAGE, INPUT) and
    takes the full stack from the first non-empty slot. The stack is merged
    into existing matching player stacks first, then placed in empty slots.

    No-op when:
    - No machine is in front of the player
    - All machine slots are empty
    - The player inventory is completely full

    Args:
        state: Current environment state.
        player_idx: Index of the acting player.

    Returns:
        Updated state with items transferred, or unchanged if invalid.
    """
    from factoriax.crafting import add_item_to_inventory

    target_x, target_y, in_bounds, machine_type, slot_items, slot_counts, num_slots = (
        _resolve_adjacent_machine(state, player_idx)
    )
    has_machine = machine_type != MachineType.NONE

    best_slot, has_source, source_item, source_count = _find_withdraw_slot(
        machine_type, slot_items, slot_counts, num_slots
    )

    # Check player has space for at least one item.
    p_items = state.inventory_items[player_idx]
    p_counts = state.inventory_counts[player_idx]
    can_stack = jnp.any((p_items == source_item) & (p_counts < MAX_STACK_SIZE))
    has_empty = jnp.any(p_items == ItemType.EMPTY)
    has_space = can_stack | has_empty

    can_withdraw = in_bounds & has_machine & has_source & has_space

    def do_withdraw(s: EnvState) -> EnvState:
        # Clear the machine slot first.
        new_m_items = s.machine_inventory_items.at[target_y, target_x, best_slot].set(0)
        new_m_counts = s.machine_inventory_counts.at[target_y, target_x, best_slot].set(
            jnp.int16(0)
        )
        s = s.replace(
            machine_inventory_items=new_m_items,
            machine_inventory_counts=new_m_counts,
        )
        return add_item_to_inventory(s, player_idx, source_item, source_count)

    return lax.cond(can_withdraw, do_withdraw, lambda s: s, state)


def _handle_player_action(
    state: EnvState, action: int | jax.Array, player_idx: int | jax.Array
) -> EnvState:
    """Handle a single player action.

    Dispatches to the appropriate handler based on action type.

    Args:
        state: Current environment state
        action: Action to take
        player_idx: Index of the player

    Returns:
        Updated environment state
    """
    is_mine = action == Action.MINE
    is_craft = action == Action.CRAFT
    is_place = action == Action.PLACE
    is_next_slot = action == Action.NEXT_SLOT
    is_prev_slot = action == Action.PREV_SLOT
    is_next_recipe = action == Action.NEXT_RECIPE
    is_prev_recipe = action == Action.PREV_RECIPE
    is_pickup = action == Action.PICKUP
    is_deposit = action == Action.DEPOSIT
    is_withdraw = action == Action.WITHDRAW

    state = lax.cond(is_mine, lambda s: mine_block(s, player_idx), lambda s: s, state)
    state = lax.cond(
        is_craft, lambda s: start_crafting(s, player_idx), lambda s: s, state
    )
    state = lax.cond(
        is_place, lambda s: place_machine(s, player_idx), lambda s: s, state
    )
    state = lax.cond(
        is_pickup, lambda s: pickup_machine(s, player_idx), lambda s: s, state
    )
    state = lax.cond(
        is_deposit,
        lambda s: deposit_to_adjacent(s, player_idx),
        lambda s: s,
        state,
    )
    state = lax.cond(
        is_withdraw,
        lambda s: withdraw_from_adjacent(s, player_idx),
        lambda s: s,
        state,
    )
    state = lax.cond(
        is_next_slot, lambda s: cycle_slot(s, player_idx, 1), lambda s: s, state
    )
    state = lax.cond(
        is_prev_slot, lambda s: cycle_slot(s, player_idx, -1), lambda s: s, state
    )
    state = lax.cond(
        is_next_recipe, lambda s: cycle_recipe(s, player_idx, 1), lambda s: s, state
    )
    state = lax.cond(
        is_prev_recipe, lambda s: cycle_recipe(s, player_idx, -1), lambda s: s, state
    )

    is_movement = ~(
        is_mine
        | is_craft
        | is_place
        | is_pickup
        | is_deposit
        | is_withdraw
        | is_next_slot
        | is_prev_slot
        | is_next_recipe
        | is_prev_recipe
    )
    state = lax.cond(
        is_movement, lambda s: move_player(s, action, player_idx), lambda s: s, state
    )

    return state


def factoriax_step(
    rng: jax.Array, state: EnvState, action: int | jax.Array, params: EnvParams
) -> EnvState:
    """Execute one step of the environment.

    Processes in order:

    1. Selected player action (move, mine, craft, place, or UI actions)
    2. Crafting progress for all players
    3. All machine updates
    4. Achievement tracking
    5. Timestep increment

    Non-selected players perform NOOP.  Reward computation is intentionally
    absent here; it is performed externally by a reward function from
    :mod:`factoriax.rewards`, which receives both the pre-step and
    post-step states.

    Args:
        rng: JAX random key (reserved for future stochastic mechanics).
        state: Current environment state.
        action: Action to take for the selected player.
        params: Environment parameters.

    Returns:
        Updated environment state.
    """
    player_idx = state.selected_player
    state = _handle_player_action(state, action, player_idx)
    state = update_crafting(state)
    state = update_all_machines(state)
    state = check_achievements(state)
    return state.replace(timestep=state.timestep + 1)  # type: ignore[attr-defined, no-any-return]


def is_game_over(state: EnvState, params: EnvParams) -> jax.Array:
    """Check if the episode has ended.

    Currently only checks if max timesteps has been reached.

    Args:
        state: Current environment state
        params: Environment parameters

    Returns:
        Boolean indicating whether the game is over
    """
    result: jax.Array = jnp.bool_(state.timestep >= params.max_timesteps)
    return result
