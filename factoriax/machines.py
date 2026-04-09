"""Machine update logic for the FactoriaX environment.

Uses the pouch inventory model: each tile stores a ``(NUM_ITEM_TYPES,)``
count array instead of slot-based ``(items, counts)`` pairs. Machine
type constraints (max distinct types, max stack per type) are enforced
during transfers.
"""

import jax.numpy as jnp

from factoriax.constants import (
    BLOCK_TO_ITEM_ARRAY,
    MACHINE_MAX_STACK,
    MACHINE_MAX_TYPES,
    MACHINE_POWER_CONSUMPTION,
    MAX_MACHINE_STACK_SIZE,
    MAX_UNDERGROUND_RANGE,
    NUM_ITEM_TYPES,
    NUM_TECHNOLOGIES,
    TECH_GATES_RECIPE,
    BlockType,
    Direction,
    ItemType,
    MachineType,
)
from factoriax.recipes import (
    ASSEMBLER_RECIPE_INPUT_COUNTS,
    ASSEMBLER_RECIPE_INPUT_ITEMS,
    ASSEMBLER_RECIPE_OUTPUTS,
    ASSEMBLER_RECIPE_TICKS,
    NUM_ASSEMBLER_RECIPES,
)
from factoriax.state import EnvParams, EnvState


def _grid_indices(h: int, w: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return broadcast-ready (rows, cols) index grids."""
    rows = jnp.broadcast_to(jnp.arange(h)[:, None], (h, w))
    cols = jnp.broadcast_to(jnp.arange(w)[None, :], (h, w))
    return rows, cols


def _direction_offsets(
    direction: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convert Direction arrays to (dx, dy) offsets."""
    dx = jnp.where(direction == Direction.LEFT, -1, 0) + jnp.where(
        direction == Direction.RIGHT, 1, 0
    )
    dy = jnp.where(direction == Direction.UP, -1, 0) + jnp.where(
        direction == Direction.DOWN, 1, 0
    )
    return dx, dy


def _buffer_item_type(inv: jnp.ndarray) -> jnp.ndarray:
    """Find the single non-zero item type in a pouch buffer.

    For belts and arms that hold at most one item type at a time.

    Args:
        inv: Machine inventory, shape ``(..., NUM_ITEM_TYPES)``.

    Returns:
        Item type index, shape ``(...)``. Zero (EMPTY) if buffer empty.
    """
    has_items = inv[..., 1:] > 0
    idx = jnp.argmax(has_items, axis=-1) + 1
    any_items = jnp.any(has_items, axis=-1)
    return jnp.where(any_items, idx, 0)


def _buffer_count(
    inv: jnp.ndarray,
    item_type: jnp.ndarray,
) -> jnp.ndarray:
    """Read the count for a specific item type per tile.

    Args:
        inv: Machine inventory, shape ``(H, W, NUM_ITEM_TYPES)``.
        item_type: Item type index per tile, shape ``(H, W)``.

    Returns:
        Count per tile, shape ``(H, W)``.
    """
    h, w = inv.shape[:2]
    rows, cols = _grid_indices(h, w)
    return inv[rows, cols, item_type]


def _distinct_types(inv: jnp.ndarray) -> jnp.ndarray:
    """Count distinct non-zero item types per tile.

    Args:
        inv: Machine inventory, shape ``(H, W, NUM_ITEM_TYPES)``.

    Returns:
        Count per tile, shape ``(H, W)``.
    """
    return jnp.sum(inv[..., 1:] > 0, axis=-1)


def update_all_machines(
    state: EnvState,
    params: EnvParams | None = None,
) -> EnvState:
    """Update all machines in parallel for one step.

    Args:
        state: Current environment state.
        params: Environment parameters.

    Returns:
        Updated state with all machines processed.
    """
    if params is None:
        params = EnvParams()
    max_asm_stack = params.max_assembler_stack_size
    state = refuel_machines(state, params)
    state = run_assemblers(state, params)
    state = run_miners(state, params)
    state = push_miner_output(state, max_asm_stack)
    state = run_conveyor_belts(state)
    state = run_arms(state, max_asm_stack)
    return state


def refuel_machines(
    state: EnvState,
    params: EnvParams | None = None,
) -> EnvState:
    """Convert coal to power for miners that need it.

    Args:
        state: Current environment state.
        params: Environment parameters.

    Returns:
        Updated state with refueled machines.
    """
    if params is None:
        params = EnvParams()
    is_miner = state.machine_types == MachineType.MINER
    needs_power = state.machine_power <= 0
    fuel_count = state.machine_inventory[..., ItemType.COAL]
    has_fuel = fuel_count > 0

    should_refuel = is_miner & needs_power & has_fuel

    new_fuel = fuel_count - should_refuel.astype(jnp.int16)
    refuel_power = should_refuel.astype(jnp.int32) * params.power_per_coal
    new_power = state.machine_power + refuel_power

    new_inv = state.machine_inventory.at[..., ItemType.COAL].set(
        new_fuel,
    )
    return state.replace(
        machine_inventory=new_inv,
        machine_power=new_power,
    )


def run_assemblers(
    state: EnvState,
    params: EnvParams | None = None,
) -> EnvState:
    """Execute assembler crafting logic for all assemblers.

    Uses ``machine_power`` as a craft progress countdown.

    Args:
        state: Current environment state.
        params: Environment parameters.

    Returns:
        Updated state with assembler operations applied.
    """
    if params is None:
        params = EnvParams()
    max_asm_stack = params.max_assembler_stack_size

    h, w = state.machine_types.shape
    rows, cols = _grid_indices(h, w)

    is_asm = (state.machine_types == MachineType.ASSEMBLER) & (state.machine_health > 0)

    recipe = state.machine_selected_recipe
    recipe_out = ASSEMBLER_RECIPE_OUTPUTS[recipe]
    recipe_ticks = ASSEMBLER_RECIPE_TICKS[recipe]
    recipe_in_items = ASSEMBLER_RECIPE_INPUT_ITEMS[recipe]  # (H,W,2)
    recipe_in_counts = ASSEMBLER_RECIPE_INPUT_COUNTS[recipe]

    power = state.machine_power
    inv = state.machine_inventory.astype(jnp.int32)

    # Phase 1: complete crafts (power == 1).
    out_count = inv[rows, cols, recipe_out]
    completing = is_asm & (power == 1)
    out_has_space = out_count < max_asm_stack
    can_complete = completing & out_has_space

    inv = inv.at[rows, cols, recipe_out].set(
        jnp.where(can_complete, out_count + 1, out_count),
    )
    power = jnp.where(can_complete, 0, power)

    # Phase 2: progress (power > 1).
    progressing = is_asm & (power > 1)
    power = jnp.where(progressing, power - 1, power)

    # Research gating.
    _all_recipes = jnp.arange(NUM_ASSEMBLER_RECIPES)
    recipe_gated = jnp.any(
        TECH_GATES_RECIPE[..., None] == _all_recipes[None, ...],
        axis=0,
    )
    _tech_for_recipe = jnp.zeros(
        NUM_ASSEMBLER_RECIPES,
        dtype=jnp.int32,
    )
    for t in range(NUM_TECHNOLOGIES):
        _tech_for_recipe = _tech_for_recipe.at[TECH_GATES_RECIPE[t]].set(t)
    recipe_tech_unlocked = state.research_unlocked[_tech_for_recipe]
    recipe_available = ~recipe_gated | recipe_tech_unlocked
    tile_recipe_allowed = recipe_available[recipe]

    # Phase 3: start new crafts.
    idle = is_asm & (power == 0) & tile_recipe_allowed

    in0_type = recipe_in_items[..., 0]
    in1_type = recipe_in_items[..., 1]
    need0 = recipe_in_counts[..., 0]
    need1 = recipe_in_counts[..., 1]

    in0_count = inv[rows, cols, in0_type]
    in1_count = inv[rows, cols, in1_type]
    has_in0 = in0_count >= need0
    has_in1 = (need1 == 0) | (in1_count >= need1)

    out_count2 = inv[rows, cols, recipe_out]
    out_ok = out_count2 < max_asm_stack

    can_start = idle & has_in0 & has_in1 & out_ok

    # Consume inputs.
    inv = inv.at[rows, cols, in0_type].set(
        jnp.where(can_start, in0_count - need0, inv[rows, cols, in0_type]),
    )
    inv = inv.at[rows, cols, in1_type].set(
        jnp.where(
            can_start & (need1 > 0),
            in1_count - need1,
            inv[rows, cols, in1_type],
        ),
    )
    power = jnp.where(can_start, recipe_ticks, power)

    return state.replace(
        machine_power=power,
        machine_inventory=inv.astype(jnp.int16),
    )


def run_miners(
    state: EnvState,
    params: EnvParams | None = None,
) -> EnvState:
    """Execute miner behavior for all miners in parallel.

    Miners with power extract resources from the block below and
    deposit into their pouch inventory under the ore's item type.

    Args:
        state: Current environment state.
        params: Environment parameters.

    Returns:
        Updated state with miner operations applied.
    """
    if params is None:
        params = EnvParams()

    h, w = state.machine_types.shape
    rows, cols = _grid_indices(h, w)

    is_miner = (state.machine_types == MachineType.MINER) & (state.machine_health > 0)
    has_power = state.machine_power > 0
    has_resources = state.block_resources > 0

    block_item = BLOCK_TO_ITEM_ARRAY[state.map]

    inv = state.machine_inventory.astype(jnp.int32)
    output_count = inv[rows, cols, block_item]
    has_space = output_count < MAX_MACHINE_STACK_SIZE

    can_mine = is_miner & has_power & has_resources & has_space

    mining_rate = jnp.where(
        state.machine_types == MachineType.MINER,
        params.miner_mining_rate,
        0,
    )
    available_space = MAX_MACHINE_STACK_SIZE - output_count
    mine_amount = jnp.minimum(mining_rate, state.block_resources)
    mine_amount = jnp.minimum(mine_amount, available_space)
    mine_amount = mine_amount * can_mine

    new_resources = state.block_resources - mine_amount.astype(jnp.int16)

    new_inv = inv.at[rows, cols, block_item].set(
        output_count + mine_amount,
    )

    actually_mined = mine_amount > 0
    power_cost = MACHINE_POWER_CONSUMPTION[state.machine_types] * actually_mined
    new_power = state.machine_power - power_cost

    is_depleted = (new_resources <= 0) & (state.block_resources > 0)
    new_map = jnp.where(is_depleted, BlockType.DIRT, state.map)

    mined_per_type = jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.int32)
    mined_per_type = mined_per_type.at[block_item.ravel()].add(
        mine_amount.ravel(),
    )
    new_items_mined = state.items_mined + mined_per_type

    return state.replace(
        block_resources=new_resources,
        machine_inventory=new_inv.astype(jnp.int16),
        machine_power=new_power,
        map=new_map,
        items_mined=new_items_mined,
    )


def push_miner_output(
    state: EnvState,
    max_asm_stack: int = 1000,
) -> EnvState:
    """Push miner output into the machine the miner is facing.

    Transfers the ore from the miner's pouch into the forward
    neighbour's pouch, respecting type and capacity constraints.

    Args:
        state: Current environment state.
        max_asm_stack: Maximum assembler stack size.

    Returns:
        Updated state with miner outputs pushed forward.
    """
    h, w = state.machine_types.shape
    rows, cols = _grid_indices(h, w)
    is_miner = state.machine_types == MachineType.MINER

    dx, dy = _direction_offsets(state.machine_direction)
    fwd_row = jnp.clip(rows + dy, 0, h - 1)
    fwd_col = jnp.clip(cols + dx, 0, w - 1)

    # Find what the miner has to push (the ore type).
    block_item = BLOCK_TO_ITEM_ARRAY[state.map]
    inv = state.machine_inventory.astype(jnp.int32)
    miner_ore_count = inv[rows, cols, block_item]

    fwd_mtype = state.machine_types[fwd_row, fwd_col]
    fwd_not_none = fwd_mtype != MachineType.NONE
    fwd_not_self = (fwd_row != rows) | (fwd_col != cols)

    fwd_inv = state.machine_inventory[fwd_row, fwd_col].astype(
        jnp.int32,
    )
    fwd_item_count = fwd_inv[rows, cols, block_item]
    fwd_max_stack = MACHINE_MAX_STACK[fwd_mtype]
    fwd_cap = jnp.where(
        fwd_mtype == MachineType.ASSEMBLER,
        max_asm_stack,
        fwd_max_stack,
    )
    fwd_has_space = fwd_item_count < fwd_cap

    # Type constraint: check the target can accept a new type.
    fwd_existing = fwd_inv[rows, cols, block_item] > 0
    fwd_distinct = _distinct_types(
        state.machine_inventory[fwd_row, fwd_col],
    )
    fwd_max_types = MACHINE_MAX_TYPES[fwd_mtype]
    fwd_type_ok = fwd_existing | (fwd_distinct < fwd_max_types)

    # Assembler recipe filter.
    is_asm_fwd = fwd_mtype == MachineType.ASSEMBLER
    fwd_recipe = state.machine_selected_recipe[fwd_row, fwd_col]
    exp_in0 = ASSEMBLER_RECIPE_INPUT_ITEMS[fwd_recipe, 0]
    exp_in1 = ASSEMBLER_RECIPE_INPUT_ITEMS[fwd_recipe, 1]
    need0 = ASSEMBLER_RECIPE_INPUT_COUNTS[fwd_recipe, 0]
    need1 = ASSEMBLER_RECIPE_INPUT_COUNTS[fwd_recipe, 1]
    item_matches_recipe = ((block_item == exp_in0) & (need0 > 0)) | (
        (block_item == exp_in1) & (need1 > 0)
    )
    asm_filter = jnp.where(is_asm_fwd, item_matches_recipe, True)

    can_push = (
        is_miner
        & (miner_ore_count > 0)
        & fwd_not_none
        & fwd_not_self
        & fwd_has_space
        & fwd_type_ok
        & asm_filter
    )

    transfer = jnp.where(
        can_push,
        jnp.minimum(miner_ore_count, fwd_cap - fwd_item_count),
        0,
    )

    # Update: add to forward, subtract from miner.
    new_inv = inv.at[rows, cols, block_item].set(
        jnp.where(can_push, miner_ore_count - transfer, miner_ore_count),
    )
    new_inv = new_inv.at[fwd_row, fwd_col, block_item].add(
        transfer.astype(jnp.int32),
    )

    return state.replace(
        machine_inventory=new_inv.astype(jnp.int16),
    )


def run_conveyor_belts(state: EnvState) -> EnvState:
    """Move items along conveyor belts and underground tunnels.

    Regular belts and underground exits push to the adjacent tile in
    the facing direction. Underground entries push to their paired
    exit (the nearest ``UNDERGROUND_EXIT`` facing the same direction
    within ``MAX_UNDERGROUND_RANGE`` tiles).

    Surface targets (regular belts and underground entries) accept
    items from adjacent tiles. Underground exits only accept from
    paired entries, not from surface neighbors.

    Args:
        state: Current environment state.

    Returns:
        Updated state with belt items advanced one tile.
    """
    h, w = state.machine_types.shape
    rows, cols = _grid_indices(h, w)
    mt = state.machine_types
    alive = state.machine_health > 0

    is_belt = (mt == MachineType.CONVEYOR_BELT) & alive
    is_entry = (mt == MachineType.UNDERGROUND_ENTRY) & alive
    is_exit = (mt == MachineType.UNDERGROUND_EXIT) & alive
    is_belt_like = is_belt | is_entry | is_exit

    inv = state.machine_inventory  # (H, W, NUM_ITEM_TYPES) int16
    src_item = _buffer_item_type(inv)
    src_count = _buffer_count(inv, src_item)

    dx, dy = _direction_offsets(state.machine_direction)

    # --- Compute targets ---
    # Adjacent tile (for regular belts and exits).
    adj_row = jnp.clip(rows + dy, 0, h - 1)
    adj_col = jnp.clip(cols + dx, 0, w - 1)

    # Paired exit (for entries): scan forward up to MAX_UNDERGROUND_RANGE.
    paired_row = rows
    paired_col = cols
    found_pair = jnp.zeros((h, w), dtype=jnp.bool_)
    dirs = state.machine_direction

    for offset in range(1, MAX_UNDERGROUND_RANGE + 1):
        cr = jnp.clip(rows + dy * offset, 0, h - 1)
        cc = jnp.clip(cols + dx * offset, 0, w - 1)
        is_match = (mt[cr, cc] == MachineType.UNDERGROUND_EXIT) & (
            dirs[cr, cc] == dirs[rows, cols]
        )
        first = is_match & ~found_pair
        paired_row = jnp.where(first, cr, paired_row)
        paired_col = jnp.where(first, cc, paired_col)
        found_pair = found_pair | first

    # Entries push to paired exit; belts and exits push to adjacent.
    use_paired = is_entry & found_pair
    tgt_row = jnp.where(use_paired, paired_row, adj_row)
    tgt_col = jnp.where(use_paired, paired_col, adj_col)

    # --- Target validation ---
    tgt_mt = mt[tgt_row, tgt_col]
    # Surface targets: regular belts and entries accept from surface.
    surface_ok = (tgt_mt == MachineType.CONVEYOR_BELT) | (
        tgt_mt == MachineType.UNDERGROUND_ENTRY
    )
    # Underground targets: exits accept only from paired entries.
    underground_ok = use_paired & (tgt_mt == MachineType.UNDERGROUND_EXIT)
    tgt_accepts = surface_ok | underground_ok

    tgt_item = _buffer_item_type(inv[tgt_row, tgt_col])
    tgt_count = _buffer_count(inv[tgt_row, tgt_col], tgt_item)
    tgt_compatible = (tgt_count == 0) | (tgt_item == src_item)
    tgt_has_space = tgt_count < MAX_MACHINE_STACK_SIZE

    not_self = (tgt_row != rows) | (tgt_col != cols)
    can_push = (
        is_belt_like
        & (src_count > 0)
        & tgt_accepts
        & tgt_compatible
        & tgt_has_space
        & not_self
    )

    push_count = jnp.where(can_push, src_count, jnp.int16(0))
    push_item = jnp.where(can_push, src_item, 0)

    # Clear source items.
    cleared_inv = inv.astype(jnp.int32)
    cleared_inv = cleared_inv.at[rows, cols, src_item].set(
        jnp.where(can_push, 0, cleared_inv[rows, cols, src_item]),
    )

    # Scatter add to targets.
    flat_idx = (tgt_row * w + tgt_col).ravel()
    flat_item = push_item.ravel()
    flat_count = push_count.ravel().astype(jnp.int32)

    flat_inv = cleared_inv.reshape(h * w, NUM_ITEM_TYPES)
    flat_inv = flat_inv.at[flat_idx, flat_item].add(flat_count)

    new_inv = flat_inv.reshape(h, w, NUM_ITEM_TYPES)
    new_inv = jnp.minimum(new_inv, MAX_MACHINE_STACK_SIZE)

    # Only modify belt-like tiles.
    final_inv = jnp.where(
        is_belt_like[..., None],
        new_inv,
        state.machine_inventory.astype(jnp.int32),
    )

    return state.replace(
        machine_inventory=final_inv.astype(jnp.int16),
    )


def run_arms(
    state: EnvState,
    max_asm_stack: int = 1000,
) -> EnvState:
    """Execute pick-and-place arm behavior for all arms.

    Phase 1 (Deposit): if the arm buffer holds items, transfer them
    into the forward neighbour.
    Phase 2 (Pick): if the arm buffer is empty, take items from the
    backward neighbour.

    Args:
        state: Current environment state.
        max_asm_stack: Maximum assembler stack size.

    Returns:
        Updated state with arm operations applied.
    """
    h, w = state.machine_types.shape
    rows, cols = _grid_indices(h, w)
    is_arm = (state.machine_types == MachineType.ARM) & (state.machine_health > 0)

    dx, dy = _direction_offsets(state.machine_direction)
    fwd_row = jnp.clip(rows + dy, 0, h - 1)
    fwd_col = jnp.clip(cols + dx, 0, w - 1)
    bwd_row = jnp.clip(rows - dy, 0, h - 1)
    bwd_col = jnp.clip(cols - dx, 0, w - 1)

    state = _arm_deposit_phase(
        state,
        is_arm,
        rows,
        cols,
        fwd_row,
        fwd_col,
        max_asm_stack,
    )
    state = _arm_pick_phase(
        state,
        is_arm,
        rows,
        cols,
        bwd_row,
        bwd_col,
    )
    return state


def _arm_deposit_phase(
    state: EnvState,
    is_arm: jnp.ndarray,
    rows: jnp.ndarray,
    cols: jnp.ndarray,
    fwd_row: jnp.ndarray,
    fwd_col: jnp.ndarray,
    max_asm_stack: int = 1000,
) -> EnvState:
    """Deposit arm buffer into the forward neighbour."""
    inv = state.machine_inventory  # (H,W,NIT) int16
    arm_item = _buffer_item_type(inv)  # (H, W)
    arm_count = _buffer_count(inv, arm_item)  # (H, W) int16

    fwd_mtype = state.machine_types[fwd_row, fwd_col]
    fwd_not_none = fwd_mtype != MachineType.NONE
    fwd_not_self = (fwd_row != rows) | (fwd_col != cols)

    fwd_inv = inv[fwd_row, fwd_col].astype(jnp.int32)
    fwd_item_count = fwd_inv[rows, cols, arm_item]
    fwd_max_stack = MACHINE_MAX_STACK[fwd_mtype]
    fwd_cap = jnp.where(
        fwd_mtype == MachineType.ASSEMBLER,
        max_asm_stack,
        fwd_max_stack,
    )
    fwd_has_space = fwd_item_count < fwd_cap

    # Type constraint.
    fwd_existing = fwd_inv[rows, cols, arm_item] > 0
    fwd_distinct = _distinct_types(inv[fwd_row, fwd_col])
    fwd_max_types = MACHINE_MAX_TYPES[fwd_mtype]
    fwd_type_ok = fwd_existing | (fwd_distinct < fwd_max_types)

    # Assembler recipe filter.
    is_asm_fwd = fwd_mtype == MachineType.ASSEMBLER
    fwd_recipe = state.machine_selected_recipe[fwd_row, fwd_col]
    exp_in0 = ASSEMBLER_RECIPE_INPUT_ITEMS[fwd_recipe, 0]
    exp_in1 = ASSEMBLER_RECIPE_INPUT_ITEMS[fwd_recipe, 1]
    need0 = ASSEMBLER_RECIPE_INPUT_COUNTS[fwd_recipe, 0]
    need1 = ASSEMBLER_RECIPE_INPUT_COUNTS[fwd_recipe, 1]
    item_matches = ((arm_item == exp_in0) & (need0 > 0)) | (
        (arm_item == exp_in1) & (need1 > 0)
    )
    asm_filter = jnp.where(is_asm_fwd, item_matches, True)

    can_deposit = (
        is_arm
        & (arm_count > 0)
        & fwd_not_none
        & fwd_not_self
        & fwd_has_space
        & fwd_type_ok
        & asm_filter
    )

    transfer = jnp.where(
        can_deposit,
        arm_count.astype(jnp.int32),
        0,
    )

    int_inv = inv.astype(jnp.int32)
    # Add to forward neighbour.
    int_inv = int_inv.at[fwd_row, fwd_col, arm_item].add(transfer)
    # Clear arm buffer.
    int_inv = int_inv.at[rows, cols, arm_item].set(
        jnp.where(can_deposit, 0, int_inv[rows, cols, arm_item]),
    )

    return state.replace(
        machine_inventory=int_inv.astype(jnp.int16),
    )


def _arm_pick_phase(
    state: EnvState,
    is_arm: jnp.ndarray,
    rows: jnp.ndarray,
    cols: jnp.ndarray,
    bwd_row: jnp.ndarray,
    bwd_col: jnp.ndarray,
) -> EnvState:
    """Fill an empty arm buffer by picking from the backward neighbour."""
    inv = state.machine_inventory  # (H,W,NIT) int16
    arm_item = _buffer_item_type(inv)
    arm_empty = arm_item == 0

    bwd_mtype = state.machine_types[bwd_row, bwd_col]
    bwd_not_none = bwd_mtype != MachineType.NONE
    bwd_not_self = (bwd_row != rows) | (bwd_col != cols)

    bwd_inv = inv[bwd_row, bwd_col]  # (H, W, NIT) int16

    # For picking: from assemblers take only the recipe output,
    # from everything else take the first non-zero type.
    is_asm_bwd = bwd_mtype == MachineType.ASSEMBLER
    bwd_recipe = state.machine_selected_recipe[bwd_row, bwd_col]
    asm_output = ASSEMBLER_RECIPE_OUTPUTS[bwd_recipe]

    # Generic pick: first non-zero type.
    generic_item = _buffer_item_type(bwd_inv)
    generic_count = _buffer_count(bwd_inv, generic_item)

    # Assembler pick: recipe output only.
    asm_count = bwd_inv[rows, cols, asm_output].astype(jnp.int32)

    pick_item = jnp.where(is_asm_bwd, asm_output, generic_item)
    pick_count = jnp.where(is_asm_bwd, asm_count, generic_count)

    can_pick = is_arm & arm_empty & bwd_not_none & bwd_not_self & (pick_count > 0)

    transfer = jnp.where(can_pick, pick_count, 0).astype(jnp.int32)

    int_inv = inv.astype(jnp.int32)
    # Subtract from backward neighbour.
    int_inv = int_inv.at[bwd_row, bwd_col, pick_item].add(-transfer)
    # Fill arm buffer.
    int_inv = int_inv.at[rows, cols, pick_item].add(transfer)

    return state.replace(
        machine_inventory=int_inv.astype(jnp.int16),
    )
