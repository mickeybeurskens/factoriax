"""Machine update logic for the FactoriaX environment."""

import jax
import jax.numpy as jnp

from factoriax.constants import (
    BLOCK_TO_ITEM_ARRAY,
    MACHINE_MINING_RATE,
    MACHINE_POWER_CONSUMPTION,
    MACHINE_SLOT_ROLES,
    MAX_MACHINE_STACK_SIZE,
    POWER_PER_COAL,
    Action,
    BlockType,
    MachineType,
    SlotRole,
)
from factoriax.state import EnvState

# Miner slot indices (within machine_inventory_items / machine_inventory_counts).
_MINER_FUEL_SLOT: int = 0  # INPUT — coal fuel consumed by the miner.
_MINER_OUTPUT_SLOT: int = 1  # OUTPUT — mined ore deposited here.

# Belt and arm each use slot 0 as their single working buffer.
_BELT_SLOT: int = 0
_ARM_SLOT: int = 0

# JAX version of MACHINE_SLOT_ROLES for vectorised role lookups inside jit.
_SLOT_ROLES_JAX: jnp.ndarray = jnp.array(MACHINE_SLOT_ROLES, dtype=jnp.int32)

# Deposit-eligible roles (INPUT, STORAGE).
_DEPOSIT_ROLE_INPUT: int = int(SlotRole.INPUT)
_DEPOSIT_ROLE_STORAGE: int = int(SlotRole.STORAGE)
# Pick-eligible roles (OUTPUT, STORAGE).
_PICK_ROLE_OUTPUT: int = int(SlotRole.OUTPUT)
_PICK_ROLE_STORAGE: int = int(SlotRole.STORAGE)


def update_all_machines(state: EnvState) -> EnvState:
    """Update all machines in parallel for one step.

    Processes machine updates in order:
    1. Refuel machines that need power
    2. Run miners to extract resources
    3. Run conveyor belts to push items
    4. Run arms to pick and deposit items

    Args:
        state: Current environment state

    Returns:
        Updated environment state with all machines processed
    """
    state = refuel_machines(state)
    state = run_miners(state)
    state = run_conveyor_belts(state)
    state = run_arms(state)
    return state


def refuel_machines(state: EnvState) -> EnvState:
    """Convert coal to power for machines that need it.

    Machines with zero power and coal in slot 0 (the fuel/INPUT slot) will
    consume one coal and gain POWER_PER_COAL units of power.

    Args:
        state: Current environment state

    Returns:
        Updated state with refueled machines
    """
    is_miner = state.machine_types == MachineType.MINER
    needs_power = state.machine_power <= 0
    fuel_count = state.machine_inventory_counts[..., _MINER_FUEL_SLOT]
    has_fuel = fuel_count > 0

    should_refuel = is_miner & needs_power & has_fuel

    new_fuel = fuel_count - should_refuel.astype(jnp.int16)
    new_power = state.machine_power + (should_refuel.astype(jnp.int32) * POWER_PER_COAL)

    new_inv_counts = state.machine_inventory_counts.at[..., _MINER_FUEL_SLOT].set(
        new_fuel
    )
    return state.replace(
        machine_inventory_counts=new_inv_counts,
        machine_power=new_power,
    )


def run_miners(state: EnvState) -> EnvState:
    """Execute miner behavior for all miners in parallel.

    Miners with power extract resources from the block below them and
    deposit the items into their output slot (slot index 1). Power is only
    consumed when actually mining.

    Args:
        state: Current environment state

    Returns:
        Updated state with miner operations applied
    """
    is_miner = state.machine_types == MachineType.MINER
    has_power = state.machine_power > 0
    has_resources = state.block_resources > 0

    block_item = BLOCK_TO_ITEM_ARRAY[state.map]

    output_count = state.machine_inventory_counts[..., _MINER_OUTPUT_SLOT]
    output_item = state.machine_inventory_items[..., _MINER_OUTPUT_SLOT]

    output_empty = output_count == 0
    output_matches = output_item == block_item
    has_space = (output_empty | output_matches) & (
        output_count < MAX_MACHINE_STACK_SIZE
    )

    can_mine = is_miner & has_power & has_resources & has_space

    mining_rate = MACHINE_MINING_RATE[state.machine_types]
    available_space = MAX_MACHINE_STACK_SIZE - output_count
    mine_amount = jnp.minimum(mining_rate, state.block_resources)
    mine_amount = jnp.minimum(mine_amount, available_space)
    mine_amount = mine_amount * can_mine

    new_resources = state.block_resources - mine_amount.astype(jnp.int16)
    new_output_count = output_count + mine_amount.astype(jnp.int16)
    new_output_item = jnp.where(can_mine, block_item, output_item)

    actually_mined = mine_amount > 0
    power_cost = MACHINE_POWER_CONSUMPTION[state.machine_types] * actually_mined
    new_power = state.machine_power - power_cost

    is_depleted = (new_resources <= 0) & (state.block_resources > 0)
    new_map = jnp.where(is_depleted, BlockType.DIRT, state.map)

    new_inv_counts = state.machine_inventory_counts.at[..., _MINER_OUTPUT_SLOT].set(
        new_output_count
    )
    new_inv_items = state.machine_inventory_items.at[..., _MINER_OUTPUT_SLOT].set(
        new_output_item
    )

    return state.replace(
        block_resources=new_resources,
        machine_inventory_items=new_inv_items,
        machine_inventory_counts=new_inv_counts,
        machine_power=new_power,
        map=new_map,
    )


def run_conveyor_belts(state: EnvState) -> EnvState:
    """Move items along all conveyor belts in parallel.

    Each belt tile pushes the contents of its storage slot (slot 0) into the
    adjacent belt tile in the belt's facing direction.  Items flow only between
    belt tiles; non-belt tiles are never written.

    When two belts target the same destination (diverging topologies), counts
    are summed and clamped to MAX_MACHINE_STACK_SIZE.  Item type conflicts on
    the same tile produce undefined results and should be avoided by design.

    Args:
        state: Current environment state

    Returns:
        Updated state with belt items advanced one tile
    """
    h, w = state.machine_types.shape
    is_belt = state.machine_types == MachineType.CONVEYOR_BELT

    src_items = state.machine_inventory_items[..., _BELT_SLOT]
    src_counts = state.machine_inventory_counts[..., _BELT_SLOT]  # int16

    direction = state.machine_direction
    dx = (
        jnp.where(direction == Action.LEFT, -1, 0)
        + jnp.where(direction == Action.RIGHT, 1, 0)
    )
    dy = (
        jnp.where(direction == Action.UP, -1, 0)
        + jnp.where(direction == Action.DOWN, 1, 0)
    )

    rows = jnp.broadcast_to(jnp.arange(h)[:, None], (h, w))
    cols = jnp.broadcast_to(jnp.arange(w)[None, :], (h, w))

    tgt_row = jnp.clip(rows + dy, 0, h - 1)
    tgt_col = jnp.clip(cols + dx, 0, w - 1)

    tgt_is_belt = (
        state.machine_types[tgt_row, tgt_col] == MachineType.CONVEYOR_BELT
    )
    tgt_items = state.machine_inventory_items[tgt_row, tgt_col, _BELT_SLOT]
    tgt_counts = state.machine_inventory_counts[tgt_row, tgt_col, _BELT_SLOT]
    tgt_has_space = (
        (tgt_counts == 0) | (tgt_items == src_items)
    ) & (tgt_counts < MAX_MACHINE_STACK_SIZE)

    not_self = (tgt_row != rows) | (tgt_col != cols)
    can_push = (
        is_belt & (src_counts > 0) & tgt_is_belt & tgt_has_space & not_self
    )

    push_counts = jnp.where(can_push, src_counts, jnp.int16(0))
    push_items = jnp.where(can_push, src_items, 0)

    flat_tgt = (tgt_row * w + tgt_col).ravel()

    stayed_counts = jnp.where(can_push, jnp.int16(0), src_counts).ravel()
    stayed_items = jnp.where(can_push, 0, src_items).ravel()

    # Scatter incoming items to target positions (int16 for counts)
    recv_counts = jnp.zeros(h * w, dtype=jnp.int16).at[flat_tgt].add(
        push_counts.ravel()
    )
    recv_items = jnp.zeros(h * w, dtype=jnp.int32).at[flat_tgt].max(
        push_items.ravel()
    )

    final_counts = jnp.minimum(
        stayed_counts + recv_counts,
        jnp.int16(MAX_MACHINE_STACK_SIZE),
    ).reshape(h, w)
    final_items = jnp.where(
        recv_items.reshape(h, w) > 0,
        recv_items.reshape(h, w),
        stayed_items.reshape(h, w),
    )

    orig_items = state.machine_inventory_items[..., _BELT_SLOT]
    orig_counts = state.machine_inventory_counts[..., _BELT_SLOT]
    new_inv_items = state.machine_inventory_items.at[..., _BELT_SLOT].set(
        jnp.where(is_belt, final_items, orig_items)
    )
    new_inv_counts = state.machine_inventory_counts.at[..., _BELT_SLOT].set(
        jnp.where(is_belt, final_counts, orig_counts)
    )

    return state.replace(
        machine_inventory_items=new_inv_items,
        machine_inventory_counts=new_inv_counts,
    )


def run_arms(state: EnvState) -> EnvState:
    """Execute pick-and-place arm behavior for all arms in parallel.

    Each arm faces a direction set at placement time.  Every tick:

    Phase 1 — Deposit: if the arm's buffer (slot 0) holds items, find the
    first INPUT or STORAGE slot in the forward neighbour that can accept them
    and transfer the whole buffer there.

    Phase 2 — Pick: if the arm's buffer is now empty, find the first OUTPUT or
    STORAGE slot in the backward neighbour that holds items and move them into
    the arm buffer.

    Both phases operate vectorised over the full grid.  Concurrent access
    conflicts (two arms targeting the same slot) are resolved by the underlying
    scatter ops (additive for counts, max for item types) and may produce
    imprecise results for overlapping arm networks; well-designed layouts avoid
    this.

    Args:
        state: Current environment state

    Returns:
        Updated state with arm operations applied
    """
    h, w = state.machine_types.shape
    is_arm = state.machine_types == MachineType.ARM

    rows = jnp.broadcast_to(jnp.arange(h)[:, None], (h, w))
    cols = jnp.broadcast_to(jnp.arange(w)[None, :], (h, w))

    direction = state.machine_direction
    dx = (
        jnp.where(direction == Action.LEFT, -1, 0)
        + jnp.where(direction == Action.RIGHT, 1, 0)
    )
    dy = (
        jnp.where(direction == Action.UP, -1, 0)
        + jnp.where(direction == Action.DOWN, 1, 0)
    )

    fwd_row = jnp.clip(rows + dy, 0, h - 1)
    fwd_col = jnp.clip(cols + dx, 0, w - 1)
    bwd_row = jnp.clip(rows - dy, 0, h - 1)
    bwd_col = jnp.clip(cols - dx, 0, w - 1)

    state = _arm_deposit_phase(state, is_arm, rows, cols, fwd_row, fwd_col)
    state = _arm_pick_phase(state, is_arm, rows, cols, bwd_row, bwd_col)
    return state


def _arm_deposit_phase(
    state: EnvState,
    is_arm: jnp.ndarray,
    rows: jnp.ndarray,
    cols: jnp.ndarray,
    fwd_row: jnp.ndarray,
    fwd_col: jnp.ndarray,
) -> EnvState:
    """Deposit arm buffer contents into the forward neighbour.

    Args:
        state: Current environment state
        is_arm: Boolean mask of arm tiles, shape (H, W)
        rows: Row index grid, shape (H, W)
        cols: Col index grid, shape (H, W)
        fwd_row: Forward-neighbour row indices, shape (H, W)
        fwd_col: Forward-neighbour col indices, shape (H, W)

    Returns:
        Updated state after deposit
    """
    arm_items = state.machine_inventory_items[..., _ARM_SLOT]    # (H, W) int32
    arm_counts = state.machine_inventory_counts[..., _ARM_SLOT]  # (H, W) int16

    fwd_mtype = state.machine_types[fwd_row, fwd_col]
    fwd_not_none = fwd_mtype != MachineType.NONE
    fwd_not_self = (fwd_row != rows) | (fwd_col != cols)

    # Slot roles for every forward neighbour: (H, W, 8)
    fwd_slot_roles = _SLOT_ROLES_JAX[fwd_mtype]
    fwd_slot_items = state.machine_inventory_items[fwd_row, fwd_col]   # (H, W, 8)
    fwd_slot_counts = state.machine_inventory_counts[fwd_row, fwd_col]  # int16

    is_deposit_role = (fwd_slot_roles == _DEPOSIT_ROLE_INPUT) | (
        fwd_slot_roles == _DEPOSIT_ROLE_STORAGE
    )
    slot_empty = fwd_slot_counts == 0
    slot_matches = fwd_slot_items == arm_items[:, :, None]
    slot_has_space = (slot_empty | slot_matches) & (
        fwd_slot_counts < MAX_MACHINE_STACK_SIZE
    )
    slot_usable = is_deposit_role & slot_has_space  # (H, W, 8)

    deposit_idx = jnp.argmax(slot_usable.astype(jnp.int32), axis=-1)  # (H, W)
    has_deposit = jnp.any(slot_usable, axis=-1)

    can_deposit = (
        is_arm & (arm_counts > 0) & fwd_not_none & fwd_not_self & has_deposit
    )

    transfer_count = jnp.where(can_deposit, arm_counts, jnp.int16(0))  # int16
    transfer_item = jnp.where(can_deposit, arm_items, 0)

    new_counts = state.machine_inventory_counts.at[
        fwd_row, fwd_col, deposit_idx
    ].add(transfer_count)
    new_items = state.machine_inventory_items.at[
        fwd_row, fwd_col, deposit_idx
    ].max(transfer_item)

    # Clear the arm buffer for arms that successfully deposited.
    # Use new_counts/new_items as the base so the deposit scatter is preserved.
    new_counts = new_counts.at[..., _ARM_SLOT].set(
        jnp.where(can_deposit, jnp.int16(0), new_counts[..., _ARM_SLOT])
    )
    new_items = new_items.at[..., _ARM_SLOT].set(
        jnp.where(can_deposit, 0, new_items[..., _ARM_SLOT])
    )

    return state.replace(
        machine_inventory_items=new_items,
        machine_inventory_counts=new_counts,
    )


def _arm_pick_phase(
    state: EnvState,
    is_arm: jnp.ndarray,
    rows: jnp.ndarray,
    cols: jnp.ndarray,
    bwd_row: jnp.ndarray,
    bwd_col: jnp.ndarray,
) -> EnvState:
    """Fill an empty arm buffer by picking from the backward neighbour.

    Args:
        state: Current environment state (post-deposit)
        is_arm: Boolean mask of arm tiles, shape (H, W)
        rows: Row index grid, shape (H, W)
        cols: Col index grid, shape (H, W)
        bwd_row: Backward-neighbour row indices, shape (H, W)
        bwd_col: Backward-neighbour col indices, shape (H, W)

    Returns:
        Updated state after pick
    """
    arm_counts = state.machine_inventory_counts[..., _ARM_SLOT]  # (H, W) int16

    bwd_mtype = state.machine_types[bwd_row, bwd_col]
    bwd_not_none = bwd_mtype != MachineType.NONE
    bwd_not_self = (bwd_row != rows) | (bwd_col != cols)

    # Slot roles for every backward neighbour: (H, W, 8)
    bwd_slot_roles = _SLOT_ROLES_JAX[bwd_mtype]
    bwd_slot_items = state.machine_inventory_items[bwd_row, bwd_col]    # (H, W, 8)
    bwd_slot_counts = state.machine_inventory_counts[bwd_row, bwd_col]  # int16

    is_pick_role = (bwd_slot_roles == _PICK_ROLE_OUTPUT) | (
        bwd_slot_roles == _PICK_ROLE_STORAGE
    )
    slot_pickable = is_pick_role & (bwd_slot_counts > 0)  # (H, W, 8)

    pick_idx = jnp.argmax(slot_pickable.astype(jnp.int32), axis=-1)  # (H, W)
    has_pick = jnp.any(slot_pickable, axis=-1)

    can_pick = is_arm & (arm_counts == 0) & bwd_not_none & bwd_not_self & has_pick

    # Gather item/count at the selected pick slot via one-hot mask.
    slot_one_hot = jax.nn.one_hot(pick_idx, 8, dtype=jnp.int16)  # (H, W, 8)
    picked_item = jnp.sum(
        bwd_slot_items * slot_one_hot.astype(jnp.int32), axis=-1
    )  # (H, W)
    picked_count = jnp.sum(
        bwd_slot_counts * slot_one_hot, axis=-1
    ).astype(jnp.int16)

    pick_item = jnp.where(can_pick, picked_item, 0)
    pick_count = jnp.where(can_pick, picked_count, jnp.int16(0))

    # Clear the picked slot in the backward neighbour.
    new_counts = state.machine_inventory_counts.at[
        bwd_row, bwd_col, pick_idx
    ].add(-pick_count)

    # Zero out item type where the picked slot was fully depleted.
    # Use a min-scatter: depleted slots scatter 0, others scatter their
    # existing value (a no-op under min since item types are >= 0).
    remaining_at_slot = jnp.sum(
        new_counts[bwd_row, bwd_col] * slot_one_hot, axis=-1
    ).astype(jnp.int16)
    slot_depleted = (remaining_at_slot <= 0) & can_pick
    depleted_item = jnp.where(
        slot_depleted,
        0,
        state.machine_inventory_items[bwd_row, bwd_col, pick_idx],
    )
    new_items = state.machine_inventory_items.at[
        bwd_row, bwd_col, pick_idx
    ].min(depleted_item)

    # Fill the arm buffer.  Use new_counts/new_items (not the original
    # state) so the scatter changes above are preserved for non-arm tiles
    # whose slot 0 was modified (e.g. the chest that was just picked from).
    new_counts = new_counts.at[..., _ARM_SLOT].set(
        jnp.where(
            can_pick,
            pick_count,
            new_counts[..., _ARM_SLOT],
        )
    )
    new_items = new_items.at[..., _ARM_SLOT].set(
        jnp.where(can_pick, pick_item, new_items[..., _ARM_SLOT])
    )

    return state.replace(
        machine_inventory_items=new_items,
        machine_inventory_counts=new_counts,
    )
