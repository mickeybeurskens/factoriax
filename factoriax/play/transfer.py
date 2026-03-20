"""Item transfer logic between player inventory and machine slots.

Pure Python — no pygame dependency — so it can be unit-tested headlessly.
These functions are intentionally outside the JAX action pipeline; machine
interaction is a UI operation and does not need to be part of the RL action
space.

Transfer rules
--------------
* **Withdraw** (machine → player): always allowed for any non-NONE slot.
  The whole stack is taken from the machine slot, merged into the first
  matching player stack that has space, then the remainder placed in the
  first empty player slot.  If the player inventory cannot absorb everything,
  the remainder stays in the machine slot.

* **Deposit** (player → machine): only allowed into INPUT or STORAGE slots.
  The focused machine slot is tried first; if it can't accept (wrong role,
  wrong item, or full) the function scans left-to-right for the first slot
  holding the same item with space, then for the first empty compatible slot.
  Up to ``MAX_MACHINE_STACK_SIZE`` items are moved; any remainder stays in
  the player slot.
"""

from __future__ import annotations

import jax.numpy as jnp

from factoriax.constants import (
    MACHINE_NUM_SLOTS,
    MACHINE_SLOT_ROLES,
    MAX_MACHINE_STACK_SIZE,
    MAX_STACK_SIZE,
    NUM_INVENTORY_SLOTS,
    SlotRole,
)
from factoriax.state import EnvState

_DEPOSIT_ROLES: frozenset[int] = frozenset(
    [int(SlotRole.INPUT), int(SlotRole.STORAGE)]
)


def swap_inventory_slots(
    state: EnvState,
    player_idx: int,
    slot_a: int,
    slot_b: int,
) -> EnvState:
    """Swap the contents of two inventory slots for a player.

    If both slots are identical this is a no-op.  Works correctly when
    one or both slots are empty.

    Args:
        state: Current environment state.
        player_idx: Index of the acting player.
        slot_a: First inventory slot index.
        slot_b: Second inventory slot index.

    Returns:
        Updated state with the two slots' items and counts exchanged.
    """
    if slot_a == slot_b:
        return state

    item_a = int(state.inventory_items[player_idx, slot_a])
    count_a = int(state.inventory_counts[player_idx, slot_a])
    item_b = int(state.inventory_items[player_idx, slot_b])
    count_b = int(state.inventory_counts[player_idx, slot_b])

    new_items = (
        state.inventory_items.at[player_idx, slot_a]
        .set(item_b)
        .at[player_idx, slot_b]
        .set(item_a)
    )
    new_counts = (
        state.inventory_counts.at[player_idx, slot_a]
        .set(count_b)
        .at[player_idx, slot_b]
        .set(count_a)
    )
    return state.replace(
        inventory_items=new_items,
        inventory_counts=new_counts,
    )


def withdraw_from_machine(
    state: EnvState,
    player_idx: int,
    tx: int,
    ty: int,
    machine_slot_idx: int,
) -> EnvState:
    """Move the full stack from a machine slot into the player's inventory.

    The stack is first merged into any existing matching player stack with
    room, then placed into the first empty player slot.  If the player
    inventory has no room for the whole stack the remainder is returned to
    the machine slot.

    Args:
        state: Current environment state.
        player_idx: Index of the acting player.
        tx: X tile coordinate of the machine.
        ty: Y tile coordinate of the machine.
        machine_slot_idx: Index of the machine slot to withdraw from.

    Returns:
        Updated state, or the original state unchanged if the slot is empty
        or the player inventory is completely full.
    """
    item = int(state.machine_inventory_items[ty, tx, machine_slot_idx])
    count = int(state.machine_inventory_counts[ty, tx, machine_slot_idx])
    if item == 0 or count == 0:
        return state

    new_m_items = state.machine_inventory_items.at[ty, tx, machine_slot_idx].set(0)
    new_m_counts = state.machine_inventory_counts.at[ty, tx, machine_slot_idx].set(0)

    p_items = [int(state.inventory_items[player_idx, s]) for s in range(NUM_INVENTORY_SLOTS)]
    p_counts = [int(state.inventory_counts[player_idx, s]) for s in range(NUM_INVENTORY_SLOTS)]

    remaining = count

    # Merge into existing matching stacks first.
    for s in range(NUM_INVENTORY_SLOTS):
        if remaining == 0:
            break
        if p_items[s] == item:
            space = MAX_STACK_SIZE - p_counts[s]
            transfer = min(remaining, space)
            p_counts[s] += transfer
            remaining -= transfer

    # Place remainder into empty slots.
    for s in range(NUM_INVENTORY_SLOTS):
        if remaining == 0:
            break
        if p_items[s] == 0:
            p_items[s] = item
            p_counts[s] = remaining
            remaining = 0

    if remaining > 0:
        new_m_items = new_m_items.at[ty, tx, machine_slot_idx].set(item)
        new_m_counts = new_m_counts.at[ty, tx, machine_slot_idx].set(remaining)

    new_inv_items = state.inventory_items.at[player_idx].set(
        jnp.array(p_items, dtype=jnp.int32)
    )
    new_inv_counts = state.inventory_counts.at[player_idx].set(
        jnp.array(p_counts, dtype=jnp.int32)
    )
    return state.replace(
        machine_inventory_items=new_m_items,
        machine_inventory_counts=new_m_counts,
        inventory_items=new_inv_items,
        inventory_counts=new_inv_counts,
    )


def deposit_to_machine(
    state: EnvState,
    player_idx: int,
    tx: int,
    ty: int,
    machine_slot_idx: int,
    player_slot_idx: int,
) -> EnvState:
    """Move items from a player slot into the machine inventory.

    The focused machine slot is tried first.  If it cannot accept (wrong
    role, mismatched item, or full) the function auto-routes: it scans
    left-to-right for the first INPUT/STORAGE slot holding the same item
    with space, then for the first empty INPUT/STORAGE slot.  Does nothing
    if no compatible slot is found.

    Args:
        state: Current environment state.
        player_idx: Index of the acting player.
        tx: X tile coordinate of the machine.
        ty: Y tile coordinate of the machine.
        machine_slot_idx: Focused machine slot (tried first).
        player_slot_idx: Source slot in the player's inventory.

    Returns:
        Updated state, or the original state unchanged if the player slot is
        empty or no compatible machine slot exists.
    """
    item = int(state.inventory_items[player_idx, player_slot_idx])
    count = int(state.inventory_counts[player_idx, player_slot_idx])
    if item == 0 or count == 0:
        return state

    machine_type = int(state.machine_types[ty, tx])
    num_slots = int(MACHINE_NUM_SLOTS[machine_type])
    target = _find_deposit_slot(state, tx, ty, item, machine_slot_idx, num_slots, machine_type)
    if target is None:
        return state

    m_count = int(state.machine_inventory_counts[ty, tx, target])
    space = MAX_MACHINE_STACK_SIZE - m_count
    transfer = min(count, space)
    if transfer <= 0:
        return state

    new_m_items = state.machine_inventory_items.at[ty, tx, target].set(item)
    new_m_counts = state.machine_inventory_counts.at[ty, tx, target].set(
        m_count + transfer
    )

    remaining = count - transfer
    if remaining == 0:
        new_inv_items = state.inventory_items.at[player_idx, player_slot_idx].set(0)
        new_inv_counts = state.inventory_counts.at[player_idx, player_slot_idx].set(0)
    else:
        new_inv_items = state.inventory_items
        new_inv_counts = state.inventory_counts.at[player_idx, player_slot_idx].set(
            remaining
        )

    return state.replace(
        machine_inventory_items=new_m_items,
        machine_inventory_counts=new_m_counts,
        inventory_items=new_inv_items,
        inventory_counts=new_inv_counts,
    )


def _find_deposit_slot(
    state: EnvState,
    tx: int,
    ty: int,
    item: int,
    focused_slot: int,
    num_slots: int,
    machine_type: int,
) -> int | None:
    """Return the best machine slot index to deposit ``item`` into, or None.

    Priority:
    1. Focused slot (if compatible role, same item or empty, has space).
    2. First slot left-to-right with same item and space.
    3. First empty INPUT/STORAGE slot left-to-right.

    Args:
        state: Current environment state.
        tx: X tile coordinate of the machine.
        ty: Y tile coordinate of the machine.
        item: Item type being deposited.
        focused_slot: Currently focused machine slot (tried first).
        num_slots: Number of active slots for this machine type.
        machine_type: Integer machine type for role lookup.

    Returns:
        Slot index to deposit into, or ``None`` if no slot can accept.
    """
    slot_roles = MACHINE_SLOT_ROLES[machine_type]

    # Step 1: try focused slot.
    role = int(slot_roles[focused_slot])
    if role in _DEPOSIT_ROLES:
        m_item = int(state.machine_inventory_items[ty, tx, focused_slot])
        m_count = int(state.machine_inventory_counts[ty, tx, focused_slot])
        if (m_item == item or m_item == 0) and m_count < MAX_MACHINE_STACK_SIZE:
            return focused_slot

    # Step 2: scan for a slot with the same item and space.
    for s in range(num_slots):
        if s == focused_slot:
            continue
        if int(slot_roles[s]) not in _DEPOSIT_ROLES:
            continue
        m_item = int(state.machine_inventory_items[ty, tx, s])
        m_count = int(state.machine_inventory_counts[ty, tx, s])
        if m_item == item and m_count < MAX_MACHINE_STACK_SIZE:
            return s

    # Step 3: scan for any empty compatible slot.
    for s in range(num_slots):
        if s == focused_slot:
            continue
        if int(slot_roles[s]) not in _DEPOSIT_ROLES:
            continue
        if int(state.machine_inventory_items[ty, tx, s]) == 0:
            return s

    return None
