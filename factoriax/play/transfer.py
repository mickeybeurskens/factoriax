"""Inventory slot swap for the interactive player UI.

Pure Python, no pygame dependency, so it can be unit-tested headlessly.
This is a UI-only convenience for rearranging inventory layout and is
intentionally outside the JAX action pipeline. Game-affecting transfers
(deposit, withdraw, rotate) are handled by :mod:`factoriax.game_logic`
so that RL agents and the interactive player share the same mechanics.
"""

from __future__ import annotations

from factoriax.constants import MAX_STACK_SIZE
from factoriax.state import EnvState


def swap_inventory_slots(
    state: EnvState,
    player_idx: int,
    slot_a: int,
    slot_b: int,
) -> EnvState:
    """Move or merge the item in *slot_a* into *slot_b*.

    When both slots hold the same item type the stacks are merged up to
    ``MAX_STACK_SIZE``, with any overflow remaining in *slot_a*.  When
    the item types differ (or one slot is empty) the two slots are
    swapped outright.  Same-slot calls are a no-op.

    Args:
        state: Current environment state.
        player_idx: Index of the acting player.
        slot_a: Source inventory slot (the "held" item).
        slot_b: Destination inventory slot.

    Returns:
        Updated state with stacks merged or swapped.
    """
    if slot_a == slot_b:
        return state

    item_a = int(state.inventory_items[player_idx, slot_a])
    count_a = int(state.inventory_counts[player_idx, slot_a])
    item_b = int(state.inventory_items[player_idx, slot_b])
    count_b = int(state.inventory_counts[player_idx, slot_b])

    # Merge when both slots hold the same non-empty item type.
    if item_a != 0 and item_a == item_b:
        transfer = min(count_a, MAX_STACK_SIZE - count_b)
        new_count_b = count_b + transfer
        new_count_a = count_a - transfer
        new_item_a = item_a if new_count_a > 0 else 0

        new_items = state.inventory_items.at[
            player_idx, slot_a
        ].set(new_item_a)
        new_counts = (
            state.inventory_counts.at[player_idx, slot_a]
            .set(new_count_a)
            .at[player_idx, slot_b]
            .set(new_count_b)
        )
    else:
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
