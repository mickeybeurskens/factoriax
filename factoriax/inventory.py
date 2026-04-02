"""Shared inventory primitives for the FactoriaX environment.

Provides pure-array inventory operations used by crafting, placement,
and game logic modules. All functions operate on raw JAX arrays (not
EnvState) so they compose freely across contexts: actual state
mutation, dry-run simulation for fit checks, and single-item fast
paths.
"""

import jax
import jax.numpy as jnp
from jax import lax

from factoriax.constants import ItemType


def add_items_to_slots(
    items: jax.Array,
    counts: jax.Array,
    item_type: int | jax.Array,
    amount: int | jax.Array,
    max_stack: int | jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Add items to inventory slots using stack-then-empty logic.

    Scans slots left to right. First fills existing stacks of the same
    item type up to ``max_stack``, then places remaining items into
    empty slots. Returns the updated arrays and any remainder that did
    not fit.

    Args:
        items: Slot item types, shape ``(N,)``.
        counts: Slot counts, shape ``(N,)``.
        item_type: Item type to add.
        amount: Quantity to add.
        max_stack: Maximum stack size per slot.

    Returns:
        Tuple of ``(new_items, new_counts, remaining)`` where
        ``remaining`` is the amount that did not fit.
    """

    def _scan_slot(
        carry: tuple[jax.Array, jax.Array, jax.Array], slot_idx: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], None]:
        items_arr, counts_arr, remaining_amt = carry
        slot_item = items_arr[slot_idx]
        slot_count = counts_arr[slot_idx]

        can_stack = (slot_item == item_type) & (slot_count < max_stack)
        is_empty = slot_item == ItemType.EMPTY

        space = jnp.where(can_stack, max_stack - slot_count, 0)
        space = jnp.where(is_empty, max_stack, space)

        to_add = jnp.minimum(remaining_amt, space)

        new_count = slot_count + to_add
        new_remaining = remaining_amt - to_add

        items_arr = jnp.where(
            is_empty & (to_add > 0),
            items_arr.at[slot_idx].set(item_type),
            items_arr,
        )
        counts_arr = counts_arr.at[slot_idx].set(new_count)

        return (items_arr, counts_arr, new_remaining), None

    num_slots = items.shape[0]
    (new_items, new_counts, remaining), _ = lax.scan(
        _scan_slot,
        (items, counts, jnp.int32(amount)),
        jnp.arange(num_slots),
    )
    return new_items, new_counts, remaining


def find_best_slot(
    items: jax.Array,
    counts: jax.Array,
    item_type: int | jax.Array,
    max_stack: int | jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Find the best slot for adding a single item.

    Prefers a matching slot (same item type with room) over an empty
    slot. This is the fast path used by ``mine_block`` where only one
    item is added at a time.

    Args:
        items: Slot item types, shape ``(N,)``.
        counts: Slot counts, shape ``(N,)``.
        item_type: Item type to add.
        max_stack: Maximum stack size per slot.

    Returns:
        Tuple of ``(slot_idx, can_add)`` where ``can_add`` is a
        boolean indicating whether any valid slot was found.
    """
    matching_mask = (items == item_type) & (counts < max_stack)
    has_matching = jnp.any(matching_mask)
    matching_idx = jnp.argmax(matching_mask)

    empty_mask = items == ItemType.EMPTY
    has_empty = jnp.any(empty_mask)
    empty_idx = jnp.argmax(empty_mask)

    can_add = has_matching | has_empty
    slot_idx = jnp.where(has_matching, matching_idx, empty_idx)

    return slot_idx, can_add
