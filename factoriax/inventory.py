"""Pouch-based inventory primitives for the FactoriaX environment.

Provides pure-array inventory operations for the typed pouch system.
Each inventory (player or machine) is a flat array of counts indexed
by ItemType. There are no slots, cursors, or ordering concerns.

All functions operate on raw JAX arrays (not EnvState) so they compose
freely across contexts: actual state mutation, dry-run fit checks, and
machine transfer logic.
"""

import jax
import jax.numpy as jnp

from factoriax.constants import ItemType


def add_to_pouch(
    counts: jax.Array,
    item_type: int | jax.Array,
    amount: int | jax.Array,
    max_stack: jax.Array | int,
) -> tuple[jax.Array, jax.Array]:
    """Add items to a pouch, respecting per-type max stack.

    Args:
        counts: Current counts per item type, shape ``(NUM_ITEM_TYPES,)``.
        item_type: ItemType index to add to.
        amount: Quantity to add.
        max_stack: Max count per type. Either a scalar or an array of
            shape ``(NUM_ITEM_TYPES,)`` for per-type limits.

    Returns:
        Tuple of ``(new_counts, remaining)`` where ``remaining`` is the
        amount that did not fit.
    """
    item_type = jnp.int32(item_type)
    amount = jnp.int32(amount)
    current = counts[item_type]
    is_array = hasattr(max_stack, "__getitem__") and jnp.ndim(max_stack) > 0
    cap = max_stack[item_type] if is_array else max_stack
    space = jnp.maximum(0, cap - current)
    to_add = jnp.minimum(amount, space)
    new_counts = counts.at[item_type].set(current + to_add)
    remaining = amount - to_add
    return new_counts, remaining


def remove_from_pouch(
    counts: jax.Array,
    item_type: int | jax.Array,
    amount: int | jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Remove items from a pouch.

    Args:
        counts: Current counts per item type, shape ``(NUM_ITEM_TYPES,)``.
        item_type: ItemType index to remove from.
        amount: Quantity to remove.

    Returns:
        Tuple of ``(new_counts, removed)`` where ``removed`` is the
        actual amount removed (may be less than requested).
    """
    item_type = jnp.int32(item_type)
    amount = jnp.int32(amount)
    current = counts[item_type]
    removed = jnp.minimum(amount, current)
    new_counts = counts.at[item_type].set(current - removed)
    return new_counts, removed


def count_item(counts: jax.Array, item_type: int | jax.Array) -> jax.Array:
    """Read the count for a specific item type.

    Args:
        counts: Counts per item type, shape ``(NUM_ITEM_TYPES,)``.
        item_type: ItemType index to read.

    Returns:
        Scalar count.
    """
    return counts[jnp.int32(item_type)]


def num_distinct_types(counts: jax.Array) -> jax.Array:
    """Count how many distinct item types are stored (non-zero, excluding EMPTY).

    Args:
        counts: Counts per item type, shape ``(NUM_ITEM_TYPES,)``.

    Returns:
        Scalar count of distinct types.
    """
    # Skip index 0 (EMPTY) which is never a real item.
    return jnp.sum(counts[1:] > 0)


def can_add_to_machine(
    machine_counts: jax.Array,
    item_type: int | jax.Array,
    amount: int | jax.Array,
    max_types: int | jax.Array,
    max_stack: int | jax.Array,
) -> jax.Array:
    """Check if items can be added to a machine pouch.

    Verifies both the per-type stack limit and the max distinct types
    constraint (e.g., belts hold 1 type, pallets hold 1).

    Args:
        machine_counts: Machine inventory counts, shape ``(NUM_ITEM_TYPES,)``.
        item_type: ItemType to add.
        amount: Quantity to add.
        max_types: Maximum distinct item types this machine holds.
        max_stack: Maximum count per type for this machine.

    Returns:
        Boolean scalar: True if the deposit is valid.
    """
    item_type = jnp.int32(item_type)
    amount = jnp.int32(amount)
    current = machine_counts[item_type]
    has_space = current + amount <= max_stack
    is_existing_type = current > 0
    distinct = num_distinct_types(machine_counts)
    has_type_room = is_existing_type | (distinct < max_types)
    valid_item = item_type != int(ItemType.EMPTY)
    return has_space & has_type_room & valid_item & (amount > 0)
