"""JAX compatible lookup tables between an item and the action that acts on it.

A ``PLACE_``, ``CRAFT_``, or ``DEPOSIT_`` action is named after the item it acts
on, so the action and the item can be looked up from each other. The environment
step function resolves action to item inside the JIT boundary. Host-side callers
(scripted agents, the play UI) ask the opposite question: which action places,
crafts, or deposits a given item. This module holds both directions as jnp
gather arrays, derived from the tuples in :mod:`factoriax.engine.constants`.
Indexing them is a traced gather, so both are usable inside ``jax.jit``.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import IntEnum

import jax.numpy as jnp

from factoriax.engine.constants import (
    CRAFT_BASE,
    CRAFT_ITEMS,
    DEPOSIT_BASE,
    DEPOSIT_ITEMS,
    NUM_ITEM_TYPES,
    PLACE_BASE,
    PLACEMENT_ITEMS,
    Action,
    ItemType,
)

#: Backward-table entry for an item with no action in that list.
NO_ACTION: int = -1


def _build_forward_table(items: Iterable[IntEnum]) -> list[int]:
    """Build the offset-to-item table for one item list.

    Parameters
    ----------
    items : Iterable[IntEnum]
        Items the actions address, in action order:
        :data:`~factoriax.engine.constants.PLACEMENT_ITEMS`,
        :data:`~factoriax.engine.constants.CRAFT_ITEMS`, or
        :data:`~factoriax.engine.constants.DEPOSIT_ITEMS`. Entries come from
        the category enums, so each name also names an
        :class:`~factoriax.engine.constants.ItemType` member.

    Returns
    -------
    list[int]
        Item ids, one per action, positioned at that action's offset from
        ``PLACE_BASE``, ``CRAFT_BASE``, or ``DEPOSIT_BASE``.

    Raises
    ------
    KeyError
        If a member name has no matching ``ItemType``, which means the
        category enums and ``ItemType`` have drifted apart.
    """
    return [int(ItemType[m.name]) for m in items]


def _build_backward_table(items: Iterable[IntEnum], prefix: str) -> list[int]:
    """Build the item-to-action table for one item list.

    Parameters
    ----------
    items : Iterable[IntEnum]
        Items the actions address, as passed to :func:`_build_forward_table`.
        Order does not matter here, because each entry is written at its own
        item id.
    prefix : str
        Action name prefix, one of ``"PLACE"``, ``"CRAFT"``, or ``"DEPOSIT"``.
        Joined to an item name with an underscore to look up the
        :class:`~factoriax.engine.constants.Action` member.

    Returns
    -------
    list[int]
        One entry per item id, of length ``NUM_ITEM_TYPES``. Each entry is an
        absolute ``Action`` id, or :data:`NO_ACTION` for an item that has no
        such action, including ``ItemType.EMPTY`` at index 0.

    Raises
    ------
    KeyError
        If ``prefix`` and a member name do not name an existing ``Action``.
    """
    table = [NO_ACTION] * NUM_ITEM_TYPES
    for m in items:
        table[int(ItemType[m.name])] = int(Action[f"{prefix}_{m.name}"])
    return table


# Forward: action offset -> item. The Python lists feed both the jnp arrays
# below and the round-trip check in _validate.
_PLACE_FWD = _build_forward_table(PLACEMENT_ITEMS)
_CRAFT_FWD = _build_forward_table(CRAFT_ITEMS)
_DEPOSIT_FWD = _build_forward_table(DEPOSIT_ITEMS)

#: Item placed by each ``PLACE_`` action, indexed by ``action - PLACE_BASE``.
PLACE_ACTION_TO_ITEM = jnp.array(_PLACE_FWD, dtype=jnp.int32)
#: Item crafted by each ``CRAFT_`` action, indexed by ``action - CRAFT_BASE``.
CRAFT_ACTION_TO_ITEM = jnp.array(_CRAFT_FWD, dtype=jnp.int32)
#: Item deposited by each ``DEPOSIT_`` action, indexed by
#: ``action - DEPOSIT_BASE``.
DEPOSIT_ACTION_TO_ITEM = jnp.array(_DEPOSIT_FWD, dtype=jnp.int32)

# Backward: item -> absolute action.
_PLACE_BWD = _build_backward_table(PLACEMENT_ITEMS, "PLACE")
_CRAFT_BWD = _build_backward_table(CRAFT_ITEMS, "CRAFT")
_DEPOSIT_BWD = _build_backward_table(DEPOSIT_ITEMS, "DEPOSIT")

#: ``PLACE_`` action for each item, indexed by ``ItemType``. :data:`NO_ACTION`
#: for the items that cannot be placed.
ITEM_TO_PLACE_ACTION = jnp.array(_PLACE_BWD, dtype=jnp.int32)
#: ``CRAFT_`` action for each item, indexed by ``ItemType``. :data:`NO_ACTION`
#: for raw resources, which are mined rather than crafted.
ITEM_TO_CRAFT_ACTION = jnp.array(_CRAFT_BWD, dtype=jnp.int32)
#: ``DEPOSIT_`` action for each item, indexed by ``ItemType``. :data:`NO_ACTION`
#: only at ``ItemType.EMPTY``.
ITEM_TO_DEPOSIT_ACTION = jnp.array(_DEPOSIT_BWD, dtype=jnp.int32)


def _validate() -> None:
    """Check that the forward and backward tables agree, in both directions.

    Runs once at import over the six tables this module builds. Each pair is
    walked twice, offset to item and back, then item to action and back. One
    pass is not enough, because it would miss an item that the backward table
    maps to an action the forward table assigns to a different item.

    Raises
    ------
    ValueError
        If either direction fails to round-trip, naming the action prefix and
        the offending offset or item. The module then fails to import, so the
        engine never runs on tables that disagree.
    """
    for fwd, inv, base, prefix in (
        (_PLACE_FWD, _PLACE_BWD, PLACE_BASE, "PLACE"),
        (_CRAFT_FWD, _CRAFT_BWD, CRAFT_BASE, "CRAFT"),
        (_DEPOSIT_FWD, _DEPOSIT_BWD, DEPOSIT_BASE, "DEPOSIT"),
    ):
        for offset, item in enumerate(fwd):
            if inv[item] != base + offset:
                raise ValueError(
                    f"{prefix} tables do not match: offset {offset} maps "
                    f"to item {item}, but that item maps back to action "
                    f"{inv[item]} (expected {base + offset})."
                )
        for item, action in enumerate(inv):
            if action == NO_ACTION:
                continue
            if fwd[action - base] != item:
                raise ValueError(
                    f"{prefix} backward table for item {item} -> action {action} does "
                    f"not round-trip (forward gives {fwd[action - base]})."
                )


_validate()
