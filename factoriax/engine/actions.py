"""Lookup tables between an item and the action that addresses it.

The name of a ``PLACE_``, ``CRAFT_``, or ``DEPOSIT_`` action contains the name
of the item that the action addresses. An action and an item therefore
determine each other. This module holds both directions as ``jnp`` gather
arrays. The item tuples in :mod:`factoriax.engine.constants` supply the
contents.

The two directions answer different questions. ``*_ACTION_TO_ITEM`` maps an
action offset to the item that the action addresses. ``execute_action`` in
:mod:`factoriax.engine.step` reads this table once it holds an action, and
indexes it with a traced value inside ``jax.jit``. ``ITEM_TO_*_ACTION`` maps an
item back to an absolute action. The play UI reads this table to turn a click
on an item into an action, and indexes it on the host with a Python int.

An import of this module runs :func:`_validate`. If two tables disagree,
:func:`_validate` raises ``ValueError``. The module then fails to import, and
the engine cannot dispatch an action to the wrong item.
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

#: Backward-table entry for an item with no action in that list. The value is
#: negative because ``0`` is itself an action (``MoveAction.NOOP``). A caller
#: must compare against this value, and must not test truthiness.
NO_ACTION: int = -1


def _build_forward_table(items: Iterable[IntEnum]) -> list[int]:
    """Build the offset-to-item table for one item list.

    Parameters
    ----------
    items
        Items that the actions address, in action order:
        :data:`~factoriax.engine.constants.PLACEMENT_ITEMS`,
        :data:`~factoriax.engine.constants.CRAFT_ITEMS`, or
        :data:`~factoriax.engine.constants.DEPOSIT_ITEMS`. The entries come
        from the category enums. Each entry name is also the name of an
        :class:`~factoriax.engine.constants.ItemType` member.

    Returns
    -------
    list[int]
        Item ids, one for each action. Each id sits at the offset of its
        action from ``PLACE_BASE``, ``CRAFT_BASE``, or ``DEPOSIT_BASE``.

    Raises
    ------
    KeyError
        If a member name has no matching ``ItemType``. The category enums and
        ``ItemType`` are then out of agreement.
    """
    return [int(ItemType[m.name]) for m in items]


def _build_backward_table(items: Iterable[IntEnum], prefix: str) -> list[int]:
    """Build the item-to-action table for one item list.

    Parameters
    ----------
    items
        Items that the actions address, as passed to
        :func:`_build_forward_table`. The order does not matter here, because
        the function writes each entry at its own item id.
    prefix
        Action name prefix, one of ``"PLACE"``, ``"CRAFT"``, or ``"DEPOSIT"``.
        The function joins this prefix to an item name with an underscore, and
        finds the :class:`~factoriax.engine.constants.Action` member of that
        name.

    Returns
    -------
    list[int]
        One entry for each item id, of length ``NUM_ITEM_TYPES``. Each entry is
        an absolute ``Action`` id, or :data:`NO_ACTION` for an item that has no
        such action. ``ItemType.EMPTY`` at index 0 is one such item.

    Raises
    ------
    KeyError
        If ``prefix`` and a member name together do not give an existing
        ``Action``.
    """
    table = [NO_ACTION] * NUM_ITEM_TYPES
    for m in items:
        table[int(ItemType[m.name])] = int(Action[f"{prefix}_{m.name}"])
    return table


# Forward: action offset -> item. The Python lists feed both the jnp arrays
# that follow and the round-trip test in _validate.
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
#: for every item that is not a machine, ``EMPTY`` included.
ITEM_TO_PLACE_ACTION = jnp.array(_PLACE_BWD, dtype=jnp.int32)
#: ``CRAFT_`` action for each item, indexed by ``ItemType``. :data:`NO_ACTION`
#: for the raw resources, which are mined rather than crafted, and for
#: ``EMPTY``.
ITEM_TO_CRAFT_ACTION = jnp.array(_CRAFT_BWD, dtype=jnp.int32)
#: ``DEPOSIT_`` action for each item, indexed by ``ItemType``. :data:`NO_ACTION`
#: at ``EMPTY`` and nowhere else: every real item can be deposited.
ITEM_TO_DEPOSIT_ACTION = jnp.array(_DEPOSIT_BWD, dtype=jnp.int32)


def _validate() -> None:
    """Make sure that the forward and backward tables agree in both directions.

    This function runs once at import, over the six tables that this module
    builds. It walks each pair of tables twice: offset to item and back, then
    item to action and back. One pass is not sufficient. The backward table can
    map an item to an action that the forward table assigns to a different
    item, and one pass does not find this error.

    Raises
    ------
    ValueError
        If a walk in either direction does not return the value it started
        from. The message gives the action prefix and the offset or item at
        fault. The module then fails to import, and the engine never runs on
        tables that disagree.
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
