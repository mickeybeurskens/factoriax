"""Item <-> action mappings for FactoriaX.

The engine resolves an *action* to the item it refers to inside the JIT step;
host-side callers (scripted agents, the play UI) need the *inverse* -- given an
item, which action crafts / places / deposits it. Both directions are jnp
gather tables derived from the same category family lists in
:mod:`factoriax.engine.constants`, usable inside ``jax.jit``.

Layout
------
* **Forward** (``action offset -> item``): indexed by ``action - <FAMILY>_BASE``;
  one entry per action in the family, in family order. Consumed by the step
  dispatcher in :mod:`factoriax.engine.game_logic`.
* **Inverse** (``item -> absolute action``): indexed by ``ItemType``; holds the
  flat ``Action`` id, or :data:`NO_ACTION` (``-1``) where the item has no
  action in that family. Resources are non-craftable and non-machines are
  non-placeable, so those families are partial; deposit is total over every
  non-EMPTY item.

The two directions are asserted to be mutual inverses at import time, mirroring
``RecipeBook``'s validate-at-construction.
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

#: Inverse-table sentinel: the item has no action in that family.
NO_ACTION: int = -1


def _forward(items: Iterable[IntEnum]) -> list[int]:
    """Family offset -> item id, one row per action (family order)."""
    return [int(ItemType[m.name]) for m in items]


def _inverse(items: Iterable[IntEnum], prefix: str) -> list[int]:
    """Item id -> absolute ``Action`` id, ``NO_ACTION`` where none exists."""
    table = [NO_ACTION] * NUM_ITEM_TYPES
    for m in items:
        table[int(ItemType[m.name])] = int(Action[f"{prefix}_{m.name}"])
    return table


# Forward: action offset -> item. Indexed by ``action - <FAMILY>_BASE``.
_PLACE_FWD = _forward(PLACEMENT_ITEMS)
_CRAFT_FWD = _forward(CRAFT_ITEMS)
_DEPOSIT_FWD = _forward(DEPOSIT_ITEMS)

PLACE_ACTION_TO_ITEM = jnp.array(_PLACE_FWD, dtype=jnp.int32)
CRAFT_ACTION_TO_ITEM = jnp.array(_CRAFT_FWD, dtype=jnp.int32)
DEPOSIT_ACTION_TO_ITEM = jnp.array(_DEPOSIT_FWD, dtype=jnp.int32)

# Inverse: item -> absolute action. Indexed by ``ItemType``; ``NO_ACTION``
# where the item has no action in the family.
_PLACE_INV = _inverse(PLACEMENT_ITEMS, "PLACE")
_CRAFT_INV = _inverse(CRAFT_ITEMS, "CRAFT")
_DEPOSIT_INV = _inverse(DEPOSIT_ITEMS, "DEPOSIT")

ITEM_TO_PLACE_ACTION = jnp.array(_PLACE_INV, dtype=jnp.int32)
ITEM_TO_CRAFT_ACTION = jnp.array(_CRAFT_INV, dtype=jnp.int32)
ITEM_TO_DEPOSIT_ACTION = jnp.array(_DEPOSIT_INV, dtype=jnp.int32)


def _validate() -> None:
    """Assert forward and inverse are mutual inverses on each family domain."""
    for fwd, inv, base, prefix in (
        (_PLACE_FWD, _PLACE_INV, PLACE_BASE, "PLACE"),
        (_CRAFT_FWD, _CRAFT_INV, CRAFT_BASE, "CRAFT"),
        (_DEPOSIT_FWD, _DEPOSIT_INV, DEPOSIT_BASE, "DEPOSIT"),
    ):
        for offset, item in enumerate(fwd):
            if inv[item] != base + offset:
                raise ValueError(
                    f"{prefix} family is not a bijection: offset {offset} maps "
                    f"to item {item}, but that item maps back to action "
                    f"{inv[item]} (expected {base + offset})."
                )
        for item, action in enumerate(inv):
            if action == NO_ACTION:
                continue
            if fwd[action - base] != item:
                raise ValueError(
                    f"{prefix} inverse for item {item} -> action {action} does "
                    f"not round-trip (forward gives {fwd[action - base]})."
                )


_validate()
