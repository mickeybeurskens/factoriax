"""Tests for the item<->action mappings (DEF2 D1c).

The forward (action offset -> item) and inverse (item -> absolute action)
tables in :mod:`factoriax.engine.actions` are both derived from the category family
lists, and must be mutual inverses on each family's domain. These are
JIT-free: the jnp tables are converted to numpy and inspected directly.
"""

from __future__ import annotations

import numpy as np

from factoriax.engine import actions
from factoriax.engine.constants import (
    CRAFT_BASE,
    CRAFT_ITEMS,
    DEPOSIT_BASE,
    DEPOSIT_ITEMS,
    NUM_ITEM_TYPES,
    PLACE_BASE,
    PLACEMENT_ITEMS,
    Action,
    HalfFabricate,
    ItemType,
    Machine,
    Resource,
)

# (forward, inverse, base, family items, Action name prefix) per family.
_FAMILIES = (
    (
        actions.PLACE_ACTION_TO_ITEM,
        actions.ITEM_TO_PLACE_ACTION,
        PLACE_BASE,
        PLACEMENT_ITEMS,
        "PLACE",
    ),
    (
        actions.CRAFT_ACTION_TO_ITEM,
        actions.ITEM_TO_CRAFT_ACTION,
        CRAFT_BASE,
        CRAFT_ITEMS,
        "CRAFT",
    ),
    (
        actions.DEPOSIT_ACTION_TO_ITEM,
        actions.ITEM_TO_DEPOSIT_ACTION,
        DEPOSIT_BASE,
        DEPOSIT_ITEMS,
        "DEPOSIT",
    ),
)


def test_forward_inverse_round_trip() -> None:
    """Forward and inverse are mutual inverses on each family domain."""
    for fwd, inv, base, items, _ in _FAMILIES:
        fwd = np.asarray(fwd)
        inv = np.asarray(inv)
        assert len(fwd) == len(items)
        # offset -> item -> action -> same offset
        for offset, item in enumerate(fwd):
            assert inv[item] == base + offset
        # every real inverse entry maps back to its item via forward
        for item, action in enumerate(inv):
            if action == actions.NO_ACTION:
                continue
            assert fwd[action - base] == item


def test_forward_matches_action_enum() -> None:
    """Each family offset resolves to the item named by its Action member."""
    for fwd, _, base, items, prefix in _FAMILIES:
        fwd = np.asarray(fwd)
        for offset, member in enumerate(items):
            assert int(fwd[offset]) == int(ItemType[member.name])
            assert int(Action[f"{prefix}_{member.name}"]) == base + offset


def test_inverse_domains() -> None:
    """Place == machines, craft == non-resources, deposit == all non-EMPTY."""
    inv_place = np.asarray(actions.ITEM_TO_PLACE_ACTION)
    inv_craft = np.asarray(actions.ITEM_TO_CRAFT_ACTION)
    inv_deposit = np.asarray(actions.ITEM_TO_DEPOSIT_ACTION)

    placeable = {int(ItemType[m.name]) for m in Machine if m.name != "NONE"}
    craftable = {
        int(ItemType[m.name]) for m in (*HalfFabricate, *Machine) if m.name != "NONE"
    }
    depositable = set(range(1, NUM_ITEM_TYPES))  # every non-EMPTY item

    def domain(inv: np.ndarray) -> set[int]:
        return {i for i, a in enumerate(inv) if a != actions.NO_ACTION}

    assert domain(inv_place) == placeable
    assert domain(inv_craft) == craftable
    assert domain(inv_deposit) == depositable
    # EMPTY belongs to no family.
    assert inv_place[int(ItemType.EMPTY)] == actions.NO_ACTION
    assert inv_deposit[int(ItemType.EMPTY)] == actions.NO_ACTION


def test_resources_are_not_craftable_or_placeable() -> None:
    """Raw resources have no craft/place action (the NO_ACTION sentinel)."""
    inv_place = np.asarray(actions.ITEM_TO_PLACE_ACTION)
    inv_craft = np.asarray(actions.ITEM_TO_CRAFT_ACTION)
    inv_deposit = np.asarray(actions.ITEM_TO_DEPOSIT_ACTION)
    for resource in Resource:
        if resource.name == "NONE":
            continue
        item = int(ItemType[resource.name])
        assert inv_craft[item] == actions.NO_ACTION
        assert inv_place[item] == actions.NO_ACTION
        # ...but resources are still depositable.
        assert inv_deposit[item] != actions.NO_ACTION
