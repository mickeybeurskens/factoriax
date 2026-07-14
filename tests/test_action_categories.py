"""Tests for the action category enums (DEF2 source of truth).

``MoveAction`` and ``InteractAction`` are the two *fixed* (non-parametric)
action categories. DEF2 composes the flat ``Action`` enum from these plus
the parametric families generated from the item categories (Placement per
Machine, Craft per non-resource item, Deposit per item). This locks the
invariant for the fixed half: the two category enums cover exactly the
non-parametric actions today, disjointly.
"""

from __future__ import annotations

from factoriax.engine.constants import (
    CRAFT_BASE,
    CRAFT_ITEMS,
    DEPOSIT_BASE,
    DEPOSIT_ITEMS,
    NUM_ACTIONS,
    NUM_ITEM_TYPES,
    PLACE_BASE,
    PLACEMENT_ITEMS,
    Action,
    HalfFabricate,
    InteractAction,
    ItemType,
    Machine,
    MoveAction,
    Resource,
)

# Prefixes of the parametric action families on the current Action enum.
_PARAMETRIC_PREFIXES = ("PLACE_", "CRAFT_", "DEPOSIT_")


def test_fixed_categories_cover_non_parametric_actions() -> None:
    """``MoveAction`` + ``InteractAction`` == today's non-parametric actions."""
    fixed = {a.name for a in Action if not a.name.startswith(_PARAMETRIC_PREFIXES)}
    move = {a.name for a in MoveAction}
    interact = {a.name for a in InteractAction}

    assert move.isdisjoint(interact)
    assert (move | interact) == fixed


def test_every_category_name_is_a_real_action() -> None:
    """Each fixed-category member name resolves to an ``Action`` member."""
    for enum_cls in (MoveAction, InteractAction):
        for member in enum_cls:
            assert member.name in Action.__members__


def test_action_space_shape() -> None:
    """The composed action layout has the expected flat size.

    DEF2 D1b completes the two families hand-numbering left short (deposit
    gains LIMESTONE, craft gains the five rocket parts), so the action count
    grows 79 -> 85. The tier-3 science pack (TIER3_SCIENCE_PACK) adds one
    item and its CRAFT_/DEPOSIT_ actions: 85 -> 87, 33 -> 34.
    """
    assert NUM_ACTIONS == 87
    assert NUM_ITEM_TYPES == 34


def test_parametric_family_sizes_match_item_categories() -> None:
    """Each parametric family has exactly one action per item it addresses."""
    place = [a for a in Action if a.name.startswith("PLACE_")]
    craft = [a for a in Action if a.name.startswith("CRAFT_")]
    deposit = [a for a in Action if a.name.startswith("DEPOSIT_")]

    real_machines = [m for m in Machine if m.name != "NONE"]
    assert len(place) == len(PLACEMENT_ITEMS) == len(real_machines)
    assert len(craft) == len(CRAFT_ITEMS)
    assert len(deposit) == len(DEPOSIT_ITEMS)
    # Deposit is total over the non-EMPTY items.
    assert len(deposit) == NUM_ITEM_TYPES - 1


def test_family_bases_partition_the_parametric_range() -> None:
    """The derived ``*_BASE`` offsets tile the action space contiguously."""
    fixed = len(MoveAction) + len(InteractAction)
    assert PLACE_BASE == fixed
    assert CRAFT_BASE == PLACE_BASE + len(PLACEMENT_ITEMS)
    assert DEPOSIT_BASE == CRAFT_BASE + len(CRAFT_ITEMS)
    assert DEPOSIT_BASE + len(DEPOSIT_ITEMS) == NUM_ACTIONS


def test_craftable_iff_non_resource_iff_recipe_outputs() -> None:
    """The Craft family addresses exactly the non-resource items, which are
    exactly the recipe outputs -- the DEF2 design invariant. A family can no
    longer fall short of (or overshoot) the set of items that actually have a
    recipe, the way hand-numbering left craft missing the rocket parts.
    """
    from factoriax.engine.recipes import BASE_RECIPES

    craft_items = {ItemType[m.name] for m in CRAFT_ITEMS}
    non_resource = {
        ItemType[m.name] for m in (*HalfFabricate, *Machine) if m.name != "NONE"
    }
    recipe_outputs = {ItemType(r.output) for r in BASE_RECIPES}
    resources = {ItemType[m.name] for m in Resource if m.name != "NONE"}

    assert craft_items == non_resource
    assert craft_items == recipe_outputs
    assert craft_items.isdisjoint(resources)
