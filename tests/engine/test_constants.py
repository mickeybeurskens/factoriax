"""Tests for the enum composition in :mod:`factoriax.engine.constants`.

Two groups live here.

The item group covers ``Resource``, ``HalfFabricate``, and ``Machine``, the
three category enums that ``ItemType`` composes from. Their real members must
partition the non-EMPTY ``ItemType`` members exactly, because a new member in
an earlier category shifts the value of every later item.

The action group covers ``MoveAction`` and ``InteractAction``, the two fixed
action categories. The flat ``Action`` enum composes from these plus the
parametric families, so the two must cover the non-parametric actions and
stay disjoint.
"""

from __future__ import annotations

from enum import IntEnum

from factoriax.engine.constants import (
    CRAFT_BASE,
    CRAFT_ITEMS,
    DEPOSIT_BASE,
    DEPOSIT_ITEMS,
    ITEM_TO_MACHINE,
    NUM_ACTIONS,
    NUM_ITEM_TYPES,
    PLACE_BASE,
    PLACEMENT_ITEMS,
    Action,
    BlockType,
    HalfFabricate,
    InteractAction,
    ItemType,
    Machine,
    MoveAction,
    Resource,
)
from factoriax.engine.tables import SOLID_BLOCKS

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


def _real_names(category: type[IntEnum]) -> set[str]:
    """Member names of a category excluding the ``NONE`` placeholder."""
    return {m.name for m in category if m.name != "NONE"}


def test_categories_carry_a_none_zero_placeholder() -> None:
    """Each category enum has ``NONE == 0`` (the uniform empty-slot shape)."""
    for category in (Resource, HalfFabricate, Machine):
        assert int(category["NONE"]) == 0


def test_categories_partition_non_empty_items() -> None:
    """The categories' real names cover every non-EMPTY item once."""
    resource = _real_names(Resource)
    half_fab = _real_names(HalfFabricate)
    machine = _real_names(Machine)

    # No name appears in two categories.
    assert resource.isdisjoint(half_fab)
    assert resource.isdisjoint(machine)
    assert half_fab.isdisjoint(machine)

    # Together they are exactly the non-EMPTY items.
    covered = resource | half_fab | machine
    non_empty = {m.name for m in ItemType if m != ItemType.EMPTY}
    assert covered == non_empty


def test_every_category_name_is_a_real_item() -> None:
    """Each real category member name resolves to an ``ItemType`` member."""
    for enum_cls in (Resource, HalfFabricate, Machine):
        for member in enum_cls:
            if member.name == "NONE":
                continue
            assert member.name in ItemType.__members__


def test_machine_category_matches_item_to_machine_keys() -> None:
    """The ``Machine`` category (sans NONE) is the placeable mapping's keys."""
    machine_names = _real_names(Machine)
    mapping_names = {it.name for it in ITEM_TO_MACHINE}
    assert machine_names == mapping_names


# -------------------------------------------------------------------------
# Enum sizes the whole engine depends on
# -------------------------------------------------------------------------


class TestConstants:
    """Tests for constants module."""

    def test_resource_blocks_are_walkable(self) -> None:
        """Resource blocks are not in SOLID_BLOCKS."""
        solid_set = set(int(b) for b in SOLID_BLOCKS)
        assert int(BlockType.IRON) not in solid_set
        assert int(BlockType.COPPER) not in solid_set
        assert int(BlockType.COAL) not in solid_set
