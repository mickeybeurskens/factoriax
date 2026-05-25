"""Tests for the item category enums (DEF2 source of truth).

``Resource``, ``HalfFabricate``, and ``Machine`` are the three category
enums that DEF2 composes ``ItemType`` from. These lock the invariant that
makes that composition sound: their member *names* partition the non-EMPTY
``ItemType`` members exactly -- complete, no overlap, no ``EMPTY`` -- and
the ``Machine`` category agrees with the ``ITEM_TO_MACHINE`` bijection.
"""

from __future__ import annotations

from factoriax.constants import (
    ITEM_TO_MACHINE,
    HalfFabricate,
    ItemType,
    Machine,
    Resource,
)


def test_categories_partition_non_empty_items() -> None:
    """The three category enums' names cover every non-EMPTY item once."""
    resource = {m.name for m in Resource}
    half_fab = {m.name for m in HalfFabricate}
    machine = {m.name for m in Machine}

    # No name appears in two categories.
    assert resource.isdisjoint(half_fab)
    assert resource.isdisjoint(machine)
    assert half_fab.isdisjoint(machine)

    # Together they are exactly the non-EMPTY items.
    covered = resource | half_fab | machine
    non_empty = {m.name for m in ItemType if m != ItemType.EMPTY}
    assert covered == non_empty


def test_every_category_name_is_a_real_item() -> None:
    """Each category member name resolves to an ``ItemType`` member."""
    for enum_cls in (Resource, HalfFabricate, Machine):
        for member in enum_cls:
            assert member.name in ItemType.__members__


def test_machine_category_matches_item_to_machine_keys() -> None:
    """The ``Machine`` category is exactly the placeable mapping's keys."""
    machine_names = {m.name for m in Machine}
    mapping_names = {it.name for it in ITEM_TO_MACHINE}
    assert machine_names == mapping_names
