"""Tests for the item category enums (DEF2/DEF3 source of truth).

``Resource``, ``HalfFabricate``, and ``Machine`` are the three category
enums that DEF2 composes ``ItemType`` from. Each carries a ``NONE = 0``
placeholder (the uniform "NONE=0, real members from 1" shape DEF3 introduced
so ``Machine`` can double as the entity tag); ``NONE`` is stripped from every
derivative. These tests lock the invariant that makes the composition sound:
the *real* (non-NONE) member names partition the non-EMPTY ``ItemType``
members exactly, and the ``Machine`` category agrees with ``ITEM_TO_MACHINE``.
"""

from __future__ import annotations

from enum import IntEnum

from factoriax.engine.constants import (
    ITEM_TO_MACHINE,
    HalfFabricate,
    ItemType,
    Machine,
    Resource,
)


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
