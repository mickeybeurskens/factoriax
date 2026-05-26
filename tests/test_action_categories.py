"""Tests for the action category enums (DEF2 source of truth).

``MoveAction`` and ``InteractAction`` are the two *fixed* (non-parametric)
action categories. DEF2 composes the flat ``Action`` enum from these plus
the parametric families generated from the item categories (Placement per
Machine, Craft per non-resource item, Deposit per item). This locks the
invariant for the fixed half: the two category enums cover exactly the
non-parametric actions today, disjointly.
"""

from __future__ import annotations

from factoriax.constants import Action, InteractAction, MoveAction

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
