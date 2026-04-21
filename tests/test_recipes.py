"""Structural tests for the unified recipe table.

The engine's Phase 3 recipe match relies on every recipe having a
distinct (unordered) set of input item types within its machine
type. If two recipes share an input type-set (or one's is a subset
of another's), depositing the larger recipe's inputs can trigger
the smaller recipe mid-deposit, producing the wrong output.
"""

from __future__ import annotations

import pytest

from factoriax.recipes import RECIPE_MACHINE_TYPE, RECIPES


def _input_type_set(recipe: dict) -> frozenset[int]:
    """Return the unordered set of input item types for a recipe."""
    return frozenset(int(it) for it, _ in recipe["inputs"])


def test_every_recipe_has_unique_input_type_set() -> None:
    """No two recipes on the same machine share an input type-set.

    Recipe matching compares input types by set equality; two
    recipes with the same input types on the same machine would be
    ambiguous.
    """
    seen: dict[tuple[int, frozenset[int]], int] = {}
    for idx, recipe in enumerate(RECIPES):
        key = (int(RECIPE_MACHINE_TYPE[idx]), _input_type_set(recipe))
        assert key not in seen, (
            f"Recipe {idx} ({recipe['output']}) shares input type-set "
            f"{set(key[1])} with recipe {seen[key]} on the same machine type "
            f"{key[0]} — one will always win over the other in Phase 3."
        )
        seen[key] = idx


def test_no_recipe_input_type_set_is_subset_of_another() -> None:
    """Subset relationships between recipe input type-sets are forbidden.

    If recipe A's inputs are ``{X}`` and recipe B's inputs are
    ``{X, Y}``, depositing B's inputs one at a time would trigger A
    on the first deposit (since the machine sees a satisfied A
    before Y arrives). Keeping all input type-sets pairwise-disjoint
    avoids this hazard and keeps per-deposit ordering irrelevant.
    """
    for i, ri in enumerate(RECIPES):
        si = _input_type_set(ri)
        mti = int(RECIPE_MACHINE_TYPE[i])
        for j, rj in enumerate(RECIPES):
            if i == j:
                continue
            if int(RECIPE_MACHINE_TYPE[j]) != mti:
                continue
            sj = _input_type_set(rj)
            assert not si < sj, (
                f"Recipe {i} ({ri['output']}) inputs {set(si)} are a "
                f"strict subset of recipe {j} ({rj['output']}) inputs "
                f"{set(sj)} on machine type {mti}."
            )


@pytest.mark.parametrize("recipe", RECIPES)
def test_recipe_has_exactly_two_inputs(recipe: dict) -> None:
    """Engine Phase 3 assumes every recipe has exactly 2 input slots."""
    assert len(recipe["inputs"]) == 2
