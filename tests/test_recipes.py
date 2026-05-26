"""Structural tests for the unified recipe table.

The engine's Phase 3 recipe match relies on every recipe having a
distinct (unordered) set of input item types within its machine
type *at the same arity*. 1-input recipes pad their unused slot
with ``(EMPTY, 0)`` in the derived arrays, so they only fire when
the machine's second input slot is physically empty — the slot
gate disambiguates them from any 2-input recipe whose input set
is a superset. Same-arity duplicates would still be ambiguous and
are rejected. As of the LIMESTONE addition every furnace recipe
is 2-input (REFRACTORY = {LIMESTONE, COAL}), but the 1-input
support stays in place for forward compatibility.
"""

from __future__ import annotations

import pytest

from factoriax.engine.constants import Machine
from factoriax.engine.recipes import (
    BASE_RECIPES,
    NUM_RECIPES,
    RECIPE_MACHINE_TYPE,
    RECIPE_OUTPUT_COUNTS,
    Recipe,
)


def _input_type_set(recipe: Recipe) -> frozenset[int]:
    """Return the unordered set of input item types for a recipe."""
    return frozenset(int(it) for it, _ in recipe.inputs)


def test_every_recipe_has_unique_input_type_set() -> None:
    """No two recipes on the same machine share an input type-set
    (at the same arity). Two 2-input recipes with the same pair
    would be ambiguous; two 1-input recipes with the same item
    likewise. Different-arity recipes with overlapping items are
    disambiguated by the slot-emptiness gate in Phase 3.
    """
    seen: dict[tuple[int, int, frozenset[int]], int] = {}
    for idx, recipe in enumerate(BASE_RECIPES):
        key = (
            int(RECIPE_MACHINE_TYPE[idx]),
            len(recipe.inputs),
            _input_type_set(recipe),
        )
        assert key not in seen, (
            f"Recipe {idx} ({recipe.output}) shares input type-set "
            f"{set(key[2])} at arity {key[1]} with recipe {seen[key]} "
            f"on the same machine type {key[0]}."
        )
        seen[key] = idx


@pytest.mark.parametrize("recipe", BASE_RECIPES)
def test_recipe_has_one_or_two_inputs(recipe: Recipe) -> None:
    """Furnace recipes take 1 or 2 inputs; assembler recipes take 2."""
    assert len(recipe.inputs) in (1, 2)


@pytest.mark.parametrize("idx", range(len(BASE_RECIPES)))
def test_assembler_recipes_have_two_inputs(idx: int) -> None:
    """Every assembler-gated recipe has exactly 2 input types.
    Furnace recipes may be 1 or 2 — every shipped furnace recipe
    is currently 2-input with coal as the second (fuel-like) slot
    (IRON_ORE + COAL → IRON_PLATE, …, LIMESTONE + COAL →
    REFRACTORY). The 1-input branch in run_combiners is exercised
    only by the slot-emptiness gate test below.
    """
    recipe = BASE_RECIPES[idx]
    machine = int(RECIPE_MACHINE_TYPE[idx])
    if machine == int(Machine.ASSEMBLER):
        assert len(recipe.inputs) == 2, (
            f"Assembler recipe {idx} ({recipe.output}) has "
            f"{len(recipe.inputs)} inputs; expected 2."
        )


def test_recipe_output_counts_defaults_to_one() -> None:
    """Recipes without an explicit ``output_count`` default to 1 — the
    historical "every craft yields one unit" behaviour. The array
    length must match NUM_RECIPES so the per-cycle yield lookup in
    ``run_assemblers`` indexes safely.
    """
    assert RECIPE_OUTPUT_COUNTS.shape == (NUM_RECIPES,)
    for idx, recipe in enumerate(BASE_RECIPES):
        actual = int(RECIPE_OUTPUT_COUNTS[idx])
        assert actual == recipe.output_count, (
            f"Recipe {idx} ({recipe.output}) output_count: expected "
            f"{recipe.output_count}, got {actual}"
        )


def test_every_furnace_recipe_is_two_input() -> None:
    """After the LIMESTONE addition every furnace recipe takes two
    inputs. This is a load-bearing structural property the rocket
    scenario's belt-logistics layout relies on (no single-input
    outliers that would need a special-case feeder shape).
    """
    for idx, recipe in enumerate(BASE_RECIPES):
        if int(RECIPE_MACHINE_TYPE[idx]) == int(Machine.FURNACE):
            assert len(recipe.inputs) == 2, (
                f"Furnace recipe {idx} ({recipe.output}) has "
                f"{len(recipe.inputs)} inputs; expected 2."
            )
