"""Validation + projection tests for :class:`RecipeBook` / :class:`RecipeTable`.

These tests cover the structural invariants enforced at book
construction time and the array shapes produced by
:meth:`RecipeTable.from_book`. The legacy ``test_recipes.py`` covers
the same uniqueness rule on the *shipped* :data:`BASE_RECIPES`; this
file covers the rule mechanism itself with synthetic books that
deliberately violate it.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from factoriax.engine.constants import ItemType, Machine
from factoriax.engine.recipes import (
    BASE_RECIPE_BOOK,
    BASE_RECIPES,
    DEFAULT_RECIPE_TABLE,
    NUM_RECIPES,
    Recipe,
    RecipeBook,
    RecipeTable,
)
from factoriax.engine.step import CRAFT_ACTION_TO_ITEM


def _plate_recipe(output: int = int(ItemType.IRON_PLATE)) -> Recipe:
    """A canonical 2-input furnace recipe used as a test scaffold."""
    return Recipe(
        output=output,
        inputs=((int(ItemType.IRON_ORE), 1), (int(ItemType.COAL), 1)),
        ticks=2,
        name="test-plate",
    )


@pytest.mark.parametrize(
    "inputs",
    [
        pytest.param((), id="zero-inputs"),
        pytest.param(
            (
                (int(ItemType.IRON_ORE), 1),
                (int(ItemType.COAL), 1),
                (int(ItemType.TIN_ORE), 1),
            ),
            id="three-inputs",
        ),
    ],
)
def test_arity_outside_engine_limit_raises(
    inputs: tuple[tuple[int, int], ...],
) -> None:
    """A recipe must consume 1 or 2 input types.

    A combiner has exactly two input slots in ``EnvState``, so a wider recipe
    has nowhere to put its third input. Without this check the projection pads
    to the widest recipe and the engine reads only the first two slots, which
    drops the extra ingredient without reporting anything.
    """
    wide = Recipe(
        output=int(ItemType.IRON_PLATE), inputs=inputs, ticks=2, name="bad-arity"
    )
    with pytest.raises(ValueError, match="input types; a recipe must have"):
        RecipeBook(recipes=(wide,))


def test_repeated_input_item_raises() -> None:
    """A recipe must not name the same input item in both slots.

    ``can_afford_recipe`` checks each slot on its own, so a repeat reads as
    affordable while the player holds enough for one slot. ``craft_recipe``
    then subtracts both slots and the inventory goes below zero. Rejecting
    the book is what keeps that unreachable.
    """
    repeated = Recipe(
        output=int(ItemType.WIRE),
        inputs=((int(ItemType.IRON_PLATE), 1), (int(ItemType.IRON_PLATE), 1)),
        ticks=1,
        name="repeated-input",
    )
    with pytest.raises(ValueError, match="names IRON_PLATE twice"):
        RecipeBook(recipes=(repeated,))


def test_unique_output_constraint_raises() -> None:
    """Two recipes producing the same ItemType must be rejected.

    The reverse-lookup ``OUTPUT_TO_RECIPE`` is single-valued; a
    duplicate output would silently overwrite the earlier mapping
    and break crafting yield + cycle deposit logic.
    """
    duplicate = (
        _plate_recipe(),
        Recipe(
            output=int(ItemType.IRON_PLATE),  # same output as first
            inputs=((int(ItemType.TIN_ORE), 1), (int(ItemType.COAL), 1)),
            ticks=2,
            name="duplicate-iron-plate",
        ),
    )
    with pytest.raises(ValueError, match="Duplicate recipe output"):
        RecipeBook(recipes=duplicate)


def test_unique_input_pair_constraint_raises() -> None:
    """Two recipes on the same machine sharing an unordered input
    type-set at the same arity must be rejected — Phase 3
    forward-match would be ambiguous.
    """
    same_machine_same_inputs = (
        Recipe(
            output=int(ItemType.WIRE),
            inputs=((int(ItemType.COPPER_PLATE), 1), (int(ItemType.TIN_PLATE), 1)),
            ticks=4,
            name="wire",
        ),
        Recipe(
            # Different output, but identical (unordered) input set on
            # the same ASSEMBLER machine type.
            output=int(ItemType.ARM),
            inputs=((int(ItemType.TIN_PLATE), 1), (int(ItemType.COPPER_PLATE), 1)),
            ticks=4,
            name="arm-conflict",
        ),
    )
    with pytest.raises(ValueError, match="both consume the input set"):
        RecipeBook(recipes=same_machine_same_inputs)


def test_same_inputs_different_machine_allowed() -> None:
    """The matcher partitions by machine type, so two recipes with
    the same input pair on *different* machines are unambiguous and
    must be allowed. This guards against an over-broad uniqueness
    check that would block a legitimate FURNACE/ASSEMBLER overlap.
    """
    cross_machine = (
        # Furnace recipe: IRON_ORE + COAL -> IRON_PLATE
        Recipe(
            output=int(ItemType.IRON_PLATE),
            inputs=((int(ItemType.IRON_ORE), 1), (int(ItemType.COAL), 1)),
            ticks=2,
            name="iron-plate",
        ),
        # Hypothetical assembler recipe with the same input type-set
        # but on the ASSEMBLER side. Not a real recipe, but the book
        # must accept it because the matcher gates on machine type.
        Recipe(
            output=int(ItemType.WIRE),
            inputs=((int(ItemType.IRON_ORE), 1), (int(ItemType.COAL), 1)),
            ticks=4,
            name="cross-machine-wire",
        ),
    )
    book = RecipeBook(recipes=cross_machine)
    assert len(book.recipes) == 2


def test_recipe_table_from_book_matches_defaults() -> None:
    """:meth:`RecipeTable.from_book` on :data:`BASE_RECIPE_BOOK`
    reproduces the module-level default arrays exactly. This is the
    contract every Step 4 call site relies on.
    """
    projected = RecipeTable.from_book(BASE_RECIPE_BOOK)

    for field_name in (
        "outputs",
        "output_counts",
        "input_items",
        "input_counts",
        "ticks",
        "machine_type",
        "output_to_recipe",
    ):
        a = getattr(projected, field_name)
        b = getattr(DEFAULT_RECIPE_TABLE, field_name)
        assert a.shape == b.shape, (
            f"{field_name} shape mismatch: {a.shape} vs {b.shape}"
        )
        assert jnp.array_equal(a, b), f"{field_name} values differ"


def test_recipe_table_shapes_match_book_size() -> None:
    """Projected table shapes derive from the book size + max arity
    — load-bearing for downstream JIT shape stability.
    """
    table = DEFAULT_RECIPE_TABLE
    assert table.outputs.shape == (NUM_RECIPES,)
    assert table.output_counts.shape == (NUM_RECIPES,)
    assert table.ticks.shape == (NUM_RECIPES,)
    assert table.machine_type.shape == (NUM_RECIPES,)
    max_inputs = max(len(r.inputs) for r in BASE_RECIPES)
    assert table.input_items.shape == (NUM_RECIPES, max_inputs)
    assert table.input_counts.shape == (NUM_RECIPES, max_inputs)
    assert table.output_to_recipe.shape == (len(ItemType),)


def test_output_to_recipe_round_trip() -> None:
    """``output_to_recipe[recipe.output]`` returns the recipe's index
    for every shipped recipe; non-output items map to ``-1``.
    """
    table = DEFAULT_RECIPE_TABLE
    output_set = {r.output for r in BASE_RECIPES}
    for idx, recipe in enumerate(BASE_RECIPES):
        assert int(table.output_to_recipe[recipe.output]) == idx, (
            f"Recipe {idx} ({ItemType(recipe.output).name}) round-trip failed"
        )
    for item in ItemType:
        if int(item) not in output_set:
            assert int(table.output_to_recipe[int(item)]) == -1, (
                f"Non-output item {item.name} must map to -1"
            )


def test_furnace_recipes_assigned_furnace_machine_type() -> None:
    """Sanity: every plate / wafer / refractory in the projected
    machine_type array is FURNACE; every other recipe is ASSEMBLER.
    """
    table = DEFAULT_RECIPE_TABLE
    furnace_outputs = {
        int(ItemType.IRON_PLATE),
        int(ItemType.COPPER_PLATE),
        int(ItemType.TIN_PLATE),
        int(ItemType.WAFER),
        int(ItemType.REFRACTORY),
    }
    for idx, recipe in enumerate(BASE_RECIPES):
        expected = (
            int(Machine.FURNACE)
            if recipe.output in furnace_outputs
            else int(Machine.ASSEMBLER)
        )
        assert int(table.machine_type[idx]) == expected, (
            f"Recipe {idx} ({ItemType(recipe.output).name}) machine_type "
            f"mismatch: got {int(table.machine_type[idx])}, expected {expected}"
        )


def _two_tier_book() -> RecipeBook:
    """A 4-recipe, 2-tier book whose row order deliberately differs from
    the CRAFT_* action order.

    Tier 1 smelts ore into plates; tier 2 assembles plates into parts.
    Rows are scrambled so a positional ``action - CRAFT_BASE`` dispatch
    would resolve the wrong recipe — only output-keyed resolution gets it
    right. A subset of craftable items, so the rest must resolve to -1.
    """
    iron_plate = Recipe(
        output=int(ItemType.IRON_PLATE),
        inputs=((int(ItemType.IRON_ORE), 1), (int(ItemType.COAL), 1)),
        ticks=2,
        name="iron-plate",
    )
    copper_plate = Recipe(
        output=int(ItemType.COPPER_PLATE),
        inputs=((int(ItemType.COPPER_ORE), 1), (int(ItemType.COAL), 1)),
        ticks=2,
        name="copper-plate",
    )
    wire = Recipe(  # tier 2: consumes a tier-1 plate
        output=int(ItemType.WIRE),
        inputs=((int(ItemType.COPPER_PLATE), 1), (int(ItemType.TIN_ORE), 1)),
        ticks=4,
        name="wire",
    )
    frame = Recipe(  # tier 2: consumes a tier-1 plate
        output=int(ItemType.FRAME),
        inputs=((int(ItemType.IRON_PLATE), 2), (int(ItemType.SILICON), 1)),
        ticks=4,
        name="frame",
    )
    return RecipeBook(recipes=(wire, iron_plate, frame, copper_plate))


def _assert_craft_dispatch_resolves_by_output(table: RecipeTable) -> None:
    """Every CRAFT_* action resolves through ``output_to_recipe`` to a
    recipe whose output is exactly the action's item; items absent from
    the table resolve to -1 (the dispatch no-ops). Pure indexing on the
    table arrays — no env, no step, no JIT."""
    present = {int(output) for output in table.outputs.tolist()}
    for offset, item in enumerate(CRAFT_ACTION_TO_ITEM.tolist()):
        row = int(table.output_to_recipe[item])
        if item in present:
            assert row >= 0, f"offset {offset}: {ItemType(item).name} missing"
            got = ItemType(int(table.outputs[row])).name
            assert int(table.outputs[row]) == item, (
                f"offset {offset}: resolved to {got}, expected {ItemType(item).name}"
            )
        else:
            assert row == -1, (
                f"offset {offset}: absent {ItemType(item).name} should resolve to -1"
            )


def test_craft_dispatch_resolves_by_output_default_book() -> None:
    """Canonical BASE_RECIPES table: every craft action resolves correctly."""
    _assert_craft_dispatch_resolves_by_output(DEFAULT_RECIPE_TABLE)


def test_craft_dispatch_resolves_by_output_alternate_book() -> None:
    """A reordered, subset 2-tier book resolves by output item, not list
    position (engine bug #6 guard)."""
    _assert_craft_dispatch_resolves_by_output(RecipeTable.from_book(_two_tier_book()))
