"""Tests for the :class:`RecipeBook` construction rules.

A book validates once, at construction. The engine's Phase 3 match reads the
unordered input set of each recipe, so two recipes for the same machine may
not share one. A recipe that names the same input twice, or that has an arity
the engine cannot hold, is refused here rather than at step time.
"""

from __future__ import annotations

import pytest

from factoriax.engine.constants import ItemType
from factoriax.engine.recipes import Recipe, RecipeBook


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

    A machine has exactly two input slots in ``EnvState``, so a wider recipe
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

    The reverse-lookup ``OUTPUT_TO_RECIPE`` is single-valued. A
    duplicate output overwrites the earlier mapping in silence
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
    type-set at the same arity must be rejected. The Phase 3
    forward-match is ambiguous.
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
    check that blocks a legitimate FURNACE/ASSEMBLER overlap.
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
