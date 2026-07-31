"""Tests for :class:`RecipeBalance` overlay + ``with_balance`` apply.

These tests cover the construction-time overlay machinery: the
sparse override container, the per-recipe override application via
:meth:`RecipeBook.with_balance`, and the end-to-end projection
through :meth:`RecipeTable.from_book` so a tuned overlay reaches
:class:`~factoriax.engine.state.EnvParams.recipe_table` cleanly.
"""

from __future__ import annotations

import pytest

from factoriax.engine.constants import ItemType
from factoriax.engine.recipes import (
    BASE_RECIPE_BOOK,
    BASE_RECIPES,
    RecipeBalance,
    RecipeBook,
    RecipeOverride,
    RecipeTable,
)


def _find_base(output: int):
    """Locate the BASE_RECIPES entry with the given output."""
    for r in BASE_RECIPES:
        if r.output == output:
            return r
    raise AssertionError(f"No BASE_RECIPES entry produces {output}")


# ---------------------------------------------------------------------------
# RecipeOverride
# ---------------------------------------------------------------------------


class TestRecipeOverride:
    def test_default_all_none(self) -> None:
        """A bare ``RecipeOverride()`` keeps every field as None. This is the
        identity overlay. Useful as a placeholder when only one of the
        three balance fields needs tuning.
        """
        ov = RecipeOverride()
        assert ov.input_counts is None
        assert ov.output_count is None
        assert ov.ticks is None

    def test_partial_set(self) -> None:
        """Setting one field leaves the others as None."""
        ov = RecipeOverride(ticks=3)
        assert ov.ticks == 3
        assert ov.input_counts is None
        assert ov.output_count is None


# ---------------------------------------------------------------------------
# RecipeBalance: duplicate-override and lookup
# ---------------------------------------------------------------------------


class TestRecipeBalance:
    def test_empty_balance_lookup_returns_none(self) -> None:
        """An empty balance never has any override."""
        b = RecipeBalance()
        assert b.get(int(ItemType.IRON_PLATE)) is None

    def test_lookup_returns_override(self) -> None:
        ov = RecipeOverride(ticks=5)
        b = RecipeBalance(overrides=((int(ItemType.IRON_PLATE), ov),))
        assert b.get(int(ItemType.IRON_PLATE)) is ov

    def test_lookup_missing_returns_none(self) -> None:
        b = RecipeBalance(
            overrides=((int(ItemType.IRON_PLATE), RecipeOverride(ticks=5)),)
        )
        assert b.get(int(ItemType.COPPER_PLATE)) is None

    def test_duplicate_override_raises(self) -> None:
        """Two overrides for the same output must be rejected. A silent
        lookup-order dependency is exactly the bug the unique-output
        invariant on RecipeBook also guards against.
        """
        with pytest.raises(ValueError, match="duplicate override"):
            RecipeBalance(
                overrides=(
                    (int(ItemType.IRON_PLATE), RecipeOverride(ticks=5)),
                    (int(ItemType.IRON_PLATE), RecipeOverride(ticks=10)),
                )
            )


# ---------------------------------------------------------------------------
# RecipeBook.with_balance: overlay application
# ---------------------------------------------------------------------------


class TestWithBalance:
    def test_empty_balance_returns_same_book(self) -> None:
        """No overrides → identical book (object identity preserved)."""
        b = BASE_RECIPE_BOOK.with_balance(RecipeBalance())
        assert b is BASE_RECIPE_BOOK

    def test_override_ticks(self) -> None:
        """Tick override changes only the targeted recipe's ticks."""
        balance = RecipeBalance(
            overrides=((int(ItemType.IRON_PLATE), RecipeOverride(ticks=10)),)
        )
        new_book = BASE_RECIPE_BOOK.with_balance(balance)

        for old, new in zip(BASE_RECIPE_BOOK.recipes, new_book.recipes):
            if new.output == int(ItemType.IRON_PLATE):
                assert new.ticks == 10
                assert new.inputs == old.inputs
                assert new.output_count == old.output_count
            else:
                assert new == old

    def test_override_output_count(self) -> None:
        balance = RecipeBalance(
            overrides=((int(ItemType.WIRE), RecipeOverride(output_count=3)),)
        )
        new_book = BASE_RECIPE_BOOK.with_balance(balance)
        wire = next(r for r in new_book.recipes if r.output == int(ItemType.WIRE))
        base_wire = _find_base(int(ItemType.WIRE))
        assert wire.output_count == 3
        assert wire.inputs == base_wire.inputs
        assert wire.ticks == base_wire.ticks

    def test_override_input_counts(self) -> None:
        """Input-count override rewrites the count tuples but keeps
        the input *item types* identical. Identity is not tunable.
        """
        balance = RecipeBalance(
            overrides=((int(ItemType.IRON_PLATE), RecipeOverride(input_counts=(2, 3))),)
        )
        new_book = BASE_RECIPE_BOOK.with_balance(balance)
        plate = next(
            r for r in new_book.recipes if r.output == int(ItemType.IRON_PLATE)
        )
        base_plate = _find_base(int(ItemType.IRON_PLATE))
        assert plate.inputs[0][0] == base_plate.inputs[0][0]  # item types preserved
        assert plate.inputs[1][0] == base_plate.inputs[1][0]
        assert plate.inputs[0][1] == 2
        assert plate.inputs[1][1] == 3

    def test_override_all_fields(self) -> None:
        balance = RecipeBalance(
            overrides=(
                (
                    int(ItemType.IRON_PLATE),
                    RecipeOverride(input_counts=(2, 1), output_count=4, ticks=1),
                ),
            )
        )
        new_book = BASE_RECIPE_BOOK.with_balance(balance)
        plate = next(
            r for r in new_book.recipes if r.output == int(ItemType.IRON_PLATE)
        )
        assert plate.inputs[0][1] == 2
        assert plate.inputs[1][1] == 1
        assert plate.output_count == 4
        assert plate.ticks == 1

    def test_unknown_output_silently_ignored(self) -> None:
        """An override for an output that the book does not produce passes
        through silently. The user can write a generous balance file
        that targets recipes from a future codebase version without
        breaking on the current one.
        """
        balance = RecipeBalance(
            overrides=((9999, RecipeOverride(ticks=5)),)  # nonexistent ItemType
        )
        new_book = BASE_RECIPE_BOOK.with_balance(balance)
        assert new_book.recipes == BASE_RECIPE_BOOK.recipes

    def test_input_counts_wrong_length_raises(self) -> None:
        balance = RecipeBalance(
            overrides=((int(ItemType.IRON_PLATE), RecipeOverride(input_counts=(2,))),)
        )
        with pytest.raises(ValueError, match="input_counts of length"):
            BASE_RECIPE_BOOK.with_balance(balance)

    def test_negative_input_count_raises(self) -> None:
        balance = RecipeBalance(
            overrides=(
                (int(ItemType.IRON_PLATE), RecipeOverride(input_counts=(2, -1))),
            )
        )
        with pytest.raises(ValueError, match="negative input count"):
            BASE_RECIPE_BOOK.with_balance(balance)

    def test_negative_output_count_raises(self) -> None:
        balance = RecipeBalance(
            overrides=((int(ItemType.IRON_PLATE), RecipeOverride(output_count=-1)),)
        )
        with pytest.raises(ValueError, match="negative output_count"):
            BASE_RECIPE_BOOK.with_balance(balance)

    def test_negative_ticks_raises(self) -> None:
        balance = RecipeBalance(
            overrides=((int(ItemType.IRON_PLATE), RecipeOverride(ticks=-1)),)
        )
        with pytest.raises(ValueError, match="negative ticks"):
            BASE_RECIPE_BOOK.with_balance(balance)

    def test_overlay_returned_book_revalidates(self) -> None:
        """The new book runs the standard ``__post_init__`` checks.
        Overlays cannot introduce duplicate outputs or input pairs
        (they only touch counts/ticks), so the result is always
        a valid :class:`RecipeBook`.
        """
        balance = RecipeBalance(
            overrides=(
                (int(ItemType.IRON_PLATE), RecipeOverride(ticks=99)),
                (int(ItemType.COPPER_PLATE), RecipeOverride(output_count=5)),
            )
        )
        new_book = BASE_RECIPE_BOOK.with_balance(balance)
        assert isinstance(new_book, RecipeBook)
        # Original outputs and machine assignments are intact.
        for old, new in zip(BASE_RECIPE_BOOK.recipes, new_book.recipes):
            assert old.output == new.output
            assert old.machine_type == new.machine_type


# ---------------------------------------------------------------------------
# End-to-end: balance flows through to RecipeTable
# ---------------------------------------------------------------------------


class TestBalanceProjectsToTable:
    def test_ticks_override_visible_in_table(self) -> None:
        balance = RecipeBalance(
            overrides=((int(ItemType.IRON_PLATE), RecipeOverride(ticks=42)),)
        )
        new_book = BASE_RECIPE_BOOK.with_balance(balance)
        table = RecipeTable.from_book(new_book)
        # Find the recipe index for IRON_PLATE in the new book.
        for idx, r in enumerate(new_book.recipes):
            if r.output == int(ItemType.IRON_PLATE):
                assert int(table.ticks[idx]) == 42
                break
        else:
            raise AssertionError("IRON_PLATE not in book")

    def test_output_count_override_visible_in_table(self) -> None:
        balance = RecipeBalance(
            overrides=((int(ItemType.WIRE), RecipeOverride(output_count=7)),)
        )
        new_book = BASE_RECIPE_BOOK.with_balance(balance)
        table = RecipeTable.from_book(new_book)
        for idx, r in enumerate(new_book.recipes):
            if r.output == int(ItemType.WIRE):
                assert int(table.output_counts[idx]) == 7
                break
        else:
            raise AssertionError("WIRE not in book")

    def test_input_counts_override_visible_in_table(self) -> None:
        balance = RecipeBalance(
            overrides=((int(ItemType.IRON_PLATE), RecipeOverride(input_counts=(5, 2))),)
        )
        new_book = BASE_RECIPE_BOOK.with_balance(balance)
        table = RecipeTable.from_book(new_book)
        for idx, r in enumerate(new_book.recipes):
            if r.output == int(ItemType.IRON_PLATE):
                assert int(table.input_counts[idx, 0]) == 5
                assert int(table.input_counts[idx, 1]) == 2
                break
        else:
            raise AssertionError("IRON_PLATE not in book")
