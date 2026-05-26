"""Unit tests for the tile-pinned ``ProduceInMachineAt`` primitive.

These tests cover ``__init__`` correctness — recipe lookup,
book-aware tuning, factory aliases. The full deposit/wait/withdraw
FSM is exercised by the slow integration test in
``test_advanced_factory_iron_cell_produces_plates`` indirectly
(through Phase 2 of the iterative bootstrap), and by a focused
smoke test below.
"""

from __future__ import annotations

import pytest

from baselines.rocket.scripted.goals import (
    ProduceInAssemblerAt,
    ProduceInFurnaceAt,
    ProduceInMachineAt,
)
from factoriax.engine.constants import ItemType
from factoriax.engine.recipes import (
    BASE_RECIPE_BOOK,
    RecipeBalance,
    RecipeBook,
    RecipeOverride,
)


def _book_with_iron_plate_ticks(ticks: int) -> RecipeBook:
    return BASE_RECIPE_BOOK.with_balance(
        RecipeBalance(
            overrides=((int(ItemType.IRON_PLATE), RecipeOverride(ticks=ticks)),)
        )
    )


class TestProduceInMachineAt:
    def test_default_book_uses_base_ticks(self) -> None:
        goal = ProduceInMachineAt((15, 16), ItemType.IRON_PLATE, 1)
        # IRON_PLATE.ticks=2 + 3 slack = 5
        assert goal.wait_ticks == 5
        assert goal.tile == (15, 16)
        assert goal.output_item == int(ItemType.IRON_PLATE)
        assert goal.count == 1

    def test_tuned_book_overrides_ticks(self) -> None:
        book = _book_with_iron_plate_ticks(7)
        goal = ProduceInMachineAt((15, 16), ItemType.IRON_PLATE, 1, book=book)
        assert goal.wait_ticks == 10  # 7 + 3 slack

    def test_tuned_book_overrides_input_counts(self) -> None:
        book = BASE_RECIPE_BOOK.with_balance(
            RecipeBalance(
                overrides=(
                    (
                        int(ItemType.WIRE),
                        RecipeOverride(input_counts=(3, 2)),
                    ),
                )
            )
        )
        goal = ProduceInMachineAt((17, 16), ItemType.WIRE, 1, book=book)
        counts = {item: qty for item, qty in goal.recipe_inputs}
        assert counts[int(ItemType.COPPER_PLATE)] == 3
        assert counts[int(ItemType.TIN_PLATE)] == 2

    def test_unknown_output_raises(self) -> None:
        synthetic_book = RecipeBook(recipes=())
        with pytest.raises(ValueError, match="no recipe produces IRON_PLATE"):
            ProduceInMachineAt((15, 16), ItemType.IRON_PLATE, 1, book=synthetic_book)


class TestFactoryAliases:
    def test_produce_in_furnace_at(self) -> None:
        goal = ProduceInFurnaceAt((15, 16), ItemType.IRON_PLATE, 3)
        assert isinstance(goal, ProduceInMachineAt)
        assert goal.tile == (15, 16)
        assert goal.output_item == int(ItemType.IRON_PLATE)
        assert goal.count == 3

    def test_produce_in_assembler_at(self) -> None:
        goal = ProduceInAssemblerAt((17, 16), ItemType.WIRE, 5)
        assert isinstance(goal, ProduceInMachineAt)
        assert goal.tile == (17, 16)
        assert goal.output_item == int(ItemType.WIRE)
        assert goal.count == 5

    def test_factory_passes_book(self) -> None:
        book = _book_with_iron_plate_ticks(11)
        goal = ProduceInFurnaceAt((15, 16), ItemType.IRON_PLATE, 1, book=book)
        assert goal.wait_ticks == 14  # 11 + 3 slack
