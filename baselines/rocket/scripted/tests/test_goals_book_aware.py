"""Tests for the book-aware initialisation of recipe-driven Goals.

``ProduceInMachine``, ``PipelinedProduce``, and ``CraftFromBus`` all
read recipe identity + balance numbers at construction time via the
``book`` keyword argument. These tests verify that:

1. The default book (``BASE_RECIPE_BOOK``) preserves legacy behaviour.
2. A tuned book (via ``RecipeBalance``) flows into the goal's
   captured fields — ``recipe_inputs``, ``wait_ticks``, ``recipe``.
3. A book that omits a recipe raises a clear ``ValueError``.

These are unit tests that only exercise ``__init__``; the full
deposit/wait/withdraw FSM is covered by ``test_produce.py``.
"""

from __future__ import annotations

import pytest

from baselines.rocket.scripted.goals import (
    CraftFromBus,
    PipelinedProduce,
    ProduceInAssembler,
    ProduceInFurnace,
    ProduceInMachine,
)
from factoriax.constants import ItemType, MachineType
from factoriax.recipes import (
    BASE_RECIPE_BOOK,
    Recipe,
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


def _book_with_wire_inputs(input_counts: tuple[int, int]) -> RecipeBook:
    return BASE_RECIPE_BOOK.with_balance(
        RecipeBalance(
            overrides=((int(ItemType.WIRE), RecipeOverride(input_counts=input_counts)),)
        )
    )


# ---------------------------------------------------------------------------
# ProduceInMachine
# ---------------------------------------------------------------------------


class TestProduceInMachineBookAware:
    def test_default_book_uses_base_ticks(self) -> None:
        """No book argument → BASE_RECIPE_BOOK ticks for IRON_PLATE."""
        goal = ProduceInMachine(ItemType.IRON_PLATE, 1)
        # IRON_PLATE.ticks=2; ProduceInMachine adds +3 wait slack.
        assert goal.wait_ticks == 5

    def test_tuned_book_overrides_ticks(self) -> None:
        book = _book_with_iron_plate_ticks(10)
        goal = ProduceInMachine(ItemType.IRON_PLATE, 1, book=book)
        assert goal.wait_ticks == 13  # 10 + 3 slack

    def test_tuned_book_overrides_input_counts(self) -> None:
        """Doubling WIRE.input_counts to (3, 2) shows up in
        ``recipe_inputs`` — the deposit FSM will then push 3+2 items
        per cycle instead of 1+1.
        """
        book = _book_with_wire_inputs((3, 2))
        goal = ProduceInMachine(ItemType.WIRE, 1, book=book)
        # WIRE inputs are (COPPER_PLATE, x) and (TIN_PLATE, y).
        counts = {item: qty for item, qty in goal.recipe_inputs}
        assert counts[int(ItemType.COPPER_PLATE)] == 3
        assert counts[int(ItemType.TIN_PLATE)] == 2

    def test_unknown_output_raises(self) -> None:
        # Use a synthetic book containing only IRON_PLATE; asking for
        # WIRE must raise.
        book = RecipeBook(
            recipes=(
                Recipe(
                    output=int(ItemType.IRON_PLATE),
                    inputs=(
                        (int(ItemType.IRON_ORE), 1),
                        (int(ItemType.COAL), 1),
                    ),
                    ticks=2,
                    name="iron-only",
                ),
            )
        )
        with pytest.raises(ValueError, match="no recipe produces WIRE"):
            ProduceInMachine(ItemType.WIRE, 1, book=book)

    def test_machine_type_inferred_from_book(self) -> None:
        """If ``machine_type`` is omitted, default to the recipe's own
        machine_type as recorded in the book — IRON_PLATE → FURNACE.
        """
        goal = ProduceInMachine(ItemType.IRON_PLATE, 1)
        assert goal.machine_type == int(MachineType.FURNACE)

    def test_explicit_machine_type_wins(self) -> None:
        """Explicit ``machine_type`` overrides book's default."""
        goal = ProduceInMachine(
            ItemType.IRON_PLATE,
            1,
            machine_type=int(MachineType.ASSEMBLER),
        )
        assert goal.machine_type == int(MachineType.ASSEMBLER)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


class TestProduceFactories:
    def test_produce_in_furnace_passes_book(self) -> None:
        book = _book_with_iron_plate_ticks(7)
        goal = ProduceInFurnace(ItemType.IRON_PLATE, 1, book=book)
        assert goal.wait_ticks == 10

    def test_produce_in_assembler_passes_book(self) -> None:
        book = _book_with_wire_inputs((4, 1))
        goal = ProduceInAssembler(ItemType.WIRE, 1, book=book)
        counts = {item: qty for item, qty in goal.recipe_inputs}
        assert counts[int(ItemType.COPPER_PLATE)] == 4


# ---------------------------------------------------------------------------
# PipelinedProduce
# ---------------------------------------------------------------------------


class TestPipelinedProduceBookAware:
    def test_default_book_inputs(self) -> None:
        goal = PipelinedProduce(
            ItemType.WIRE,
            1,
            machine_type=int(MachineType.ASSEMBLER),
        )
        counts = {item: qty for item, qty in goal._recipe_inputs}
        assert counts[int(ItemType.COPPER_PLATE)] == 1
        assert counts[int(ItemType.TIN_PLATE)] == 1

    def test_tuned_book_inputs(self) -> None:
        book = _book_with_wire_inputs((5, 2))
        goal = PipelinedProduce(
            ItemType.WIRE,
            1,
            machine_type=int(MachineType.ASSEMBLER),
            book=book,
        )
        counts = {item: qty for item, qty in goal._recipe_inputs}
        assert counts[int(ItemType.COPPER_PLATE)] == 5
        assert counts[int(ItemType.TIN_PLATE)] == 2


# ---------------------------------------------------------------------------
# CraftFromBus
# ---------------------------------------------------------------------------


class TestCraftFromBusBookAware:
    def test_default_book_recipe(self) -> None:
        bus = {
            int(ItemType.COPPER_PLATE): (5, 5),
            int(ItemType.TIN_PLATE): (5, 6),
        }
        goal = CraftFromBus(ItemType.WIRE, 1, bus_tiles=bus)
        # recipe.inputs preserves order from the book; default is
        # ((COPPER_PLATE, 1), (TIN_PLATE, 1)).
        assert goal.recipe.inputs == (
            (int(ItemType.COPPER_PLATE), 1),
            (int(ItemType.TIN_PLATE), 1),
        )

    def test_tuned_book_recipe(self) -> None:
        bus = {
            int(ItemType.COPPER_PLATE): (5, 5),
            int(ItemType.TIN_PLATE): (5, 6),
        }
        book = _book_with_wire_inputs((4, 3))
        goal = CraftFromBus(ItemType.WIRE, 1, bus_tiles=bus, book=book)
        assert goal.recipe.inputs == (
            (int(ItemType.COPPER_PLATE), 4),
            (int(ItemType.TIN_PLATE), 3),
        )
        # Book is captured for the inner ProduceInMachine so the FSM
        # uses tuned ticks too.
        assert goal.book is book

    def test_missing_bus_tile_raises(self) -> None:
        # Bus has only one of the two WIRE inputs.
        bus = {int(ItemType.COPPER_PLATE): (5, 5)}
        with pytest.raises(ValueError, match="missing bus tile"):
            CraftFromBus(ItemType.WIRE, 1, bus_tiles=bus)
