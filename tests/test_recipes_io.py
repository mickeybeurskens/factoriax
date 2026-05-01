"""Tests for the TOML balance loader in :mod:`factoriax.recipes_io`."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from factoriax.constants import ItemType
from factoriax.recipes import RecipeBalance, RecipeOverride
from factoriax.recipes_io import load_balance_from_toml


def _write(tmp_path: Path, body: str) -> Path:
    """Write a TOML body to a temp file and return the path."""
    p = tmp_path / "balance.toml"
    p.write_text(textwrap.dedent(body))
    return p


class TestLoadBalanceFromToml:
    def test_load_empty_file_returns_empty_balance(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "")
        balance = load_balance_from_toml(path)
        assert balance == RecipeBalance()

    def test_load_single_override_ticks(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
            [iron_plate]
            ticks = 5
            """,
        )
        balance = load_balance_from_toml(path)
        assert balance.get(int(ItemType.IRON_PLATE)) == RecipeOverride(ticks=5)

    def test_load_all_fields(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
            [wire]
            input_counts = [2, 1]
            output_count = 3
            ticks = 4
            """,
        )
        balance = load_balance_from_toml(path)
        assert balance.get(int(ItemType.WIRE)) == RecipeOverride(
            input_counts=(2, 1),
            output_count=3,
            ticks=4,
        )

    def test_case_insensitive_keys(self, tmp_path: Path) -> None:
        """TOML keys can be lowercase, UPPERCASE, or MixedCase."""
        path = _write(
            tmp_path,
            """
            [Iron_Plate]
            ticks = 7

            [WIRE]
            ticks = 8
            """,
        )
        balance = load_balance_from_toml(path)
        assert balance.get(int(ItemType.IRON_PLATE)) == RecipeOverride(ticks=7)
        assert balance.get(int(ItemType.WIRE)) == RecipeOverride(ticks=8)

    def test_multiple_overrides(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
            [iron_plate]
            ticks = 1

            [copper_plate]
            output_count = 2

            [wire]
            input_counts = [3, 1]
            """,
        )
        balance = load_balance_from_toml(path)
        assert balance.get(int(ItemType.IRON_PLATE)) == RecipeOverride(ticks=1)
        assert balance.get(int(ItemType.COPPER_PLATE)) == RecipeOverride(output_count=2)
        assert balance.get(int(ItemType.WIRE)) == RecipeOverride(input_counts=(3, 1))

    def test_unknown_item_type_raises(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
            [made_up_item]
            ticks = 1
            """,
        )
        with pytest.raises(ValueError, match="Unknown recipe key 'made_up_item'"):
            load_balance_from_toml(path)

    def test_unknown_override_field_raises(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
            [iron_plate]
            cycle_time = 1
            """,
        )
        with pytest.raises(ValueError, match="unknown field"):
            load_balance_from_toml(path)

    def test_wrong_type_for_field_raises(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
            [iron_plate]
            ticks = "fast"
            """,
        )
        with pytest.raises(ValueError, match="expected int"):
            load_balance_from_toml(path)

    def test_input_counts_not_list_raises(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
            [iron_plate]
            input_counts = 5
            """,
        )
        with pytest.raises(ValueError, match="expected list"):
            load_balance_from_toml(path)

    def test_input_counts_non_int_entry_raises(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
            [iron_plate]
            input_counts = [2, "x"]
            """,
        )
        with pytest.raises(ValueError, match="must contain integers"):
            load_balance_from_toml(path)

    def test_empty_input_counts_raises(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
            [iron_plate]
            input_counts = []
            """,
        )
        with pytest.raises(ValueError, match="non-empty list"):
            load_balance_from_toml(path)

    def test_top_level_value_not_table_raises(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
            iron_plate = 5
            """,
        )
        with pytest.raises(ValueError, match="must be a TOML table"):
            load_balance_from_toml(path)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_balance_from_toml(tmp_path / "nope.toml")

    def test_apply_loaded_balance_to_book(self, tmp_path: Path) -> None:
        """End-to-end: load a TOML, apply via with_balance, see the
        change in the projected RecipeTable.
        """
        from factoriax.recipes import BASE_RECIPE_BOOK, RecipeTable

        path = _write(
            tmp_path,
            """
            [iron_plate]
            ticks = 99
            """,
        )
        balance = load_balance_from_toml(path)
        new_book = BASE_RECIPE_BOOK.with_balance(balance)
        table = RecipeTable.from_book(new_book)
        for idx, r in enumerate(new_book.recipes):
            if r.output == int(ItemType.IRON_PLATE):
                assert int(table.ticks[idx]) == 99
                break
        else:
            raise AssertionError("IRON_PLATE not in book")
