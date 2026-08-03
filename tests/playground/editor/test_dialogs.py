"""Tests for :mod:`factoriax.playground.editor.dialogs`.

:class:`FileDialog` picks a level file for a save or a load. It draws its
own list and never imports tkinter, so the editor needs no toolkit beyond
pygame.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

from factoriax.playground.editor.dialogs import FileDialog, _list_level_files


class TestFileDialogSave:
    """FileDialog in save mode returns the correct path."""

    def test_get_path_appends_json(self) -> None:
        """The path ends with .json even without the extension."""
        with tempfile.TemporaryDirectory() as d:
            levels_dir = Path(d) / "levels"
            with patch.object(FileDialog, "__post_init__", lambda self: None):
                dlg = FileDialog(mode="save", filename_text="my_level")
            with patch("factoriax.playground.editor.dialogs.LEVELS_DIR", levels_dir):
                result = dlg.get_path()
        assert result is not None
        assert result.name == "my_level.json"

    def test_creates_directory(self) -> None:
        """get_path creates the levels directory."""
        with tempfile.TemporaryDirectory() as d:
            levels_dir = Path(d) / "levels"
            assert not levels_dir.exists()
            with patch.object(FileDialog, "__post_init__", lambda self: None):
                dlg = FileDialog(mode="save", filename_text="test")
            with patch("factoriax.playground.editor.dialogs.LEVELS_DIR", levels_dir):
                dlg.get_path()
            assert levels_dir.is_dir()

    def test_empty_returns_none(self) -> None:
        """An empty filename returns None."""
        with patch.object(FileDialog, "__post_init__", lambda self: None):
            dlg = FileDialog(mode="save", filename_text="")
        assert dlg.get_path() is None


class TestFileDialogLoad:
    """FileDialog in load mode lists existing files."""

    def test_lists_existing_files(self) -> None:
        """The dialog scans the levels directory for json files."""
        with tempfile.TemporaryDirectory() as d:
            levels_dir = Path(d) / "levels"
            levels_dir.mkdir()
            (levels_dir / "alpha.json").write_text("{}")
            (levels_dir / "beta.json").write_text("{}")
            with patch("factoriax.playground.editor.dialogs.LEVELS_DIR", levels_dir):
                files = _list_level_files()
        assert files == ["alpha", "beta"]

    def test_load_mode_selects_first(self) -> None:
        """Load mode pre-selects the first file."""
        with tempfile.TemporaryDirectory() as d:
            levels_dir = Path(d) / "levels"
            levels_dir.mkdir()
            (levels_dir / "alpha.json").write_text("{}")
            with patch("factoriax.playground.editor.dialogs.LEVELS_DIR", levels_dir):
                dlg = FileDialog(mode="load")
            assert dlg.selected_index == 0
            assert dlg.filename_text == "alpha"

    def test_empty_dir_returns_none(self) -> None:
        """With no files, get_path returns None for empty input."""
        with tempfile.TemporaryDirectory() as d:
            levels_dir = Path(d) / "levels"
            levels_dir.mkdir()
            with patch("factoriax.playground.editor.dialogs.LEVELS_DIR", levels_dir):
                dlg = FileDialog(mode="load")
        assert dlg.get_path() is None


class TestFileDialogNavigation:
    """Arrow keys navigate the file list."""

    def test_down_moves_selection(self) -> None:
        """The down arrow advances the selection."""
        with patch.object(FileDialog, "__post_init__", lambda self: None):
            dlg = FileDialog(mode="load")
        dlg.files = ["a", "b", "c"]
        dlg.selected_index = 0
        dlg.filename_text = "a"
        dlg._move_selection(1)
        assert dlg.selected_index == 1
        assert dlg.filename_text == "b"

    def test_up_clamps_at_zero(self) -> None:
        """The up arrow at index 0 stays at 0."""
        with patch.object(FileDialog, "__post_init__", lambda self: None):
            dlg = FileDialog(mode="load")
        dlg.files = ["a", "b"]
        dlg.selected_index = 0
        dlg.filename_text = "a"
        dlg._move_selection(-1)
        assert dlg.selected_index == 0


class TestFileDialogImports:
    """The dialog must not pull in tkinter."""

    def test_no_tkinter_import(self) -> None:
        """FileDialog must not import tkinter."""
        import sys

        had_tkinter = "tkinter" in sys.modules
        with tempfile.TemporaryDirectory() as d:
            levels_dir = Path(d) / "levels"
            levels_dir.mkdir()
            (levels_dir / "test.json").write_text("{}")
            with patch("factoriax.playground.editor.dialogs.LEVELS_DIR", levels_dir):
                FileDialog(mode="save", filename_text="test")
                FileDialog(mode="load")
        if not had_tkinter:
            assert "tkinter" not in sys.modules
