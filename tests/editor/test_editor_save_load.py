"""Tests for editor save/load via FileDialog without tkinter dependency."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from factoriax.editor.dialogs import FileDialog, _list_level_files
from factoriax.editor.state import (
    editor_state_from_level,
    editor_state_to_level,
    new_editor_state,
    set_machine,
)
from factoriax.engine.constants import Direction, Machine
from factoriax.engine.levels import load_level, save_level


class TestFileDialogSave:
    """FileDialog in save mode returns the correct path."""

    def test_get_path_appends_json(self) -> None:
        """Path should end with .json even without the extension."""
        with tempfile.TemporaryDirectory() as d:
            levels_dir = Path(d) / "levels"
            with patch.object(FileDialog, "__post_init__", lambda self: None):
                dlg = FileDialog(mode="save", filename_text="my_level")
            with patch("factoriax.editor.dialogs.LEVELS_DIR", levels_dir):
                result = dlg.get_path()
        assert result is not None
        assert result.name == "my_level.json"

    def test_creates_directory(self) -> None:
        """get_path should create the levels directory."""
        with tempfile.TemporaryDirectory() as d:
            levels_dir = Path(d) / "levels"
            assert not levels_dir.exists()
            with patch.object(FileDialog, "__post_init__", lambda self: None):
                dlg = FileDialog(mode="save", filename_text="test")
            with patch("factoriax.editor.dialogs.LEVELS_DIR", levels_dir):
                dlg.get_path()
            assert levels_dir.is_dir()

    def test_empty_returns_none(self) -> None:
        """Empty filename should return None."""
        with patch.object(FileDialog, "__post_init__", lambda self: None):
            dlg = FileDialog(mode="save", filename_text="")
        assert dlg.get_path() is None


class TestFileDialogLoad:
    """FileDialog in load mode lists existing files."""

    def test_lists_existing_files(self) -> None:
        """Should scan the levels directory for json files."""
        with tempfile.TemporaryDirectory() as d:
            levels_dir = Path(d) / "levels"
            levels_dir.mkdir()
            (levels_dir / "alpha.json").write_text("{}")
            (levels_dir / "beta.json").write_text("{}")
            with patch("factoriax.editor.dialogs.LEVELS_DIR", levels_dir):
                files = _list_level_files()
        assert files == ["alpha", "beta"]

    def test_load_mode_selects_first(self) -> None:
        """Load mode should pre-select the first file."""
        with tempfile.TemporaryDirectory() as d:
            levels_dir = Path(d) / "levels"
            levels_dir.mkdir()
            (levels_dir / "alpha.json").write_text("{}")
            with patch("factoriax.editor.dialogs.LEVELS_DIR", levels_dir):
                dlg = FileDialog(mode="load")
            assert dlg.selected_index == 0
            assert dlg.filename_text == "alpha"

    def test_empty_dir_returns_none(self) -> None:
        """No files should mean get_path returns None for empty input."""
        with tempfile.TemporaryDirectory() as d:
            levels_dir = Path(d) / "levels"
            levels_dir.mkdir()
            with patch("factoriax.editor.dialogs.LEVELS_DIR", levels_dir):
                dlg = FileDialog(mode="load")
        assert dlg.get_path() is None


class TestFileDialogNavigation:
    """Arrow keys navigate the file list."""

    def test_down_moves_selection(self) -> None:
        """Down arrow should advance selection."""
        with patch.object(FileDialog, "__post_init__", lambda self: None):
            dlg = FileDialog(mode="load")
        dlg.files = ["a", "b", "c"]
        dlg.selected_index = 0
        dlg.filename_text = "a"
        dlg._move_selection(1)
        assert dlg.selected_index == 1
        assert dlg.filename_text == "b"

    def test_up_clamps_at_zero(self) -> None:
        """Up arrow at index 0 should stay at 0."""
        with patch.object(FileDialog, "__post_init__", lambda self: None):
            dlg = FileDialog(mode="load")
        dlg.files = ["a", "b"]
        dlg.selected_index = 0
        dlg.filename_text = "a"
        dlg._move_selection(-1)
        assert dlg.selected_index == 0


class TestEditorSaveLoadRoundTrip:
    """Full save-then-load round trip through editor functions."""

    def test_round_trip_preserves_machines(self) -> None:
        """Machines and directions survive save/load."""
        state = new_editor_state(8, 8, name="roundtrip")
        set_machine(state, 3, 3, int(Machine.CONVEYOR_BELT), int(Direction.LEFT))

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "test.json"
            level = editor_state_to_level(state)
            save_level(level, path)
            loaded = load_level(path)

        restored = editor_state_from_level(loaded)
        assert restored.machine_types[3, 3] == int(Machine.CONVEYOR_BELT)
        assert restored.machine_directions[3, 3] == int(Direction.LEFT)
        np.testing.assert_array_equal(restored.block_map, state.block_map)

    def test_no_tkinter_import(self) -> None:
        """FileDialog must not import tkinter."""
        import sys

        had_tkinter = "tkinter" in sys.modules
        with tempfile.TemporaryDirectory() as d:
            levels_dir = Path(d) / "levels"
            levels_dir.mkdir()
            (levels_dir / "test.json").write_text("{}")
            with patch("factoriax.editor.dialogs.LEVELS_DIR", levels_dir):
                FileDialog(mode="save", filename_text="test")
                FileDialog(mode="load")
        if not had_tkinter:
            assert "tkinter" not in sys.modules
