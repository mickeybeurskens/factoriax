"""Tests for editor save/load without tkinter dependency."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from factoriax.constants import Action, MachineType
from factoriax.editor.dialogs import ask_load_path, ask_save_path
from factoriax.editor.state import (
    editor_state_from_level,
    editor_state_to_level,
    new_editor_state,
    set_machine,
)
from factoriax.levels import load_level, save_level


class TestAskSavePath:
    """ask_save_path returns a path in the levels directory."""

    def test_returns_json_path(self) -> None:
        """Path should be levels/{name}.json."""
        with tempfile.TemporaryDirectory() as d:
            levels_dir = Path(d) / "levels"
            with patch(
                "factoriax.editor.dialogs._LEVELS_DIR", levels_dir
            ):
                result = ask_save_path("my_level")
        assert result is not None
        assert result.name == "my_level.json"
        assert result.parent.name == "levels"

    def test_creates_directory(self) -> None:
        """The levels directory should be created if missing."""
        with tempfile.TemporaryDirectory() as d:
            levels_dir = Path(d) / "levels"
            assert not levels_dir.exists()
            with patch(
                "factoriax.editor.dialogs._LEVELS_DIR", levels_dir
            ):
                ask_save_path("test")
            assert levels_dir.is_dir()


class TestAskLoadPath:
    """ask_load_path returns the newest json file."""

    def test_returns_none_when_empty(self) -> None:
        """No levels directory should return None."""
        with tempfile.TemporaryDirectory() as d:
            levels_dir = Path(d) / "levels"
            with patch(
                "factoriax.editor.dialogs._LEVELS_DIR", levels_dir
            ):
                assert ask_load_path() is None

    def test_returns_newest_file(self) -> None:
        """Should return the most recently modified json file."""
        with tempfile.TemporaryDirectory() as d:
            levels_dir = Path(d) / "levels"
            levels_dir.mkdir()
            (levels_dir / "old.json").write_text("{}")
            (levels_dir / "new.json").write_text("{}")
            with patch(
                "factoriax.editor.dialogs._LEVELS_DIR", levels_dir
            ):
                result = ask_load_path()
            assert result is not None
            assert result.name == "new.json"


class TestEditorSaveLoadRoundTrip:
    """Full save-then-load round trip through editor functions."""

    def test_round_trip_preserves_machines(self) -> None:
        """Machines and directions survive save/load."""
        state = new_editor_state(8, 8, name="roundtrip")
        set_machine(state, 3, 3, int(MachineType.ARM), int(Action.LEFT))

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "test.json"
            level = editor_state_to_level(state)
            save_level(level, path)
            loaded = load_level(path)

        restored = editor_state_from_level(loaded)
        assert restored.machine_types[3, 3] == int(MachineType.ARM)
        assert restored.machine_directions[3, 3] == int(Action.LEFT)
        np.testing.assert_array_equal(
            restored.block_map, state.block_map
        )

    def test_no_tkinter_import(self) -> None:
        """Save/load must not import tkinter."""
        import sys

        had_tkinter = "tkinter" in sys.modules
        with tempfile.TemporaryDirectory() as d:
            levels_dir = Path(d) / "levels"
            with patch(
                "factoriax.editor.dialogs._LEVELS_DIR", levels_dir
            ):
                ask_save_path("test")
                # Create a file so load has something to find
                levels_dir.mkdir(exist_ok=True)
                (levels_dir / "test.json").write_text("{}")
                ask_load_path()
        if not had_tkinter:
            assert "tkinter" not in sys.modules
