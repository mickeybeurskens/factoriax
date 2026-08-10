"""Tests for the player and entity edits in the editor state.

A player position and a resource brush are entity-level edits. They do not
write the tile grid, so they carry their own bounds and identity rules."""

import numpy as np

from factoriax.playground.editor.state import (
    ResourceBrush,
    editor_state_from_level,
    editor_state_to_level,
    erase_entity,
    new_editor_state,
    remove_player_at,
    sample_resource,
    set_player_position,
)


class TestSampleResource:
    """Tests for sample_resource."""

    def test_exact_mode(self) -> None:
        brush = ResourceBrush(mode="exact", exact_value=77)
        rng = np.random.default_rng(0)
        assert sample_resource(brush, rng) == 77

    def test_range_mode(self) -> None:
        brush = ResourceBrush(mode="range", range_min=10, range_max=20)
        rng = np.random.default_rng(0)
        for _ in range(50):
            val = sample_resource(brush, rng)
            assert 10 <= val <= 20


# ---------------------------------------------------------------------------
# Entity placement
# ---------------------------------------------------------------------------


class TestSetPlayerPosition:
    """Tests for set_player_position."""

    def test_place_player(self) -> None:
        state = new_editor_state(5, 5)
        set_player_position(state, 0, 2, 3)
        assert state.player_positions == {0: (2, 3)}
        assert state.dirty is True

    def test_move_player(self) -> None:
        state = new_editor_state(5, 5)
        set_player_position(state, 0, 1, 1)
        set_player_position(state, 0, 3, 4)
        assert state.player_positions == {0: (3, 4)}

    def test_multiple_players(self) -> None:
        state = new_editor_state(5, 5)
        set_player_position(state, 0, 0, 0)
        set_player_position(state, 3, 4, 4)
        assert state.player_positions == {0: (0, 0), 3: (4, 4)}

    def test_out_of_bounds_ignored(self) -> None:
        state = new_editor_state(5, 5)
        set_player_position(state, 0, -1, 0)
        set_player_position(state, 1, 5, 0)
        assert state.player_positions == {}
        assert state.dirty is False


class TestRemovePlayerAt:
    """Tests for remove_player_at."""

    def test_remove_existing(self) -> None:
        state = new_editor_state(5, 5)
        set_player_position(state, 0, 2, 2)
        state.dirty = False
        remove_player_at(state, 2, 2)
        assert state.player_positions == {}
        assert state.dirty is True

    def test_remove_nonexistent(self) -> None:
        state = new_editor_state(5, 5)
        remove_player_at(state, 2, 2)
        assert state.dirty is False


class TestEraseEntity:
    """Tests for erase_entity."""

    def test_erases_both(self) -> None:
        state = new_editor_state(5, 5)
        set_player_position(state, 0, 2, 2)
        erase_entity(state, 2, 2)
        assert state.player_positions == {}


class TestEntityRoundTrip:
    """Tests for entity save/load through EditorState ↔ Level."""

    def test_player_positions_round_trip(self) -> None:
        state = new_editor_state(8, 8)
        set_player_position(state, 0, 1, 2)
        set_player_position(state, 2, 5, 5)
        level = editor_state_to_level(state)
        assert level.player_positions == [(1, 2), (5, 5)]
        state2 = editor_state_from_level(level)
        assert state2.player_positions == {0: (1, 2), 1: (5, 5)}

    def test_empty_entities_become_none(self) -> None:
        state = new_editor_state(5, 5)
        level = editor_state_to_level(state)
        assert level.player_positions is None


# ---------------------------------------------------------------------------
# Inventory helpers
# ---------------------------------------------------------------------------
