"""Tests for the grid resize operations in the editor state.

Adding or removing a row or a column rewrites every array in the state. The
tests cover what the new cells hold, what a removal drops, and which entities
a shrink clips."""

import numpy as np

from factoriax.engine.constants import BlockType, Direction, Machine
from factoriax.playground.editor.state import (
    ResourceBrush,
    add_column,
    add_row,
    editor_state_to_level,
    new_editor_state,
    remove_column,
    remove_row,
    set_machine,
    set_player_position,
    set_tile,
)


class TestAddColumn:
    """Tests for add_column."""

    def test_width_increases(self) -> None:
        state = new_editor_state(5, 3)
        add_column(state)
        assert state.map_width == 6
        assert state.block_map.shape == (3, 6)

    def test_new_column_is_dirt(self) -> None:
        state = new_editor_state(5, 3)
        add_column(state)
        assert np.all(state.block_map[:, 5] == int(BlockType.DIRT))
        assert np.all(state.block_resources[:, 5] == 0)
        assert np.all(state.machine_types[:, 5] == int(Machine.NONE))

    def test_preserves_existing_data(self) -> None:
        state = new_editor_state(3, 3)
        rng = np.random.default_rng(0)
        brush = ResourceBrush(mode="exact", exact_value=42)
        set_tile(state, 1, 1, int(BlockType.COAL), brush, rng)
        set_machine(state, 0, 0, int(Machine.MINER), int(Direction.DOWN))
        add_column(state)
        assert state.block_map[1, 1] == int(BlockType.COAL)
        assert state.block_resources[1, 1] == 42
        assert state.machine_types[0, 0] == int(Machine.MINER)

    def test_marks_dirty(self) -> None:
        state = new_editor_state(5, 5)
        add_column(state)
        assert state.dirty is True

    def test_all_arrays_consistent_shape(self) -> None:
        state = new_editor_state(4, 6)
        add_column(state)
        expected = (6, 5)
        assert state.block_map.shape == expected
        assert state.block_resources.shape == expected
        assert state.machine_types.shape == expected
        assert state.machine_directions.shape == expected

class TestRemoveColumn:
    """Tests for remove_column."""

    def test_width_decreases(self) -> None:
        state = new_editor_state(5, 3)
        remove_column(state)
        assert state.map_width == 4
        assert state.block_map.shape == (3, 4)

    def test_removes_rightmost(self) -> None:
        state = new_editor_state(5, 3)
        rng = np.random.default_rng(0)
        brush = ResourceBrush(mode="exact", exact_value=10)
        set_tile(state, 4, 0, int(BlockType.IRON), brush, rng)
        set_tile(state, 0, 0, int(BlockType.COAL), brush, rng)
        remove_column(state)
        assert state.block_map[0, 0] == int(BlockType.COAL)
        assert state.map_width == 4

    def test_noop_at_width_1(self) -> None:
        state = new_editor_state(1, 5)
        remove_column(state)
        assert state.map_width == 1
        assert state.dirty is False

    def test_marks_dirty(self) -> None:
        state = new_editor_state(5, 5)
        remove_column(state)
        assert state.dirty is True

    def test_all_arrays_consistent_shape(self) -> None:
        state = new_editor_state(4, 6)
        remove_column(state)
        expected = (6, 3)
        assert state.block_map.shape == expected
        assert state.block_resources.shape == expected
        assert state.machine_types.shape == expected
        assert state.machine_directions.shape == expected

class TestAddRow:
    """Tests for add_row."""

    def test_height_increases(self) -> None:
        state = new_editor_state(3, 5)
        add_row(state)
        assert state.map_height == 6
        assert state.block_map.shape == (6, 3)

    def test_new_row_is_dirt(self) -> None:
        state = new_editor_state(3, 5)
        add_row(state)
        assert np.all(state.block_map[5, :] == int(BlockType.DIRT))
        assert np.all(state.block_resources[5, :] == 0)
        assert np.all(state.machine_types[5, :] == int(Machine.NONE))

    def test_preserves_existing_data(self) -> None:
        state = new_editor_state(3, 3)
        set_machine(state, 1, 1, int(Machine.PALLET), int(Direction.RIGHT))
        add_row(state)
        assert state.machine_types[1, 1] == int(Machine.PALLET)
        assert state.machine_directions[1, 1] == int(Direction.RIGHT)

    def test_all_arrays_consistent_shape(self) -> None:
        state = new_editor_state(4, 6)
        add_row(state)
        expected = (7, 4)
        assert state.block_map.shape == expected
        assert state.block_resources.shape == expected
        assert state.machine_types.shape == expected
        assert state.machine_directions.shape == expected

class TestRemoveRow:
    """Tests for remove_row."""

    def test_height_decreases(self) -> None:
        state = new_editor_state(3, 5)
        remove_row(state)
        assert state.map_height == 4
        assert state.block_map.shape == (4, 3)

    def test_removes_bottom(self) -> None:
        state = new_editor_state(3, 5)
        rng = np.random.default_rng(0)
        brush = ResourceBrush(mode="exact", exact_value=10)
        set_tile(state, 0, 4, int(BlockType.IRON), brush, rng)
        set_tile(state, 0, 0, int(BlockType.COAL), brush, rng)
        remove_row(state)
        assert state.block_map[0, 0] == int(BlockType.COAL)
        assert state.map_height == 4

    def test_noop_at_height_1(self) -> None:
        state = new_editor_state(5, 1)
        remove_row(state)
        assert state.map_height == 1
        assert state.dirty is False

    def test_all_arrays_consistent_shape(self) -> None:
        state = new_editor_state(4, 6)
        remove_row(state)
        expected = (5, 4)
        assert state.block_map.shape == expected
        assert state.block_resources.shape == expected
        assert state.machine_types.shape == expected
        assert state.machine_directions.shape == expected

class TestResizeRoundTrip:
    """Tests that resize + save/load produces valid levels."""

    def test_add_column_then_save(self) -> None:
        state = new_editor_state(5, 5)
        add_column(state)
        level = editor_state_to_level(state)
        assert level.map_width == 6
        assert level.block_map.shape == (5, 6)

    def test_remove_row_then_save(self) -> None:
        state = new_editor_state(5, 5)
        remove_row(state)
        level = editor_state_to_level(state)
        assert level.map_height == 4
        assert level.block_map.shape == (4, 5)

    def test_repeated_add_remove_is_stable(self) -> None:
        state = new_editor_state(5, 5)
        for _ in range(10):
            add_column(state)
            add_row(state)
        for _ in range(10):
            remove_column(state)
            remove_row(state)
        assert state.map_width == 5
        assert state.map_height == 5
        assert state.block_map.shape == (5, 5)

    def test_resize_then_paint_in_new_area(self) -> None:
        state = new_editor_state(3, 3)
        add_column(state)
        add_row(state)
        rng = np.random.default_rng(0)
        brush = ResourceBrush(mode="exact", exact_value=10)
        set_tile(state, 3, 3, int(BlockType.COAL), brush, rng)
        assert state.block_map[3, 3] == int(BlockType.COAL)
        assert state.block_resources[3, 3] == 10

    def test_resize_then_place_machine_in_new_area(self) -> None:
        state = new_editor_state(3, 3)
        add_column(state)
        add_row(state)
        set_machine(state, 3, 3, int(Machine.PALLET), int(Direction.DOWN))
        assert state.machine_types[3, 3] == int(Machine.PALLET)

    def test_shrink_discards_edge_data(self) -> None:
        state = new_editor_state(5, 5)
        rng = np.random.default_rng(0)
        brush = ResourceBrush(mode="exact", exact_value=10)
        set_tile(state, 4, 4, int(BlockType.IRON), brush, rng)
        set_machine(state, 4, 4, int(Machine.MINER), int(Direction.DOWN))
        remove_column(state)
        remove_row(state)
        assert state.map_width == 4
        assert state.map_height == 4
        assert state.block_map.shape == (4, 4)

    def test_grow_large_all_arrays_valid(self) -> None:
        state = new_editor_state(2, 2)
        for _ in range(50):
            add_column(state)
            add_row(state)
        assert state.map_width == 52
        assert state.map_height == 52
        expected = (52, 52)
        assert state.block_map.shape == expected
        assert state.block_resources.shape == expected
        assert state.machine_types.shape == expected
        assert state.machine_directions.shape == expected
        assert np.all(state.block_map[:, 2:] == int(BlockType.DIRT))
        assert np.all(state.machine_types[:, 2:] == int(Machine.NONE))

class TestResizeClipsEntities:
    """Tests for entity clipping on map resize."""

    def test_remove_column_clips(self) -> None:
        state = new_editor_state(5, 5)
        set_player_position(state, 0, 4, 0)
        remove_column(state)
        assert 0 not in state.player_positions

    def test_remove_row_clips(self) -> None:
        state = new_editor_state(5, 5)
        set_player_position(state, 0, 0, 4)
        remove_row(state)
        assert 0 not in state.player_positions

    def test_entities_inside_bounds_kept(self) -> None:
        state = new_editor_state(5, 5)
        set_player_position(state, 0, 0, 0)
        remove_column(state)
        assert state.player_positions == {0: (0, 0)}

