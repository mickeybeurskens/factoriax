"""Tests for the editor state module."""

import numpy as np

from factoriax.constants import BLOCK_MAX_RESOURCES, Action, BlockType, MachineType
from factoriax.editor.state import (
    ResourceBrush,
    add_column,
    add_row,
    editor_state_from_level,
    editor_state_to_level,
    erase_block,
    erase_machine,
    erase_tile,
    fill_rect_tiles,
    new_editor_state,
    remove_column,
    remove_row,
    sample_resource,
    set_machine,
    set_tile,
)
from factoriax.levels import Level


class TestNewEditorState:
    """Tests for new_editor_state."""

    def test_creates_all_dirt(self) -> None:
        state = new_editor_state(10, 8)
        assert state.map_width == 10
        assert state.map_height == 8
        assert state.block_map.shape == (8, 10)
        assert np.all(state.block_map == int(BlockType.DIRT))

    def test_resources_zero(self) -> None:
        state = new_editor_state(5, 5)
        assert np.all(state.block_resources == 0)

    def test_no_machines(self) -> None:
        state = new_editor_state(5, 5)
        assert np.all(state.machine_types == int(MachineType.NONE))

    def test_not_dirty(self) -> None:
        state = new_editor_state(5, 5)
        assert state.dirty is False

    def test_name(self) -> None:
        state = new_editor_state(5, 5, name="test_level")
        assert state.name == "test_level"


class TestSetTile:
    """Tests for set_tile."""

    def test_paint_ore_exact(self) -> None:
        state = new_editor_state(5, 5)
        rng = np.random.default_rng(0)
        brush = ResourceBrush(mode="exact", exact_value=42)
        set_tile(state, 2, 3, int(BlockType.COAL), brush, rng)
        assert state.block_map[3, 2] == int(BlockType.COAL)
        assert state.block_resources[3, 2] == 42
        assert state.dirty is True

    def test_paint_dirt_zeroes_resource(self) -> None:
        state = new_editor_state(5, 5)
        rng = np.random.default_rng(0)
        brush = ResourceBrush(mode="exact", exact_value=50)
        set_tile(state, 1, 1, int(BlockType.COAL), brush, rng)
        set_tile(state, 1, 1, int(BlockType.DIRT), brush, rng)
        assert state.block_resources[1, 1] == 0

    def test_out_of_bounds_ignored(self) -> None:
        state = new_editor_state(5, 5)
        rng = np.random.default_rng(0)
        brush = ResourceBrush()
        set_tile(state, -1, 0, int(BlockType.COAL), brush, rng)
        set_tile(state, 5, 0, int(BlockType.COAL), brush, rng)
        assert state.dirty is False

    def test_paint_ore_range(self) -> None:
        state = new_editor_state(5, 5)
        rng = np.random.default_rng(0)
        brush = ResourceBrush(mode="range", range_min=10, range_max=20)
        set_tile(state, 0, 0, int(BlockType.IRON), brush, rng)
        assert 10 <= state.block_resources[0, 0] <= 20


class TestSetMachine:
    """Tests for set_machine."""

    def test_place_machine(self) -> None:
        state = new_editor_state(5, 5)
        set_machine(state, 2, 2, int(MachineType.MINER), int(Action.DOWN))
        assert state.machine_types[2, 2] == int(MachineType.MINER)
        assert state.machine_directions[2, 2] == int(Action.DOWN)
        assert state.dirty is True

    def test_out_of_bounds(self) -> None:
        state = new_editor_state(5, 5)
        set_machine(state, 10, 10, int(MachineType.MINER), int(Action.DOWN))
        assert state.dirty is False


class TestFillRectTiles:
    """Tests for fill_rect_tiles."""

    def test_fill_ore(self) -> None:
        state = new_editor_state(10, 10)
        rng = np.random.default_rng(0)
        brush = ResourceBrush(mode="exact", exact_value=50)
        fill_rect_tiles(state, 1, 1, 3, 3, int(BlockType.IRON), brush, rng)
        assert np.all(state.block_map[1:4, 1:4] == int(BlockType.IRON))
        assert np.all(state.block_resources[1:4, 1:4] == 50)
        assert state.dirty is True

    def test_fill_clamps_to_bounds(self) -> None:
        state = new_editor_state(5, 5)
        rng = np.random.default_rng(0)
        brush = ResourceBrush()
        fill_rect_tiles(state, -2, -2, 3, 3, int(BlockType.COAL), brush, rng)
        assert state.block_map[0, 0] == int(BlockType.COAL)

    def test_fill_range_mode(self) -> None:
        state = new_editor_state(10, 10)
        rng = np.random.default_rng(0)
        brush = ResourceBrush(mode="range", range_min=5, range_max=15)
        fill_rect_tiles(state, 0, 0, 4, 4, int(BlockType.COPPER), brush, rng)
        resources = state.block_resources[0:5, 0:5]
        assert np.all(resources >= 5)
        assert np.all(resources <= 15)


class TestEraseTile:
    """Tests for erase_tile, erase_block, and erase_machine."""

    def test_erase_resets_to_dirt(self) -> None:
        state = new_editor_state(5, 5)
        rng = np.random.default_rng(0)
        brush = ResourceBrush()
        set_tile(state, 1, 1, int(BlockType.COAL), brush, rng)
        set_machine(state, 1, 1, int(MachineType.MINER), int(Action.DOWN))
        erase_tile(state, 1, 1)
        assert state.block_map[1, 1] == int(BlockType.DIRT)
        assert state.block_resources[1, 1] == 0
        assert state.machine_types[1, 1] == int(MachineType.NONE)

    def test_erase_block_preserves_machine(self) -> None:
        state = new_editor_state(5, 5)
        rng = np.random.default_rng(0)
        brush = ResourceBrush(mode="exact", exact_value=50)
        set_tile(state, 2, 2, int(BlockType.COAL), brush, rng)
        set_machine(state, 2, 2, int(MachineType.MINER), int(Action.DOWN))
        erase_block(state, 2, 2)
        assert state.block_map[2, 2] == int(BlockType.DIRT)
        assert state.block_resources[2, 2] == 0
        assert state.machine_types[2, 2] == int(MachineType.MINER)
        assert state.machine_directions[2, 2] == int(Action.DOWN)

    def test_erase_machine_preserves_terrain(self) -> None:
        state = new_editor_state(5, 5)
        rng = np.random.default_rng(0)
        brush = ResourceBrush(mode="exact", exact_value=50)
        set_tile(state, 2, 2, int(BlockType.COAL), brush, rng)
        set_machine(state, 2, 2, int(MachineType.MINER), int(Action.DOWN))
        erase_machine(state, 2, 2)
        assert state.block_map[2, 2] == int(BlockType.COAL)
        assert state.block_resources[2, 2] == 50
        assert state.machine_types[2, 2] == int(MachineType.NONE)
        assert state.machine_directions[2, 2] == 0

    def test_erase_block_out_of_bounds(self) -> None:
        state = new_editor_state(5, 5)
        erase_block(state, -1, 0)
        assert state.dirty is False

    def test_erase_machine_out_of_bounds(self) -> None:
        state = new_editor_state(5, 5)
        erase_machine(state, 10, 10)
        assert state.dirty is False


class TestLevelConversion:
    """Tests for editor_state_from_level and editor_state_to_level."""

    def test_round_trip(self) -> None:
        original = Level(
            name="test",
            map_width=5,
            map_height=5,
            block_map=np.full((5, 5), int(BlockType.DIRT), dtype=np.int32),
        )
        state = editor_state_from_level(original)
        assert state.name == "test"
        assert state.map_width == 5
        level = editor_state_to_level(state)
        assert level.name == "test"
        assert np.array_equal(level.block_map, original.block_map)

    def test_resources_preserved(self) -> None:
        block_map = np.full((5, 5), int(BlockType.DIRT), dtype=np.int32)
        block_map[0, 0] = int(BlockType.COAL)
        resources = np.zeros((5, 5), dtype=np.int32)
        resources[0, 0] = 42
        original = Level(
            name="res_test",
            map_width=5,
            map_height=5,
            block_map=block_map,
            block_resources=resources,
        )
        state = editor_state_from_level(original)
        assert state.block_resources[0, 0] == 42
        level = editor_state_to_level(state)
        assert level.block_resources is not None
        assert level.block_resources[0, 0] == 42

    def test_machines_preserved(self) -> None:
        block_map = np.full((5, 5), int(BlockType.DIRT), dtype=np.int32)
        machines = np.full((5, 5), int(MachineType.NONE), dtype=np.int32)
        machines[2, 2] = int(MachineType.CHEST)
        original = Level(
            name="mach_test",
            map_width=5,
            map_height=5,
            block_map=block_map,
            machine_types=machines,
        )
        state = editor_state_from_level(original)
        assert state.machine_types[2, 2] == int(MachineType.CHEST)
        level = editor_state_to_level(state)
        assert level.machine_types is not None
        assert level.machine_types[2, 2] == int(MachineType.CHEST)

    def test_all_zero_resources_become_none(self) -> None:
        state = new_editor_state(5, 5)
        level = editor_state_to_level(state)
        assert level.block_resources is None

    def test_all_none_machines_become_none(self) -> None:
        state = new_editor_state(5, 5)
        level = editor_state_to_level(state)
        assert level.machine_types is None

    def test_none_resources_get_defaults(self) -> None:
        block_map = np.full((5, 5), int(BlockType.DIRT), dtype=np.int32)
        block_map[0, 0] = int(BlockType.COAL)
        original = Level(
            name="default_res",
            map_width=5,
            map_height=5,
            block_map=block_map,
        )
        state = editor_state_from_level(original)
        assert state.block_resources[0, 0] == BLOCK_MAX_RESOURCES
        assert state.block_resources[1, 1] == 0

    def test_directions_preserved(self) -> None:
        """Machine directions must survive editor round-trip."""
        block_map = np.full((5, 5), int(BlockType.DIRT), dtype=np.int32)
        machines = np.full(
            (5, 5), int(MachineType.NONE), dtype=np.int32
        )
        machines[1, 1] = int(MachineType.ARM)
        dirs = np.zeros((5, 5), dtype=np.int32)
        dirs[1, 1] = int(Action.RIGHT)
        original = Level(
            name="dir_test",
            map_width=5,
            map_height=5,
            block_map=block_map,
            machine_types=machines,
            machine_directions=dirs,
        )
        state = editor_state_from_level(original)
        assert state.machine_directions[1, 1] == int(Action.RIGHT)
        level = editor_state_to_level(state)
        assert level.machine_directions is not None
        assert level.machine_directions[1, 1] == int(Action.RIGHT)

    def test_all_zero_directions_become_none(self) -> None:
        """All-zero directions should be stored as None."""
        state = new_editor_state(5, 5)
        level = editor_state_to_level(state)
        assert level.machine_directions is None


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
        assert np.all(state.machine_types[:, 5] == int(MachineType.NONE))

    def test_preserves_existing_data(self) -> None:
        state = new_editor_state(3, 3)
        rng = np.random.default_rng(0)
        brush = ResourceBrush(mode="exact", exact_value=42)
        set_tile(state, 1, 1, int(BlockType.COAL), brush, rng)
        set_machine(state, 0, 0, int(MachineType.MINER), int(Action.DOWN))
        add_column(state)
        assert state.block_map[1, 1] == int(BlockType.COAL)
        assert state.block_resources[1, 1] == 42
        assert state.machine_types[0, 0] == int(MachineType.MINER)

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
        assert np.all(state.machine_types[5, :] == int(MachineType.NONE))

    def test_preserves_existing_data(self) -> None:
        state = new_editor_state(3, 3)
        set_machine(state, 1, 1, int(MachineType.CHEST), int(Action.RIGHT))
        add_row(state)
        assert state.machine_types[1, 1] == int(MachineType.CHEST)
        assert state.machine_directions[1, 1] == int(Action.RIGHT)

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
        set_machine(state, 3, 3, int(MachineType.CHEST), int(Action.DOWN))
        assert state.machine_types[3, 3] == int(MachineType.CHEST)

    def test_shrink_discards_edge_data(self) -> None:
        state = new_editor_state(5, 5)
        rng = np.random.default_rng(0)
        brush = ResourceBrush(mode="exact", exact_value=10)
        set_tile(state, 4, 4, int(BlockType.IRON), brush, rng)
        set_machine(state, 4, 4, int(MachineType.MINER), int(Action.DOWN))
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
        assert np.all(state.machine_types[:, 2:] == int(MachineType.NONE))


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
