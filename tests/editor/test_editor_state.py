"""Tests for the editor state module."""

import numpy as np

from factoriax.constants import (
    BLOCK_MAX_RESOURCES,
    BlockType,
    Direction,
    ItemType,
    MachineType,
)
from factoriax.editor.state import (
    InvTarget,
    ResourceBrush,
    add_biter,
    add_column,
    add_row,
    clear_inventory_slot,
    editor_state_from_level,
    editor_state_to_level,
    erase_block,
    erase_entity,
    erase_machine,
    erase_tile,
    fill_rect_tiles,
    get_inventory_slots,
    get_num_slots,
    new_editor_state,
    remove_biters_at,
    remove_column,
    remove_player_at,
    remove_row,
    sample_resource,
    set_inventory_slot,
    set_machine,
    set_player_position,
    set_tile,
    swap_inventory_slots,
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
        set_machine(state, 2, 2, int(MachineType.MINER), int(Direction.DOWN))
        assert state.machine_types[2, 2] == int(MachineType.MINER)
        assert state.machine_directions[2, 2] == int(Direction.DOWN)
        assert state.dirty is True

    def test_out_of_bounds(self) -> None:
        state = new_editor_state(5, 5)
        set_machine(state, 10, 10, int(MachineType.MINER), int(Direction.DOWN))
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
        set_machine(state, 1, 1, int(MachineType.MINER), int(Direction.DOWN))
        erase_tile(state, 1, 1)
        assert state.block_map[1, 1] == int(BlockType.DIRT)
        assert state.block_resources[1, 1] == 0
        assert state.machine_types[1, 1] == int(MachineType.NONE)

    def test_erase_block_preserves_machine(self) -> None:
        state = new_editor_state(5, 5)
        rng = np.random.default_rng(0)
        brush = ResourceBrush(mode="exact", exact_value=50)
        set_tile(state, 2, 2, int(BlockType.COAL), brush, rng)
        set_machine(state, 2, 2, int(MachineType.MINER), int(Direction.DOWN))
        erase_block(state, 2, 2)
        assert state.block_map[2, 2] == int(BlockType.DIRT)
        assert state.block_resources[2, 2] == 0
        assert state.machine_types[2, 2] == int(MachineType.MINER)
        assert state.machine_directions[2, 2] == int(Direction.DOWN)

    def test_erase_machine_preserves_terrain(self) -> None:
        state = new_editor_state(5, 5)
        rng = np.random.default_rng(0)
        brush = ResourceBrush(mode="exact", exact_value=50)
        set_tile(state, 2, 2, int(BlockType.COAL), brush, rng)
        set_machine(state, 2, 2, int(MachineType.MINER), int(Direction.DOWN))
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
        machines[2, 2] = int(MachineType.PALLET)
        original = Level(
            name="mach_test",
            map_width=5,
            map_height=5,
            block_map=block_map,
            machine_types=machines,
        )
        state = editor_state_from_level(original)
        assert state.machine_types[2, 2] == int(MachineType.PALLET)
        level = editor_state_to_level(state)
        assert level.machine_types is not None
        assert level.machine_types[2, 2] == int(MachineType.PALLET)

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
        dirs[1, 1] = int(Direction.RIGHT)
        original = Level(
            name="dir_test",
            map_width=5,
            map_height=5,
            block_map=block_map,
            machine_types=machines,
            machine_directions=dirs,
        )
        state = editor_state_from_level(original)
        assert state.machine_directions[1, 1] == int(Direction.RIGHT)
        level = editor_state_to_level(state)
        assert level.machine_directions is not None
        assert level.machine_directions[1, 1] == int(Direction.RIGHT)

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
        set_machine(state, 0, 0, int(MachineType.MINER), int(Direction.DOWN))
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
        set_machine(state, 1, 1, int(MachineType.PALLET), int(Direction.RIGHT))
        add_row(state)
        assert state.machine_types[1, 1] == int(MachineType.PALLET)
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
        set_machine(state, 3, 3, int(MachineType.PALLET), int(Direction.DOWN))
        assert state.machine_types[3, 3] == int(MachineType.PALLET)

    def test_shrink_discards_edge_data(self) -> None:
        state = new_editor_state(5, 5)
        rng = np.random.default_rng(0)
        brush = ResourceBrush(mode="exact", exact_value=10)
        set_tile(state, 4, 4, int(BlockType.IRON), brush, rng)
        set_machine(state, 4, 4, int(MachineType.MINER), int(Direction.DOWN))
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


class TestAddBiter:
    """Tests for add_biter."""

    def test_add_biter(self) -> None:
        state = new_editor_state(5, 5)
        add_biter(state, 1, 1)
        assert state.biter_positions == [(1, 1)]
        assert state.dirty is True

    def test_multiple_at_same_tile(self) -> None:
        state = new_editor_state(5, 5)
        add_biter(state, 2, 2)
        add_biter(state, 2, 2)
        assert len(state.biter_positions) == 2

    def test_out_of_bounds_ignored(self) -> None:
        state = new_editor_state(5, 5)
        add_biter(state, -1, 0)
        add_biter(state, 5, 0)
        assert state.biter_positions == []
        assert state.dirty is False


class TestRemoveBitersAt:
    """Tests for remove_biters_at."""

    def test_remove_all_at_tile(self) -> None:
        state = new_editor_state(5, 5)
        add_biter(state, 2, 2)
        add_biter(state, 2, 2)
        add_biter(state, 3, 3)
        state.dirty = False
        remove_biters_at(state, 2, 2)
        assert state.biter_positions == [(3, 3)]
        assert state.dirty is True

    def test_remove_nonexistent(self) -> None:
        state = new_editor_state(5, 5)
        remove_biters_at(state, 2, 2)
        assert state.dirty is False


class TestEraseEntity:
    """Tests for erase_entity."""

    def test_erases_both(self) -> None:
        state = new_editor_state(5, 5)
        set_player_position(state, 0, 2, 2)
        add_biter(state, 2, 2)
        erase_entity(state, 2, 2)
        assert state.player_positions == {}
        assert state.biter_positions == []


class TestResizeClipsEntities:
    """Tests for entity clipping on map resize."""

    def test_remove_column_clips(self) -> None:
        state = new_editor_state(5, 5)
        set_player_position(state, 0, 4, 0)
        add_biter(state, 4, 2)
        remove_column(state)
        assert 0 not in state.player_positions
        assert (4, 2) not in state.biter_positions

    def test_remove_row_clips(self) -> None:
        state = new_editor_state(5, 5)
        set_player_position(state, 0, 0, 4)
        add_biter(state, 2, 4)
        remove_row(state)
        assert 0 not in state.player_positions
        assert (2, 4) not in state.biter_positions

    def test_entities_inside_bounds_kept(self) -> None:
        state = new_editor_state(5, 5)
        set_player_position(state, 0, 0, 0)
        add_biter(state, 1, 1)
        remove_column(state)
        assert state.player_positions == {0: (0, 0)}
        assert state.biter_positions == [(1, 1)]


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

    def test_biter_positions_round_trip(self) -> None:
        state = new_editor_state(8, 8)
        add_biter(state, 0, 0)
        add_biter(state, 3, 4)
        level = editor_state_to_level(state)
        assert level.biter_positions == [(0, 0), (3, 4)]
        state2 = editor_state_from_level(level)
        assert state2.biter_positions == [(0, 0), (3, 4)]

    def test_empty_entities_become_none(self) -> None:
        state = new_editor_state(5, 5)
        level = editor_state_to_level(state)
        assert level.player_positions is None
        assert level.biter_positions is None


# ---------------------------------------------------------------------------
# Inventory helpers
# ---------------------------------------------------------------------------


class TestGetInventorySlots:
    """Tests for get_inventory_slots."""

    def test_empty_player_inventory(self) -> None:
        state = new_editor_state(5, 5)
        target: InvTarget = ("player", 0, 0)
        slots = get_inventory_slots(state, target)
        assert len(slots) == 10
        assert all(item == int(ItemType.EMPTY) for item, _ in slots)

    def test_player_with_items(self) -> None:
        state = new_editor_state(5, 5)
        state.player_inventories[0] = [
            (int(ItemType.COAL), 50),
            (int(ItemType.IRON), 30),
        ]
        slots = get_inventory_slots(state, ("player", 0, 0))
        assert slots[0] == (int(ItemType.COAL), 50)
        assert slots[1] == (int(ItemType.IRON), 30)
        assert slots[2] == (int(ItemType.EMPTY), 0)

    def test_machine_inventory(self) -> None:
        state = new_editor_state(5, 5)
        set_machine(state, 2, 2, int(MachineType.PALLET), 0)
        state.machine_inventory_items[2, 2, 0] = int(ItemType.IRON)
        state.machine_inventory_counts[2, 2, 0] = 10
        slots = get_inventory_slots(state, ("machine", 2, 2))
        assert slots[0] == (int(ItemType.IRON), 10)


class TestSetInventorySlot:
    """Tests for set_inventory_slot."""

    def test_set_player_slot(self) -> None:
        state = new_editor_state(5, 5)
        target: InvTarget = ("player", 0, 0)
        set_inventory_slot(state, target, 3, int(ItemType.COPPER), 25)
        slots = get_inventory_slots(state, target)
        assert slots[3] == (int(ItemType.COPPER), 25)
        assert state.dirty is True

    def test_set_machine_slot(self) -> None:
        state = new_editor_state(5, 5)
        set_machine(state, 1, 1, int(MachineType.MINER), 0)
        target: InvTarget = ("machine", 1, 1)
        set_inventory_slot(state, target, 0, int(ItemType.COAL), 5)
        assert state.machine_inventory_items[1, 1, 0] == int(ItemType.COAL)
        assert state.machine_inventory_counts[1, 1, 0] == 5


class TestClearInventorySlot:
    """Tests for clear_inventory_slot."""

    def test_clear_player_slot(self) -> None:
        state = new_editor_state(5, 5)
        target: InvTarget = ("player", 0, 0)
        set_inventory_slot(state, target, 0, int(ItemType.IRON), 10)
        clear_inventory_slot(state, target, 0)
        slots = get_inventory_slots(state, target)
        assert slots[0] == (int(ItemType.EMPTY), 0)


class TestSwapInventorySlots:
    """Tests for swap_inventory_slots."""

    def test_swap(self) -> None:
        state = new_editor_state(5, 5)
        target: InvTarget = ("player", 0, 0)
        set_inventory_slot(state, target, 0, int(ItemType.COAL), 10)
        set_inventory_slot(state, target, 1, int(ItemType.IRON), 20)
        swap_inventory_slots(state, target, 0, 1)
        slots = get_inventory_slots(state, target)
        assert slots[0] == (int(ItemType.IRON), 20)
        assert slots[1] == (int(ItemType.COAL), 10)


class TestGetNumSlots:
    """Tests for get_num_slots."""

    def test_player_has_ten(self) -> None:
        state = new_editor_state(5, 5)
        assert get_num_slots(state, ("player", 0, 0)) == 10

    def test_miner_has_two(self) -> None:
        state = new_editor_state(5, 5)
        set_machine(state, 0, 0, int(MachineType.MINER), 0)
        assert get_num_slots(state, ("machine", 0, 0)) == 2

    def test_pallet_has_one(self) -> None:
        state = new_editor_state(5, 5)
        set_machine(state, 0, 0, int(MachineType.PALLET), 0)
        assert get_num_slots(state, ("machine", 0, 0)) == 1


class TestPerPlayerInventoryRoundTrip:
    """Tests for per-player inventory through Level save/load."""

    def test_round_trip(self) -> None:
        state = new_editor_state(8, 8)
        set_player_position(state, 0, 1, 1)
        set_inventory_slot(state, ("player", 0, 0), 0, int(ItemType.COAL), 50)
        set_inventory_slot(state, ("player", 0, 0), 1, int(ItemType.IRON), 30)
        level = editor_state_to_level(state)
        assert level.player_inventories is not None
        assert 0 in level.player_inventories
        state2 = editor_state_from_level(level)
        slots = get_inventory_slots(state2, ("player", 0, 0))
        assert slots[0] == (int(ItemType.COAL), 50)
        assert slots[1] == (int(ItemType.IRON), 30)

    def test_empty_becomes_none(self) -> None:
        state = new_editor_state(5, 5)
        level = editor_state_to_level(state)
        assert level.player_inventories is None
