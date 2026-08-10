"""Tests for the tile and machine edits in the editor state.

An edit writes into the state arrays in place. These tests cover the write,
the bounds guard, and what an erase leaves behind."""

import numpy as np

from factoriax.engine.constants import (
    BlockType,
    Direction,
    Machine,
)
from factoriax.playground.editor.state import (
    ResourceBrush,
    erase_block,
    erase_machine,
    erase_tile,
    fill_rect_tiles,
    new_editor_state,
    set_machine,
    set_tile,
)


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
        assert np.all(state.machine_types == int(Machine.NONE))

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
        set_machine(state, 2, 2, int(Machine.MINER), int(Direction.DOWN))
        assert state.machine_types[2, 2] == int(Machine.MINER)
        assert state.machine_directions[2, 2] == int(Direction.DOWN)
        assert state.dirty is True

    def test_out_of_bounds(self) -> None:
        state = new_editor_state(5, 5)
        set_machine(state, 10, 10, int(Machine.MINER), int(Direction.DOWN))
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
        set_machine(state, 1, 1, int(Machine.MINER), int(Direction.DOWN))
        erase_tile(state, 1, 1)
        assert state.block_map[1, 1] == int(BlockType.DIRT)
        assert state.block_resources[1, 1] == 0
        assert state.machine_types[1, 1] == int(Machine.NONE)

    def test_erase_block_preserves_machine(self) -> None:
        state = new_editor_state(5, 5)
        rng = np.random.default_rng(0)
        brush = ResourceBrush(mode="exact", exact_value=50)
        set_tile(state, 2, 2, int(BlockType.COAL), brush, rng)
        set_machine(state, 2, 2, int(Machine.MINER), int(Direction.DOWN))
        erase_block(state, 2, 2)
        assert state.block_map[2, 2] == int(BlockType.DIRT)
        assert state.block_resources[2, 2] == 0
        assert state.machine_types[2, 2] == int(Machine.MINER)
        assert state.machine_directions[2, 2] == int(Direction.DOWN)

    def test_erase_machine_preserves_terrain(self) -> None:
        state = new_editor_state(5, 5)
        rng = np.random.default_rng(0)
        brush = ResourceBrush(mode="exact", exact_value=50)
        set_tile(state, 2, 2, int(BlockType.COAL), brush, rng)
        set_machine(state, 2, 2, int(Machine.MINER), int(Direction.DOWN))
        erase_machine(state, 2, 2)
        assert state.block_map[2, 2] == int(BlockType.COAL)
        assert state.block_resources[2, 2] == 50
        assert state.machine_types[2, 2] == int(Machine.NONE)
        assert state.machine_directions[2, 2] == 0

    def test_erase_block_out_of_bounds(self) -> None:
        state = new_editor_state(5, 5)
        erase_block(state, -1, 0)
        assert state.dirty is False

    def test_erase_machine_out_of_bounds(self) -> None:
        state = new_editor_state(5, 5)
        erase_machine(state, 10, 10)
        assert state.dirty is False
