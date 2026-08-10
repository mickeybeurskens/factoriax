"""Tests for the conversion between an editor state and a :class:`Level`.

The editor holds a state the engine cannot read. ``editor_state_to_level``
projects it into the shape the engine takes, and ``editor_state_from_level``
reads it back."""

import numpy as np

from factoriax.engine.constants import (
    BLOCK_MAX_RESOURCES,
    BlockType,
    Direction,
    Machine,
)
from factoriax.engine.levels import Level
from factoriax.playground.editor.state import (
    editor_state_from_level,
    editor_state_to_level,
    new_editor_state,
)


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
        machines = np.full((5, 5), int(Machine.NONE), dtype=np.int32)
        machines[2, 2] = int(Machine.PALLET)
        original = Level(
            name="mach_test",
            map_width=5,
            map_height=5,
            block_map=block_map,
            machine_types=machines,
        )
        state = editor_state_from_level(original)
        assert state.machine_types[2, 2] == int(Machine.PALLET)
        level = editor_state_to_level(state)
        assert level.machine_types is not None
        assert level.machine_types[2, 2] == int(Machine.PALLET)

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
        machines = np.full((5, 5), int(Machine.NONE), dtype=np.int32)
        machines[1, 1] = int(Machine.CONVEYOR_BELT)
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
        """All-zero directions are stored as None."""
        state = new_editor_state(5, 5)
        level = editor_state_to_level(state)
        assert level.machine_directions is None
