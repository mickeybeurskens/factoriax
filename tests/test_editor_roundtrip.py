"""Round-trip test: Level -> EditorState -> Level preserves all fields.

Catches drift between the Level dataclass and the editor's conversion
functions. If a new optional field is added to Level but not threaded
through editor_state_from_level / editor_state_to_level, this test fails.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt

from factoriax.editor.state import editor_state_from_level, editor_state_to_level
from factoriax.engine.constants import (
    NUM_ITEM_TYPES,
    BlockType,
    Direction,
    ItemType,
    Machine,
)
from factoriax.engine.levels import Level


def _make_fully_populated_level() -> Level:
    """Build a Level with every optional field explicitly set.

    The values are non-trivial (not all-zero) so that the round-trip
    test catches fields that are silently dropped or zeroed out.
    """
    h, w = 4, 5
    block_map = np.full((h, w), int(BlockType.DIRT), dtype=np.int32)
    block_map[0, 0] = int(BlockType.IRON)
    block_map[1, 2] = int(BlockType.COAL)

    block_resources = np.zeros((h, w), dtype=np.int32)
    block_resources[0, 0] = 100
    block_resources[1, 2] = 50

    machine_types = np.full((h, w), int(Machine.NONE), dtype=np.int32)
    machine_types[2, 3] = int(Machine.PALLET)
    machine_types[3, 1] = int(Machine.ASSEMBLER)

    machine_directions = np.zeros((h, w), dtype=np.int32)
    machine_directions[2, 3] = int(Direction.RIGHT)
    machine_directions[3, 1] = int(Direction.UP)

    machine_inventory = np.zeros((h, w, NUM_ITEM_TYPES), dtype=np.int32)
    machine_inventory[2, 3, int(ItemType.IRON_ORE)] = 42

    machine_selected_recipe = np.zeros((h, w), dtype=np.int32)
    machine_selected_recipe[3, 1] = 1

    player_inventory = [
        (int(ItemType.IRON_ORE), 30),
        (int(ItemType.COAL), 10),
    ]

    return Level(
        name="roundtrip_test",
        map_width=w,
        map_height=h,
        block_map=block_map,
        block_resources=block_resources,
        machine_types=machine_types,
        machine_directions=machine_directions,
        machine_inventory=machine_inventory,
        machine_selected_recipe=machine_selected_recipe,
        player_inventory=player_inventory,
    )


class TestEditorRoundTrip:
    """Level -> EditorState -> Level must preserve every field."""

    def test_round_trip_preserves_all_fields(self) -> None:
        """Convert to editor state and back; compare field by field."""
        original = _make_fully_populated_level()
        editor = editor_state_from_level(original)
        restored = editor_state_to_level(editor)

        assert restored.name == original.name
        assert restored.map_width == original.map_width
        assert restored.map_height == original.map_height

        npt.assert_array_equal(restored.block_map, original.block_map)

        assert restored.block_resources is not None
        npt.assert_array_equal(
            restored.block_resources,
            original.block_resources,
        )

        assert restored.machine_types is not None
        npt.assert_array_equal(
            restored.machine_types,
            original.machine_types,
        )

        assert restored.machine_directions is not None
        npt.assert_array_equal(
            restored.machine_directions,
            original.machine_directions,
        )

        assert restored.machine_inventory is not None
        npt.assert_array_equal(
            restored.machine_inventory,
            original.machine_inventory,
        )

        assert restored.machine_selected_recipe is not None
        npt.assert_array_equal(
            restored.machine_selected_recipe,
            original.machine_selected_recipe,
        )

        assert restored.player_inventory == original.player_inventory

    def test_none_fields_stay_none(self) -> None:
        """A Level with all optional fields as None should round-trip cleanly."""
        original = Level(
            name="bare",
            map_width=3,
            map_height=3,
            block_map=np.full((3, 3), int(BlockType.DIRT), dtype=np.int32),
        )
        editor = editor_state_from_level(original)
        restored = editor_state_to_level(editor)

        assert restored.name == original.name
        # Editor zero-fills optional arrays, and editor_state_to_level
        # converts all-zero arrays back to None for compactness.
        assert restored.machine_types is None
        assert restored.machine_directions is None
        assert restored.machine_inventory is None
        assert restored.machine_selected_recipe is None
        assert restored.player_inventory is None


class TestFieldCoverage:
    """Structural check: every Level field is represented in EditorState."""

    # Fields that exist on Level but have no EditorState counterpart by
    # design (dimensions are stored as plain ints, block_map is always
    # present, etc.).
    _EXCLUDED = {
        "name",
        "map_width",
        "map_height",
        "block_map",
        "player_positions",
        # machine_inventory is converted to slot-based on the editor side
        "machine_inventory",
    }

    def test_level_optional_fields_in_editor_state(self) -> None:
        """Every optional Level field must have a matching EditorState field.

        If this test fails after adding a field to Level, add the
        corresponding field to EditorState and thread it through
        editor_state_from_level / editor_state_to_level.
        """
        import dataclasses

        from factoriax.editor.state import EditorState

        level_fields = {
            f.name for f in dataclasses.fields(Level) if f.name not in self._EXCLUDED
        }
        editor_fields = {f.name for f in dataclasses.fields(EditorState)}

        missing = level_fields - editor_fields
        assert not missing, (
            f"Level fields {missing} have no EditorState counterpart. "
            f"Add them to EditorState and update the conversion functions."
        )
