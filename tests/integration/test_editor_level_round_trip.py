"""Tests that an edit survives a trip through :class:`Level` unchanged.

This file spans two subpackages, so it is not in the mirror.
``playground/editor/state.py`` holds the edit, and ``engine/levels.py``
holds the format it is saved in. A field that changes meaning between them
is a silent edit that neither side's own tests can see.

The editor saves by converting its state to a ``Level`` and loads by converting
back. A field that changes meaning on the way is a silent edit to the level, so
these tests drive the conversion in both directions and compare.

The existing round trip test covers a level with every field populated. These
cover the edges that the populated case hides: a zero that means "nothing", an
item that appears twice, more items than a machine shows, a shared list, and a
gap in the player numbering.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from factoriax.engine.constants import (
    NUM_ITEM_TYPES,
    BlockType,
    Direction,
    ItemType,
    Machine,
)
from factoriax.engine.levels import Level, build_state, load_level, save_level
from factoriax.playground.editor.state import (
    editor_state_from_level,
    editor_state_to_level,
    get_inventory_slots,
    new_editor_state,
    remove_player_at,
    set_inventory_slot,
    set_machine,
)


def _dirt(h: int, w: int) -> np.ndarray:
    return np.full((h, w), int(BlockType.DIRT), dtype=np.int32)


def _round_trip(level: Level) -> Level:
    return editor_state_to_level(editor_state_from_level(level))


class TestBlockResources:
    """Ore counts, where zero and "unset" are different instructions."""

    def test_depleted_ore_tile_stays_depleted(self) -> None:
        """Zeros mean the world starts with nothing, not "pick a default"."""
        block_map = _dirt(3, 3)
        block_map[1, 1] = int(BlockType.IRON)
        level = Level(
            name="depleted",
            map_width=3,
            map_height=3,
            block_map=block_map,
            block_resources=np.zeros((3, 3), dtype=np.int32),
        )

        restored = editor_state_from_level(_round_trip(level))

        assert int(restored.block_resources[1, 1]) == 0

    def test_partly_depleted_ore_keeps_its_count(self) -> None:
        block_map = _dirt(3, 3)
        block_map[1, 1] = int(BlockType.IRON)
        resources = np.zeros((3, 3), dtype=np.int32)
        resources[1, 1] = 17
        level = Level(
            name="partial",
            map_width=3,
            map_height=3,
            block_map=block_map,
            block_resources=resources,
        )

        restored = editor_state_from_level(_round_trip(level))

        assert int(restored.block_resources[1, 1]) == 17

    def test_unset_resources_still_take_the_default(self) -> None:
        """``None`` keeps its own meaning: let the engine decide."""
        block_map = _dirt(3, 3)
        block_map[1, 1] = int(BlockType.IRON)
        level = Level(
            name="unset",
            map_width=3,
            map_height=3,
            block_map=block_map,
            block_resources=None,
        )

        restored = editor_state_from_level(level)

        assert int(restored.block_resources[1, 1]) > 0


class TestMachineInventory:
    """Machine contents, which the level records by item and not by slot."""

    def test_an_item_entered_twice_holds_one_count(self) -> None:
        """The level records an item once, so the last entry wins.

        A machine cannot hold the same item in two slots and save it: the
        level format indexes contents by item. The editor therefore keeps one
        entry per item, and shows what a save keeps.
        """
        state = new_editor_state(2, 2)
        state.machine_types[0, 0] = int(Machine.FURNACE)
        target = ("machine", 0, 0)
        set_inventory_slot(state, target, 0, int(ItemType.COAL), 3)
        set_inventory_slot(state, target, 1, int(ItemType.COAL), 5)

        restored = editor_state_from_level(editor_state_to_level(state))

        assert int(restored.machine_inventory[0, 0, int(ItemType.COAL)]) == 5

    def test_an_item_entered_twice_leaves_one_slot_filled(self) -> None:
        state = new_editor_state(2, 2)
        state.machine_types[0, 0] = int(Machine.FURNACE)
        target = ("machine", 0, 0)
        set_inventory_slot(state, target, 0, int(ItemType.COAL), 3)
        set_inventory_slot(state, target, 1, int(ItemType.COAL), 5)

        filled = [s for s in get_inventory_slots(state, target) if s[1] > 0]

        assert filled == [(int(ItemType.COAL), 5)]

    def test_more_items_than_slots_survive(self) -> None:
        """The level format holds them, so the editor must not drop them."""
        pouch = np.zeros((1, 1, NUM_ITEM_TYPES), dtype=np.int32)
        wanted = (
            ItemType.COAL,
            ItemType.IRON_ORE,
            ItemType.COPPER_ORE,
            ItemType.TIN_ORE,
        )
        for item in wanted:
            pouch[0, 0, int(item)] = 7
        level = Level(
            name="full",
            map_width=1,
            map_height=1,
            block_map=_dirt(1, 1),
            machine_inventory=pouch,
        )

        restored = _round_trip(level)

        assert restored.machine_inventory is not None
        for item in wanted:
            assert int(restored.machine_inventory[0, 0, int(item)]) == 7

    def test_empty_machine_inventory_round_trips(self) -> None:
        state = new_editor_state(2, 2)
        state.machine_types[0, 0] = int(Machine.FURNACE)

        restored = editor_state_to_level(state)

        assert restored.machine_inventory is None


class TestAliasing:
    """A load must copy, so an edit cannot reach back into the source."""

    def test_player_inventory_is_copied(self) -> None:
        shared = [(int(ItemType.COAL), 5)]
        level = Level(
            name="alias",
            map_width=2,
            map_height=2,
            block_map=_dirt(2, 2),
            player_inventory=shared,
        )

        state = editor_state_from_level(level)

        assert state.player_inventory is not level.player_inventory

    def test_editing_the_editor_leaves_the_level_alone(self) -> None:
        shared = [(int(ItemType.COAL), 5)]
        level = Level(
            name="alias",
            map_width=2,
            map_height=2,
            block_map=_dirt(2, 2),
            player_inventory=shared,
        )

        state = editor_state_from_level(level)
        assert state.player_inventory is not None
        state.player_inventory.append((int(ItemType.IRON_ORE), 1))

        assert level.player_inventory == [(int(ItemType.COAL), 5)]


class TestPlayerNumbering:
    """Removing a player closes the gap, and the contents must follow."""

    @pytest.fixture
    def three_players(self):
        state = new_editor_state(5, 5)
        state.player_positions = {0: (0, 0), 1: (1, 1), 2: (2, 2)}
        state.player_inventories = {
            0: [(int(ItemType.COAL), 1)],
            1: [(int(ItemType.COAL), 2)],
            2: [(int(ItemType.COAL), 3)],
        }
        return state

    def test_remaining_player_keeps_its_own_inventory(self, three_players) -> None:
        """The player at (2, 2) must not inherit the deleted player's items."""
        remove_player_at(three_players, 1, 1)

        restored = editor_state_from_level(editor_state_to_level(three_players))
        moved = next(i for i, p in restored.player_positions.items() if p == (2, 2))

        assert restored.player_inventories[moved] == [(int(ItemType.COAL), 3)]

    def test_no_inventory_is_left_orphaned(self, three_players) -> None:
        remove_player_at(three_players, 1, 1)

        restored = editor_state_from_level(editor_state_to_level(three_players))

        assert set(restored.player_inventories) <= set(restored.player_positions)

    def test_players_are_numbered_without_a_gap(self, three_players) -> None:
        remove_player_at(three_players, 1, 1)

        restored = editor_state_from_level(editor_state_to_level(three_players))

        assert sorted(restored.player_positions) == [0, 1]


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


# -------------------------------------------------------------------------
# The populated round trip, and the field-coverage guard
# -------------------------------------------------------------------------


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
        player_inventory=player_inventory,
    )


class TestEditorRoundTrip:
    """Level -> EditorState -> Level must preserve every field."""

    def test_round_trip_preserves_all_fields(self) -> None:
        """Convert to editor state and back, then compare field by field."""
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

        assert restored.player_inventory == original.player_inventory

    def test_none_fields_stay_none(self) -> None:
        """A Level with every optional field as None round-trips cleanly."""
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
        assert restored.player_inventory is None


class TestFieldCoverage:
    """Structural check: every Level field is represented in EditorState."""

    # Fields that exist on Level but have no EditorState counterpart by
    # design. Dimensions are stored as plain ints, and block_map is
    # always present.
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

        from factoriax.playground.editor.state import EditorState

        level_fields = {
            f.name for f in dataclasses.fields(Level) if f.name not in self._EXCLUDED
        }
        editor_fields = {f.name for f in dataclasses.fields(EditorState)}

        missing = level_fields - editor_fields
        assert not missing, (
            f"Level fields {missing} have no EditorState counterpart. "
            f"Add them to EditorState and update the conversion functions."
        )


# -------------------------------------------------------------------------
# A level's machine inventories reach the entity arrays
# -------------------------------------------------------------------------


def _make_level_with_pallet_items() -> Level:
    """Create a Level with a pallet containing iron ore."""
    h, w = 5, 5
    block_map = np.full((h, w), int(BlockType.DIRT), dtype=np.int32)

    machine_types = np.full(
        (h, w),
        int(Machine.NONE),
        dtype=np.int32,
    )
    machine_types[2, 2] = int(Machine.PALLET)

    machine_inv = np.zeros((h, w, NUM_ITEM_TYPES), dtype=np.int32)
    machine_inv[2, 2, int(ItemType.IRON_ORE)] = 10

    return Level(
        name="test_pallet",
        map_width=w,
        map_height=h,
        block_map=block_map,
        machine_types=machine_types,
        machine_inventory=machine_inv,
        player_positions=[(0, 0)],
    )


class TestEditorInventoryInit:
    """Machine inventories from the editor must appear in EnvState."""

    def test_pallet_items_go_to_buffer(self) -> None:
        """Items in a pallet's inventory populate the buffer."""
        level = _make_level_with_pallet_items()
        state = build_state(level, num_players=1)

        eidx = int(state.tile_entity[2, 2])
        assert eidx >= 0, "Pallet entity not found"
        assert int(state.ent_buf_type[eidx]) == int(
            ItemType.IRON_ORE,
        ), f"Expected buf_type=IRON_ORE, got {int(state.ent_buf_type[eidx])}"
        assert int(state.ent_buf_count[eidx]) == 10, (
            f"Expected buf_count=10, got {int(state.ent_buf_count[eidx])}"
        )
