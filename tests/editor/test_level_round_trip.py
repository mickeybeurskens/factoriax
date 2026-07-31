"""Tests that an edit survives a trip through :class:`Level` unchanged.

The editor saves by converting its state to a ``Level`` and loads by converting
back. A field that changes meaning on the way is a silent edit to the level, so
these tests drive the conversion in both directions and compare.

The existing round trip test covers a level with every field populated. These
cover the edges that the populated case hides: a zero that means "nothing", an
item that appears twice, more items than a machine shows, a shared list, and a
gap in the player numbering.
"""

from __future__ import annotations

import numpy as np
import pytest

from factoriax.engine.constants import (
    NUM_ITEM_TYPES,
    BlockType,
    ItemType,
    Machine,
)
from factoriax.engine.levels import Level
from factoriax.playground.editor.state import (
    editor_state_from_level,
    editor_state_to_level,
    get_inventory_slots,
    new_editor_state,
    remove_player_at,
    set_inventory_slot,
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
        entry per item, and shows what a save would keep.
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
