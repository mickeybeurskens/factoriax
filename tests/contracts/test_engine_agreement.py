"""Tests that the play UI constants agree with the engine enums and tables.

The engine leads. Every item id, item name, and placement action the play UI
holds must come from :mod:`factoriax.engine`, because a new member in an
earlier item category shifts the value of every later item. A literal that was
correct when it was written goes silently wrong after such a change, so these
tests compare the UI constants against the engine instead of against a second
literal.
"""

from __future__ import annotations

from factoriax.engine.actions import ITEM_TO_PLACE_ACTION, NO_ACTION
from factoriax.engine.constants import PLACEABLE_ITEM_LIST, ItemType
from factoriax.playground.play.game_ui import _ITEM_TO_PLACE_ACTION, _SLOT_ACTIONS
from factoriax.playground.play.play_state import PlayState
from factoriax.playground.play.ui import _ITEM_NAMES


class TestDefaultSelectedItem:
    """The item a new session starts with."""

    def test_default_selected_item_is_the_first_placeable(self) -> None:
        assert PlayState().selected_item == int(PLACEABLE_ITEM_LIST[0])

    def test_default_selected_item_is_placeable(self) -> None:
        """A default that has no PLACE action makes the place key do nothing."""
        action = int(ITEM_TO_PLACE_ACTION[PlayState().selected_item])
        assert action != NO_ACTION


class TestItemNames:
    """Display names the play UI draws for items."""

    def test_every_item_has_a_name(self) -> None:
        """A missing entry renders as an empty label, not as a visible error."""
        missing = [
            m.name for m in ItemType if m.name != "EMPTY" and int(m) not in _ITEM_NAMES
        ]
        assert missing == []

    def test_every_placeable_has_a_name(self) -> None:
        for item in PLACEABLE_ITEM_LIST:
            assert _ITEM_NAMES.get(int(item))

    def test_no_name_is_empty(self) -> None:
        assert [k for k, v in _ITEM_NAMES.items() if not v.strip()] == []

    def test_empty_is_not_named(self) -> None:
        assert int(ItemType.EMPTY) not in _ITEM_NAMES


class TestPlacement:
    """Every hotbar pocket must place the machine it draws."""

    def test_every_placeable_item_has_a_place_action(self) -> None:
        """A pocket with no action is selectable but silently does nothing."""
        for item in PLACEABLE_ITEM_LIST:
            assert _ITEM_TO_PLACE_ACTION.get(int(item)) is not None

    def test_place_actions_match_the_engine(self) -> None:
        for item, action in _ITEM_TO_PLACE_ACTION.items():
            assert action == int(ITEM_TO_PLACE_ACTION[item])

    def test_no_extra_items_are_placeable(self) -> None:
        assert set(_ITEM_TO_PLACE_ACTION) == {int(i) for i in PLACEABLE_ITEM_LIST}


class TestNumberKeys:
    """Number keys select hotbar pockets."""

    def test_every_pocket_has_a_number_key(self) -> None:
        """A pocket past the last bound key cannot be reached from the keyboard."""
        assert len(_SLOT_ACTIONS) == len(PLACEABLE_ITEM_LIST)

    def test_number_keys_follow_hotbar_order(self) -> None:
        assert list(_SLOT_ACTIONS.values()) == [int(i) for i in PLACEABLE_ITEM_LIST]

    def test_every_number_key_selects_a_placeable_item(self) -> None:
        for item in _SLOT_ACTIONS.values():
            assert _ITEM_TO_PLACE_ACTION.get(item) is not None
