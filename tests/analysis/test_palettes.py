"""Tests that the default label and color lists cover their enums.

Plot code indexes these lists by enum value. Entry ``i`` therefore
describes the action or item whose value is ``i``. A list shorter than
its enum leaves the entries past its end undefined.

Two failures follow from a list that does not line up. A short label
list raises ``IndexError``, or worse, names the wrong thing. A short
color list does not raise at all. Matplotlib gives every index past
the end of a colormap the same last color. Unrelated actions then look
identical, and nothing reports it.

Both lists were written by hand against an older enum and drifted.
These tests hold them to the enum instead.
"""

from __future__ import annotations

from factoriax.analysis.actions import (
    DEFAULT_ACTION_COLORS,
    DEFAULT_ACTION_LABELS,
    _get_action_cmap,
)
from factoriax.analysis.state import DEFAULT_ITEM_COLORS, DEFAULT_ITEM_LABELS
from factoriax.engine.constants import NUM_ACTIONS, NUM_ITEM_TYPES, Action, ItemType


class TestActionLabels:
    """``DEFAULT_ACTION_LABELS`` against the ``Action`` enum."""

    def test_covers_every_action(self) -> None:
        """The list holds one name for each action."""
        assert len(DEFAULT_ACTION_LABELS) == NUM_ACTIONS

    def test_entry_names_the_action_with_that_value(self) -> None:
        """Entry ``i`` is the name of the action whose value is ``i``."""
        for i in range(NUM_ACTIONS):
            assert DEFAULT_ACTION_LABELS[i] == Action(i).name


class TestActionColors:
    """``DEFAULT_ACTION_COLORS`` against the ``Action`` enum."""

    def test_covers_every_action(self) -> None:
        """The list holds one color for each action.

        A shorter list is what makes the colormap short, and a short
        colormap silently reuses its last color.
        """
        assert len(DEFAULT_ACTION_COLORS) == NUM_ACTIONS

    def test_colors_are_distinct(self) -> None:
        """No two actions share a color."""
        assert len(set(DEFAULT_ACTION_COLORS)) == len(DEFAULT_ACTION_COLORS)

    def test_default_colormap_separates_every_action(self) -> None:
        """A default colormap gives each action its own color.

        This is the failure a reader meets. With a short color list,
        ``cmap(i)`` clamps. Dozens of actions then render alike, in
        the raster and in its legend.
        """
        cmap = _get_action_cmap(NUM_ACTIONS)
        assert cmap.N == NUM_ACTIONS
        assert len({cmap(i) for i in range(NUM_ACTIONS)}) == NUM_ACTIONS


class TestItemLabels:
    """``DEFAULT_ITEM_LABELS`` against the ``ItemType`` enum."""

    def test_covers_every_item(self) -> None:
        """The list holds one name for each item type."""
        assert len(DEFAULT_ITEM_LABELS) == NUM_ITEM_TYPES

    def test_entry_names_the_item_with_that_value(self) -> None:
        """Entry ``i`` is the name of the item whose value is ``i``.

        The hand-written list had ``IRON``, ``COPPER``, and ``MINER``
        at ids 2, 3, and 4, where the engine has ``IRON_ORE``,
        ``COPPER_ORE``, and ``TIN_ORE``. A default inventory plot
        therefore labelled the tin ore line "MINER".
        """
        for i in range(NUM_ITEM_TYPES):
            assert DEFAULT_ITEM_LABELS[i] == ItemType(i).name


class TestItemColors:
    """``DEFAULT_ITEM_COLORS`` against the label list."""

    def test_covers_every_item(self) -> None:
        """The list holds one color for each item type."""
        assert len(DEFAULT_ITEM_COLORS) == NUM_ITEM_TYPES

    def test_colors_are_distinct(self) -> None:
        """No two items share a color."""
        assert len(set(DEFAULT_ITEM_COLORS)) == len(DEFAULT_ITEM_COLORS)
