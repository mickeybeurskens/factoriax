"""Every placeable machine must have a renderable icon.

Catches the case where a new Machine is added to ITEM_TO_MACHINE
in constants.py but ``factoriax.playground.ui.icons.MACHINE_TO_ITEM`` or
``render_item_icon`` doesn't know about it. The arm was invisible on
the map because of exactly this kind of disconnect.
"""

from __future__ import annotations

import numpy as np
import pytest

from factoriax.engine.constants import ITEM_TO_MACHINE, Direction, ItemType, Machine
from factoriax.playground.ui.icons import MACHINE_TO_ITEM, render_item_icon


class TestMachineIconCoverage:
    """Every machine in ITEM_TO_MACHINE must render without error."""

    def test_all_machines_in_renderer_dict(self) -> None:
        """MACHINE_TO_ITEM covers every Machine from constants."""
        for item_type, machine_type in ITEM_TO_MACHINE.items():
            assert int(machine_type) in MACHINE_TO_ITEM, (
                f"Machine {machine_type!r} (from ItemType "
                f"{item_type!r}) missing from ui.icons.MACHINE_TO_ITEM"
            )

    def test_all_machines_render_icon(self) -> None:
        """render_item_icon produces a valid array for every machine."""
        for item_type in ITEM_TO_MACHINE:
            icon = render_item_icon(
                int(item_type),
                32,
                int(Direction.RIGHT),
            )
            assert icon.shape == (32, 32, 4), (
                f"ItemType {item_type!r}: bad icon shape {icon.shape}"
            )
            assert icon.dtype.name == "uint8"

    def test_machine_type_none_excluded(self) -> None:
        """Machine.NONE should not appear in the mapping."""
        assert int(Machine.NONE) not in MACHINE_TO_ITEM


class TestDirectionalIcons:
    """Icons for directional machines must reflect their facing."""

    def test_splitter_icon_changes_with_facing(self) -> None:
        """The splitter glyph rotates 90° between vertical-facing and
        horizontal-facing directions; the two icons must not be pixel-equal."""
        icon_up = render_item_icon(int(ItemType.SPLITTER), 24, int(Direction.UP))
        icon_left = render_item_icon(int(ItemType.SPLITTER), 24, int(Direction.LEFT))
        assert icon_up.shape == (24, 24, 4)
        assert not np.array_equal(icon_up, icon_left), (
            "Splitter icon did not change between UP and LEFT facings."
        )

    @pytest.mark.parametrize("encoding", [1, 2, 3, 4])
    def test_crossing_icon_renders_for_each_encoding(self, encoding: int) -> None:
        """All four crossing encodings produce a non-uniform icon (the
        diagonal stripe is drawn). Encoding 0 leaves the icon flat."""
        icon = render_item_icon(int(ItemType.CROSSING), 24, encoding)
        assert icon.shape == (24, 24, 4)
        # The diagonal must paint at least some pixels different from the
        # uniform fill colour.
        base_pixel = icon[icon.shape[0] // 2, 0, :3]
        assert not np.all(icon[..., :3] == base_pixel), (
            f"Crossing icon for encoding={encoding} is uniformly coloured — "
            "the diagonal stripe was not drawn."
        )

    def test_crossing_icon_diagonals_match_glyph_table(self) -> None:
        r"""Encodings 1+4 share the ``\`` diagonal; 2+3 share ``/``.

        Same-glyph encodings must render to pixel-identical icons (the
        visual is purely a function of the diagonal direction).
        """
        cr1 = render_item_icon(int(ItemType.CROSSING), 24, 1)
        cr2 = render_item_icon(int(ItemType.CROSSING), 24, 2)
        cr3 = render_item_icon(int(ItemType.CROSSING), 24, 3)
        cr4 = render_item_icon(int(ItemType.CROSSING), 24, 4)
        np.testing.assert_array_equal(cr1, cr4)  # both \
        np.testing.assert_array_equal(cr2, cr3)  # both /
        assert not np.array_equal(cr1, cr2), "Backslash and slash icons match."
