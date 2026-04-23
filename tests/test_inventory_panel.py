"""Tests for :mod:`factoriax.analysis.inventory`.

The load-bearing invariant is **every non-EMPTY ItemType always gets a
slot in the rendered panel**. This held in practice before, then broke
silently once new rocket-chain items pushed past the single-column
height budget on the 320x240 quadrant. These tests lock the invariant
down at a range of aspect ratios so a future item addition that doesn't
fit can't slip through unnoticed.
"""

from __future__ import annotations

import numpy as np
import pytest

from factoriax.analysis.inventory import (
    INVENTORY_ITEMS,
    _inventory_slot_positions,
    render_inventory_panel,
)
from factoriax.constants import ITEM_COLORS, NUM_ITEM_TYPES, ItemType


class TestInventoryItemsConstant:
    """The ``INVENTORY_ITEMS`` tuple is the public layout contract."""

    def test_includes_every_non_empty_item(self) -> None:
        """Contract: one entry per ItemType, EMPTY excluded."""
        expected = {it for it in ItemType if it != ItemType.EMPTY}
        assert set(INVENTORY_ITEMS) == expected

    def test_stable_order_by_enum_value(self) -> None:
        """Order is enum value ascending — a layout guarantee, not an accident."""
        values = [int(it) for it in INVENTORY_ITEMS]
        assert values == sorted(values)

    def test_excludes_empty(self) -> None:
        """EMPTY (enum value 0) must never be in the panel."""
        assert ItemType.EMPTY not in INVENTORY_ITEMS


class TestSlotPositionsAlwaysFits:
    """Layout helper must place every item within bounds at realistic sizes."""

    @pytest.mark.parametrize(
        "width,height",
        [
            (320, 240),  # default debugger quadrant
            (160, 320),  # tall narrow column (eval-video panel shape)
            (200, 150),  # MIN_QUADRANT_W x MIN_QUADRANT_H (smallest debugger)
            (400, 500),  # oversized, forces single-column path
            (120, 800),  # extreme single-column case
        ],
    )
    def test_every_item_placed_inside_panel(self, width: int, height: int) -> None:
        """For any realistic panel size, every item lands inside bounds."""
        positions = _inventory_slot_positions(
            width=width,
            height=height,
            num_items=len(INVENTORY_ITEMS),
            row_top=25,
        )
        assert len(positions) == len(INVENTORY_ITEMS)
        for item, (x, y, w, h) in zip(INVENTORY_ITEMS, positions, strict=True):
            assert 0 <= x, f"{item.name}: x={x} out of bounds"
            assert 0 <= y, f"{item.name}: y={y} out of bounds"
            assert x + w <= width, f"{item.name}: x+w={x + w} > width={width}"
            assert y + h <= height, f"{item.name}: y+h={y + h} > height={height}"

    def test_single_column_layout_when_tall(self) -> None:
        """A tall panel picks the single-column layout."""
        positions = _inventory_slot_positions(
            width=320,
            height=800,
            num_items=len(INVENTORY_ITEMS),
            row_top=25,
        )
        xs = {x for (x, _y, _w, _h) in positions}
        assert len(xs) == 1, f"expected one column, got x-origins {sorted(xs)}"

    def test_two_column_layout_when_short(self) -> None:
        """A short panel falls back to the two-column layout."""
        positions = _inventory_slot_positions(
            width=320,
            height=240,
            num_items=len(INVENTORY_ITEMS),
            row_top=25,
        )
        xs = sorted({x for (x, _y, _w, _h) in positions})
        assert len(xs) == 2, f"expected two columns, got x-origins {xs}"

    def test_empty_input_returns_empty(self) -> None:
        """Zero items yields an empty list, not a crash."""
        assert (
            _inventory_slot_positions(width=320, height=240, num_items=0, row_top=25)
            == []
        )


class TestRenderInventoryPanelShape:
    """Trivial shape / dtype invariants of the rendered panel."""

    def test_output_shape(self) -> None:
        inv = np.zeros(NUM_ITEM_TYPES, dtype=np.int32)
        img = render_inventory_panel(inv, width=320, height=240)
        assert img.shape == (240, 320, 3)

    def test_output_dtype(self) -> None:
        inv = np.zeros(NUM_ITEM_TYPES, dtype=np.int32)
        img = render_inventory_panel(inv, width=200, height=150)
        assert img.dtype == np.uint8

    @pytest.mark.parametrize(
        "width,height",
        [(200, 150), (320, 240), (160, 320), (400, 500)],
    )
    def test_shape_matches_requested_dims(self, width: int, height: int) -> None:
        inv = np.zeros(NUM_ITEM_TYPES, dtype=np.int32)
        img = render_inventory_panel(inv, width=width, height=height)
        assert img.shape == (height, width, 3)


class TestRenderInventoryPanelAllItemsVisible:
    """The "always show all items" invariant, enforced at render time."""

    @pytest.mark.parametrize(
        "width,height",
        [
            (320, 240),  # two-column default
            (200, 150),  # smallest debugger quadrant
            (160, 320),  # eval-video tall strip
            (400, 500),  # single-column oversized
        ],
    )
    def test_every_item_renders_its_color_swatch(self, width: int, height: int) -> None:
        """Every non-EMPTY item's signature RGB appears in the panel when active.

        Setting every count to 1 means the active (full-brightness)
        swatch branch runs for every item. We then scan the panel for
        each item's exact swatch color — if an item was clipped or
        silently dropped, its color will be absent.
        """
        inv = np.ones(NUM_ITEM_TYPES, dtype=np.int32)
        img = render_inventory_panel(inv, width=width, height=height)

        missing: list[str] = []
        for item in INVENTORY_ITEMS:
            rgb = ITEM_COLORS.get(int(item), (120, 120, 120))
            match = np.all(img == np.asarray(rgb, dtype=np.uint8), axis=-1)
            if not bool(match.any()):
                missing.append(item.name)
        assert not missing, f"items missing from {width}x{height} panel: {missing}"

    def test_zero_inventory_still_shows_all_dimmed_swatches(self) -> None:
        """Even with an all-zero inventory, every item has a (dimmed) swatch.

        Dim color = ``tuple(c // 3 for c in rgb)`` per the renderer.
        That dim color must be present for every item, so the operator
        can still see the full item list at a glance.
        """
        inv = np.zeros(NUM_ITEM_TYPES, dtype=np.int32)
        img = render_inventory_panel(inv, width=320, height=240)

        missing: list[str] = []
        for item in INVENTORY_ITEMS:
            rgb = ITEM_COLORS.get(int(item), (120, 120, 120))
            dim = np.asarray([max(0, c // 3) for c in rgb], dtype=np.uint8)
            match = np.all(img == dim, axis=-1)
            if not bool(match.any()):
                missing.append(item.name)
        assert not missing, f"dimmed swatches missing for zero inventory: {missing}"


class TestRenderInventoryPanelContent:
    """Per-item content: count formatting and activity contrast."""

    def test_active_vs_zero_background_differs(self) -> None:
        """All-zero and all-one inventories produce visibly different panels.

        A defensive check: if counts were ignored entirely, the output
        would be bit-identical regardless of inventory contents.
        """
        img_zero = render_inventory_panel(
            np.zeros(NUM_ITEM_TYPES, dtype=np.int32), width=320, height=240
        )
        img_active = render_inventory_panel(
            np.ones(NUM_ITEM_TYPES, dtype=np.int32), width=320, height=240
        )
        assert not np.array_equal(img_zero, img_active)

    def test_shorter_inventory_array_is_tolerated(self) -> None:
        """Array shorter than NUM_ITEM_TYPES is read defensively as zero.

        Some recorded trajectories use trimmed inventory columns. The
        renderer must treat missing indices as zero rather than crash.
        """
        short_inv = np.zeros(5, dtype=np.int32)
        img = render_inventory_panel(short_inv, width=320, height=240)
        assert img.shape == (240, 320, 3)
