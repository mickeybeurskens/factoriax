"""Tests for the scroll system primitives and scrollable menus in ui.py."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from factoriax.play.ui import (
    SCROLL_STEP,
    ClickRegion,
    blit_scroll_view,
    clip_scroll_offset,
    render_achievement_menu,
    render_inventory_menu,
    scroll_adjust_regions,
)

_MAP = jnp.zeros((4, 4), dtype=jnp.int32)


# ---------------------------------------------------------------------------
# clip_scroll_offset
# ---------------------------------------------------------------------------


class TestClipScrollOffset:
    def test_zero_when_content_fits(self) -> None:
        """No scrolling possible when content is shorter than viewport."""
        assert clip_scroll_offset(0, content_h=100, viewport_h=200) == 0

    def test_clamps_to_zero_below(self) -> None:
        """Negative offsets are clamped to 0."""
        assert clip_scroll_offset(-10, content_h=200, viewport_h=100) == 0

    def test_clamps_to_max_above(self) -> None:
        """Offsets beyond the scroll range are clamped to content_h - viewport_h."""
        assert clip_scroll_offset(9999, content_h=300, viewport_h=100) == 200

    def test_valid_midpoint_passes_through(self) -> None:
        """Offsets within range are returned unchanged."""
        assert clip_scroll_offset(50, content_h=300, viewport_h=100) == 50

    def test_exact_max(self) -> None:
        """The maximum valid offset is returned as-is."""
        assert clip_scroll_offset(200, content_h=300, viewport_h=100) == 200

    def test_equal_content_and_viewport(self) -> None:
        """When content equals viewport height, any offset clamps to 0."""
        assert clip_scroll_offset(10, content_h=100, viewport_h=100) == 0

    def test_scroll_step_constant_positive(self) -> None:
        """SCROLL_STEP is a positive integer."""
        assert isinstance(SCROLL_STEP, int)
        assert SCROLL_STEP > 0


# ---------------------------------------------------------------------------
# blit_scroll_view
# ---------------------------------------------------------------------------


class TestBlitScrollView:
    def _make_overlay(self, h: int = 200, w: int = 100) -> np.ndarray:
        return np.zeros((h, w, 4), dtype=np.uint8)

    def _make_content(self, h: int, w: int, fill: int = 200) -> np.ndarray:
        c = np.zeros((h, w, 4), dtype=np.uint8)
        c[:, :, :3] = fill
        c[:, :, 3] = 255
        return c

    def test_no_scrollbar_when_content_fits(self) -> None:
        """When content fits in viewport, no scrollbar column is drawn."""
        overlay = self._make_overlay(100, 60)
        content = self._make_content(50, 60)
        blit_scroll_view(
            overlay, content, vp_x=0, vp_y=0, vp_w=60, vp_h=100, scroll_offset=0
        )
        # Right edge should retain content colour, not scrollbar background
        assert overlay[25, 59, 3] == 255

    def test_scrollbar_appears_when_content_overflows(self) -> None:
        """A scrollbar column is drawn on the right when content exceeds viewport."""
        from factoriax.ui.theme import SCROLLBAR_W as _SCROLLBAR_W

        overlay = self._make_overlay(100, 60)
        content = self._make_content(300, 60, fill=10)
        blit_scroll_view(
            overlay, content, vp_x=0, vp_y=0, vp_w=60, vp_h=100, scroll_offset=0
        )
        bar_x = 60 - _SCROLLBAR_W
        # Scrollbar background should be present (non-zero alpha)
        assert overlay[50, bar_x, 3] > 0

    def test_scroll_offset_shifts_visible_content(self) -> None:
        """Scrolling down reveals lower content rows."""
        overlay = self._make_overlay(100, 60)
        # Two-colour content: top half red, bottom half green
        content = np.zeros((200, 60, 4), dtype=np.uint8)
        content[:100, :, 0] = 200  # red
        content[:100, :, 3] = 255
        content[100:, :, 1] = 200  # green
        content[100:, :, 3] = 255

        blit_scroll_view(
            overlay, content, vp_x=0, vp_y=0, vp_w=60, vp_h=100, scroll_offset=100
        )
        assert overlay[0, 0, 1] == 200  # green channel now visible at top
        assert overlay[0, 0, 0] == 0  # red scrolled out of view

    def test_viewport_offset_applied_correctly(self) -> None:
        """Content is blitted to the correct screen position."""
        overlay = self._make_overlay(200, 200)
        content = self._make_content(50, 60)
        blit_scroll_view(
            overlay, content, vp_x=40, vp_y=80, vp_w=60, vp_h=100, scroll_offset=0
        )
        assert overlay[80, 40, 3] == 255   # inside viewport — has content
        assert overlay[79, 40, 3] == 0    # above viewport — untouched

    def test_scrollbar_thumb_at_top_when_offset_zero(self) -> None:
        """Scrollbar thumb starts at the top when scroll_offset is 0."""
        from factoriax.ui.theme import (
            SCROLLBAR_THUMB,
            SCROLLBAR_W,
        )

        overlay = self._make_overlay(100, 60)
        content = self._make_content(300, 60, fill=10)
        blit_scroll_view(
            overlay, content, vp_x=0, vp_y=0, vp_w=60, vp_h=100, scroll_offset=0
        )
        bar_x = 60 - SCROLLBAR_W
        assert tuple(overlay[0, bar_x]) == SCROLLBAR_THUMB

    def test_scrollbar_thumb_at_bottom_when_fully_scrolled(self) -> None:
        """Scrollbar thumb sits at the bottom when fully scrolled."""
        from factoriax.ui.theme import (
            SCROLLBAR_THUMB,
            SCROLLBAR_W,
        )

        overlay = self._make_overlay(100, 60)
        content = self._make_content(200, 60, fill=10)
        blit_scroll_view(
            overlay, content, vp_x=0, vp_y=0, vp_w=60, vp_h=100, scroll_offset=100
        )
        bar_x = 60 - SCROLLBAR_W
        assert tuple(overlay[99, bar_x]) == SCROLLBAR_THUMB


# ---------------------------------------------------------------------------
# scroll_adjust_regions
# ---------------------------------------------------------------------------


class TestScrollAdjustRegions:
    def _region(
        self,
        y: int,
        h: int = 20,
        action: str = "select_recipe",
        param: int = 0,
    ) -> ClickRegion:
        return ClickRegion(x=0, y=y, w=80, h=h, action=action, param=param)

    def test_translates_y_by_viewport_offset(self) -> None:
        """Region y is shifted by vp_y minus scroll_offset."""
        regions = [self._region(y=0)]
        result = scroll_adjust_regions(
            regions, vp_x=10, vp_y=50, vp_h=200, scroll_offset=0
        )
        assert len(result) == 1
        assert result[0].y == 50

    def test_translates_x_by_viewport_x(self) -> None:
        """Region x is shifted by vp_x."""
        r = ClickRegion(x=5, y=0, w=80, h=20, action="select_recipe", param=0)
        result = scroll_adjust_regions(
            [r], vp_x=20, vp_y=0, vp_h=200, scroll_offset=0
        )
        assert result[0].x == 25

    def test_scroll_offset_reduces_screen_y(self) -> None:
        """Scrolling down moves content y upward on screen."""
        regions = [self._region(y=100)]
        result = scroll_adjust_regions(
            regions, vp_x=0, vp_y=50, vp_h=200, scroll_offset=80
        )
        assert result[0].y == 70  # 100 - 80 + 50

    def test_filters_out_regions_above_viewport(self) -> None:
        """Regions entirely above the viewport top are excluded."""
        regions = [self._region(y=0, h=10)]
        result = scroll_adjust_regions(
            regions, vp_x=0, vp_y=0, vp_h=200, scroll_offset=50
        )
        assert len(result) == 0

    def test_filters_out_regions_below_viewport(self) -> None:
        """Regions entirely below the viewport bottom are excluded."""
        regions = [self._region(y=300, h=20)]
        result = scroll_adjust_regions(
            regions, vp_x=0, vp_y=0, vp_h=100, scroll_offset=0
        )
        assert len(result) == 0

    def test_preserves_action_and_param(self) -> None:
        """Action and param fields are preserved through the transform."""
        r = ClickRegion(x=0, y=0, w=80, h=20, action="select_recipe", param=3)
        result = scroll_adjust_regions(
            [r], vp_x=0, vp_y=0, vp_h=200, scroll_offset=0
        )
        assert result[0].action == "select_recipe"
        assert result[0].param == 3

    def test_empty_input_returns_empty(self) -> None:
        """Empty input list returns empty list."""
        result = scroll_adjust_regions(
            [], vp_x=0, vp_y=0, vp_h=200, scroll_offset=0
        )
        assert result == []

    def test_partially_visible_region_included(self) -> None:
        """A region that overlaps the viewport boundary is kept."""
        regions = [self._region(y=0, h=20)]  # screen_y = -15, bottom = 5
        result = scroll_adjust_regions(
            regions, vp_x=0, vp_y=0, vp_h=200, scroll_offset=15
        )
        assert len(result) == 1


# ---------------------------------------------------------------------------
# render_achievement_menu — scroll integration
# ---------------------------------------------------------------------------


class TestRenderAchievementMenuScroll:
    def test_renders_without_error_at_zero_offset(self, state_factory) -> None:
        """Default zero offset renders successfully."""
        state = state_factory(world_map=_MAP)
        result = render_achievement_menu(state, 480, 480, scroll_offset=0)
        assert result.shape == (480, 480, 4)

    def test_renders_with_nonzero_offset(self, state_factory) -> None:
        """Non-zero scroll offset renders without error."""
        state = state_factory(world_map=_MAP)
        result = render_achievement_menu(state, 480, 480, scroll_offset=50)
        assert result.shape == (480, 480, 4)

    def test_scroll_changes_pixels(self, state_factory) -> None:
        """Scrolling the list produces a visually different frame."""
        state = state_factory(world_map=_MAP)
        frame0 = render_achievement_menu(state, 480, 480, scroll_offset=0)
        frame1 = render_achievement_menu(state, 480, 480, scroll_offset=SCROLL_STEP)
        assert not np.array_equal(frame0, frame1)

    def test_excessive_offset_clamped(self, state_factory) -> None:
        """An offset far beyond the content end is clamped; result is stable."""
        state = state_factory(world_map=_MAP)
        frame_huge = render_achievement_menu(state, 480, 480, scroll_offset=999999)
        frame_max = render_achievement_menu(state, 480, 480, scroll_offset=999998)
        assert np.array_equal(frame_huge, frame_max)


# ---------------------------------------------------------------------------
# render_inventory_menu — recipe scroll integration
# ---------------------------------------------------------------------------


class TestRenderInventoryMenuRecipeScroll:
    def test_renders_without_error(self, state_factory) -> None:
        """Inventory menu with crafting focus renders without error."""
        state = state_factory(world_map=_MAP)
        overlay, regions = render_inventory_menu(state, 480, 480, menu_focus="crafting")
        assert overlay.shape == (480, 480, 4)
        assert any(r.action == "select_recipe" for r in regions)

    def test_recipe_click_regions_within_screen_bounds(self, state_factory) -> None:
        """All recipe click regions returned are inside the screen."""
        state = state_factory(world_map=_MAP)
        _, regions = render_inventory_menu(state, 480, 480, menu_focus="crafting")
        for r in regions:
            if r.action == "select_recipe":
                assert 0 <= r.x < 480
                assert 0 <= r.y < 480
