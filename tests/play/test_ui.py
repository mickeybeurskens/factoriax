"""Regression tests for play.ui rendering functions.

These tests guard the output contract (shape, dtype) of the render functions
and verify that they do not crash under common state configurations.  Visual
correctness is validated by running the game — pixel-level assertions are
intentionally absent.
"""

import jax.numpy as jnp
import numpy as np

from factoriax.engine.constants import MAX_ACHIEVEMENTS, ItemType
from factoriax.engine.state import EnvParams
from factoriax.playground.play.ui import (
    ClickRegion,
    render_achievement_menu,
    render_inventory_menu,
    render_pause_menu,
    render_welcome_screen,
)

_SW = 128
_SH = 128
_PARAMS = EnvParams()


class TestRenderAchievementMenu:
    """Output-contract tests for render_achievement_menu."""

    def test_returns_uint8_rgba(self) -> None:
        """Must return a uint8 RGBA array matching the requested dimensions."""
        achievements = np.zeros(MAX_ACHIEVEMENTS, dtype=np.bool_)
        result = render_achievement_menu(achievements, _SW, _SH)
        assert result.dtype == np.uint8
        assert result.shape == (_SH, _SW, 4)

    def test_all_locked(self) -> None:
        """Should not crash when no achievements are unlocked."""
        achievements = np.zeros(MAX_ACHIEVEMENTS, dtype=np.bool_)
        result = render_achievement_menu(achievements, _SW, _SH)
        assert result.shape == (_SH, _SW, 4)

    def test_all_unlocked(self) -> None:
        """Should not crash when every achievement is unlocked."""
        achievements = np.ones(MAX_ACHIEVEMENTS, dtype=np.bool_)
        result = render_achievement_menu(achievements, _SW, _SH)
        assert result.shape == (_SH, _SW, 4)


class TestRenderInventoryMenu:
    """Output-contract tests for render_inventory_menu."""

    def test_returns_uint8_rgba(self, state_factory) -> None:
        """Must return a uint8 RGBA array matching the requested dimensions."""
        state = state_factory(world_map=jnp.zeros((8, 8), dtype=jnp.int32))
        result, click_regions = render_inventory_menu(state, _PARAMS, _SW, _SH)
        assert result.dtype == np.uint8
        assert result.shape == (_SH, _SW, 4)
        assert isinstance(click_regions, list)
        assert all(isinstance(r, ClickRegion) for r in click_regions)

    def test_inventory_focus(self, state_factory) -> None:
        """Should not crash with inventory focus."""
        state = state_factory(world_map=jnp.zeros((8, 8), dtype=jnp.int32))
        result, _ = render_inventory_menu(
            state, _PARAMS, _SW, _SH, menu_focus="inventory"
        )
        assert result.shape == (_SH, _SW, 4)

    def test_crafting_focus(self, state_factory) -> None:
        """Should not crash with crafting focus."""
        state = state_factory(world_map=jnp.zeros((8, 8), dtype=jnp.int32))
        result, _ = render_inventory_menu(
            state, _PARAMS, _SW, _SH, menu_focus="crafting"
        )
        assert result.shape == (_SH, _SW, 4)

    def test_populated_inventory(self, state_factory) -> None:
        """Should not crash when inventory contains items."""
        from factoriax.engine.constants import NUM_ITEM_TYPES

        inv = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int32)
        inv = inv.at[0, int(ItemType.COAL)].set(5)
        inv = inv.at[0, int(ItemType.IRON_ORE)].set(3)
        inv = inv.at[0, int(ItemType.COPPER_ORE)].set(12)
        state = state_factory(
            world_map=jnp.zeros((8, 8), dtype=jnp.int32),
            player_inventory=inv,
        )
        result, _ = render_inventory_menu(state, _PARAMS, _SW, _SH)
        assert result.shape == (_SH, _SW, 4)

    def test_craft_in_progress(self, state_factory) -> None:
        """Should not crash when a craft is in progress."""
        state = state_factory(
            world_map=jnp.zeros((8, 8), dtype=jnp.int32),
            craft_progress=jnp.array([2], dtype=jnp.int32),
        )
        result, _ = render_inventory_menu(
            state, _PARAMS, _SW, _SH, menu_focus="crafting"
        )
        assert result.shape == (_SH, _SW, 4)


class TestRenderWelcomeScreen:
    """Output-contract tests for render_welcome_screen."""

    _W = 480
    _H = 480

    def test_returns_uint8_rgba(self) -> None:
        """Must return a uint8 RGBA array matching the requested dimensions."""
        result, regions = render_welcome_screen(self._W, self._H)
        assert result.dtype == np.uint8
        assert result.shape == (self._H, self._W, 4)
        assert isinstance(regions, list)

    def test_fully_opaque(self) -> None:
        """Welcome screen must be fully opaque (alpha=255 everywhere)."""
        result, _ = render_welcome_screen(self._W, self._H)
        assert np.all(result[:, :, 3] == 255)

    def test_not_all_black(self) -> None:
        """At least some pixels must be non-black (panel and text are visible)."""
        result, _ = render_welcome_screen(self._W, self._H)
        assert np.any(result[:, :, :3] > 20)

    def test_various_sizes(self) -> None:
        """Should render cleanly at different screen sizes without crashing."""
        for w, h in [(320, 320), (640, 480), (1024, 768)]:
            result, _ = render_welcome_screen(w, h)
            assert result.shape == (h, w, 4)

    def test_no_record_toggle_region(self) -> None:
        """Record toggle was moved out of the welcome screen."""
        _, regions = render_welcome_screen(self._W, self._H)
        record_regions = [r for r in regions if r.action == "toggle_record"]
        assert len(record_regions) == 0

    def test_mine_control_describes_look_at(self) -> None:
        """The SPACE/mine row should describe the look-at semantic.

        Source-level check: the welcome screen rasterizes text, so we
        verify the controls list in factoriax.playground.play.ui matches the new
        phrasing by reading the source.
        """
        import inspect

        from factoriax.playground.play import ui

        src = inspect.getsource(ui.render_welcome_screen)
        # Old bare label is gone; new look-at phrasing is present.
        assert '"Mine ore"' not in src, "Stale 'Mine ore' label still present"
        assert '"Mine the tile you face"' in src or "look at" in src.lower()


class TestRenderPauseMenu:
    """Output-contract tests for render_pause_menu."""

    _PAUSE_W = 480
    _PAUSE_H = 480

    def test_returns_uint8_rgba(self) -> None:
        """Must return a uint8 RGBA array matching the requested dimensions."""
        result, click_regions = render_pause_menu(self._PAUSE_W, self._PAUSE_H)
        assert result.dtype == np.uint8
        assert result.shape == (self._PAUSE_H, self._PAUSE_W, 4)
        assert isinstance(click_regions, list)
        assert len(click_regions) == 4
        assert all(isinstance(r, ClickRegion) for r in click_regions)

    def test_selection_zero(self) -> None:
        """Should render correctly with first option selected."""
        result, _ = render_pause_menu(self._PAUSE_W, self._PAUSE_H, selected_option=0)
        assert result.shape == (self._PAUSE_H, self._PAUSE_W, 4)

    def test_selection_one(self) -> None:
        """Should render correctly with second option selected."""
        result, _ = render_pause_menu(self._PAUSE_W, self._PAUSE_H, selected_option=1)
        assert result.shape == (self._PAUSE_H, self._PAUSE_W, 4)
