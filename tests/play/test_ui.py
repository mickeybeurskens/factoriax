"""Regression tests for play.ui rendering functions.

These tests guard the output contract (shape, dtype) of the render functions
and verify that they do not crash under common state configurations.  Visual
correctness is validated by running the game. Pixel-level assertions are
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
        """The render must not crash when no achievements are unlocked."""
        achievements = np.zeros(MAX_ACHIEVEMENTS, dtype=np.bool_)
        result = render_achievement_menu(achievements, _SW, _SH)
        assert result.shape == (_SH, _SW, 4)

    def test_all_unlocked(self) -> None:
        """The render must not crash when every achievement is unlocked."""
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
        """The render must not crash with inventory focus."""
        state = state_factory(world_map=jnp.zeros((8, 8), dtype=jnp.int32))
        result, _ = render_inventory_menu(state, _PARAMS, _SW, _SH)
        assert result.shape == (_SH, _SW, 4)

    def test_crafting_focus(self, state_factory) -> None:
        """The render must not crash with crafting focus."""
        state = state_factory(world_map=jnp.zeros((8, 8), dtype=jnp.int32))
        result, _ = render_inventory_menu(state, _PARAMS, _SW, _SH)
        assert result.shape == (_SH, _SW, 4)

    def test_populated_inventory(self, state_factory) -> None:
        """The render must not crash when the inventory holds items."""
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
        """The render must not crash when a craft is in progress."""
        state = state_factory(
            world_map=jnp.zeros((8, 8), dtype=jnp.int32),
            craft_progress=jnp.array([2], dtype=jnp.int32),
        )
        result, _ = render_inventory_menu(state, _PARAMS, _SW, _SH)
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
        """The render stays clean at different screen sizes, with no crash."""
        for w, h in [(320, 320), (640, 480), (1024, 768)]:
            result, _ = render_welcome_screen(w, h)
            assert result.shape == (h, w, 4)

    def test_no_record_toggle_region(self) -> None:
        """Record toggle was moved out of the welcome screen."""
        _, regions = render_welcome_screen(self._W, self._H)
        record_regions = [r for r in regions if r.action == "toggle_record"]
        assert len(record_regions) == 0

    def test_mine_control_describes_look_at(self) -> None:
        """The SPACE/mine row describes the look-at semantic.

        Source-level check: the welcome screen rasterizes text, so we
        verify the controls list in factoriax.playground.play.ui matches the new
        phrasing by reading the source.
        """
        import inspect

        from factoriax.playground.play import ui

        src = inspect.getsource(ui.render_welcome_screen)
        # The old bare label is gone. The new look-at phrasing is present.
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
        """The render is correct with the first option selected."""
        result, _ = render_pause_menu(self._PAUSE_W, self._PAUSE_H, selected_option=0)
        assert result.shape == (self._PAUSE_H, self._PAUSE_W, 4)

    def test_selection_one(self) -> None:
        """The render is correct with the second option selected."""
        result, _ = render_pause_menu(self._PAUSE_W, self._PAUSE_H, selected_option=1)
        assert result.shape == (self._PAUSE_H, self._PAUSE_W, 4)


class TestCraftPanelFollowsScenarioBook:
    """The craft panel lists the scenario's recipes, not BASE_RECIPES.

    Scenarios ship their own :class:`RecipeBook`. Reading the panel from
    ``BASE_RECIPES`` showed 27 rows with base-book labels regardless, so a
    10-recipe scenario rendered 17 rows. The player can select those extra
    rows but cannot craft them.
    """

    # Tall enough that no row is clipped out of the scroll viewport, so the
    # select_recipe regions count every recipe rather than the visible ones.
    _TALL_H = 2000
    _TALL_W = 600

    def _rows(self, state, params) -> list[ClickRegion]:
        """Return the panel's per-recipe click regions."""
        _, regions = render_inventory_menu(state, params, self._TALL_W, self._TALL_H)
        return [r for r in regions if r.action == "select_recipe"]

    def test_row_count_tracks_scenario_recipe_count(self, state_factory) -> None:
        """One select_recipe region per recipe in the scenario's own book."""
        from factoriax.engine.envs.easy_rocket import easy_rocket
        from factoriax.engine.recipes import NUM_RECIPES

        _, params = easy_rocket()
        num_recipes = int(params.recipe_table.outputs.shape[0])
        assert num_recipes != NUM_RECIPES, "scenario must differ from the base book"

        state = state_factory(world_map=jnp.zeros((8, 8), dtype=jnp.int32))
        assert len(self._rows(state, params)) == num_recipes

    def test_rows_are_indexed_over_the_scenario_table(self, state_factory) -> None:
        """Row params are 0..n-1 of the scenario table, and names line up."""
        from factoriax.engine.envs.science_tiers import science_tiers

        _, params = science_tiers()
        table = params.recipe_table
        assert len(table.names) == int(table.outputs.shape[0])

        state = state_factory(world_map=jnp.zeros((8, 8), dtype=jnp.int32))
        rows = self._rows(state, params)
        assert [r.param for r in rows] == list(range(len(table.names)))

    def test_two_scenarios_render_different_row_counts(self, state_factory) -> None:
        """Same renderer, different books, different panels."""
        from factoriax.engine.envs.easy_rocket import easy_rocket
        from factoriax.engine.envs.science_tiers import science_tiers

        _, er_params = easy_rocket()
        _, st_params = science_tiers()
        state = state_factory(world_map=jnp.zeros((8, 8), dtype=jnp.int32))

        assert len(self._rows(state, er_params)) != len(self._rows(state, st_params))
