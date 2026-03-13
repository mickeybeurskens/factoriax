"""Regression tests for player_ui rendering functions.

These tests guard the output contract (shape, dtype) of the render functions
and verify that they do not crash under common state configurations.  Visual
correctness is validated by running the game — pixel-level assertions are
intentionally absent.
"""

import jax.numpy as jnp
import numpy as np
import pygame

from factoriax.achievements import NUM_ACHIEVEMENTS
from factoriax.constants import ItemType
from factoriax.player_ui import render_achievement_menu, render_inventory_menu


def setup_module(module: object) -> None:
    """Initialise the pygame font subsystem (no display required)."""
    pygame.font.init()


def teardown_module(module: object) -> None:
    """Shut down the font subsystem after all tests in this module."""
    pygame.font.quit()


_SW = 128
_SH = 128


class TestRenderAchievementMenu:
    """Output-contract tests for render_achievement_menu."""

    def test_returns_uint8_rgba(self, state_factory) -> None:
        """Must return a uint8 RGBA array matching the requested dimensions."""
        state = state_factory(world_map=jnp.zeros((8, 8), dtype=jnp.int32))
        result = render_achievement_menu(state, _SW, _SH)
        assert result.dtype == np.uint8
        assert result.shape == (_SH, _SW, 4)

    def test_all_locked(self, state_factory) -> None:
        """Should not crash when no achievements are unlocked."""
        state = state_factory(
            world_map=jnp.zeros((8, 8), dtype=jnp.int32),
            achievements_unlocked=jnp.zeros(NUM_ACHIEVEMENTS, dtype=jnp.bool_),
        )
        result = render_achievement_menu(state, _SW, _SH)
        assert result.shape == (_SH, _SW, 4)

    def test_all_unlocked(self, state_factory) -> None:
        """Should not crash when every achievement is unlocked."""
        state = state_factory(
            world_map=jnp.zeros((8, 8), dtype=jnp.int32),
            achievements_unlocked=jnp.ones(NUM_ACHIEVEMENTS, dtype=jnp.bool_),
        )
        result = render_achievement_menu(state, _SW, _SH)
        assert result.shape == (_SH, _SW, 4)


class TestRenderInventoryMenu:
    """Output-contract tests for render_inventory_menu."""

    def test_returns_uint8_rgba(self, state_factory) -> None:
        """Must return a uint8 RGBA array matching the requested dimensions."""
        state = state_factory(world_map=jnp.zeros((8, 8), dtype=jnp.int32))
        result = render_inventory_menu(state, _SW, _SH)
        assert result.dtype == np.uint8
        assert result.shape == (_SH, _SW, 4)

    def test_inventory_focus(self, state_factory) -> None:
        """Should not crash with inventory focus."""
        state = state_factory(world_map=jnp.zeros((8, 8), dtype=jnp.int32))
        result = render_inventory_menu(state, _SW, _SH, menu_focus="inventory")
        assert result.shape == (_SH, _SW, 4)

    def test_crafting_focus(self, state_factory) -> None:
        """Should not crash with crafting focus."""
        state = state_factory(world_map=jnp.zeros((8, 8), dtype=jnp.int32))
        result = render_inventory_menu(state, _SW, _SH, menu_focus="crafting")
        assert result.shape == (_SH, _SW, 4)

    def test_populated_inventory(self, state_factory) -> None:
        """Should not crash when inventory slots contain items."""
        inv_items = jnp.array(
            [[ItemType.COAL, ItemType.IRON, ItemType.COPPER, 0, 0, 0, 0, 0, 0, 0]],
            dtype=jnp.int32,
        )
        inv_counts = jnp.array(
            [[5, 3, 12, 0, 0, 0, 0, 0, 0, 0]],
            dtype=jnp.int32,
        )
        state = state_factory(
            world_map=jnp.zeros((8, 8), dtype=jnp.int32),
            inventory_items=inv_items,
            inventory_counts=inv_counts,
        )
        result = render_inventory_menu(state, _SW, _SH)
        assert result.shape == (_SH, _SW, 4)

    def test_craft_in_progress(self, state_factory) -> None:
        """Should not crash when a craft is in progress."""
        state = state_factory(
            world_map=jnp.zeros((8, 8), dtype=jnp.int32),
            craft_progress=jnp.array([2], dtype=jnp.int32),
        )
        result = render_inventory_menu(state, _SW, _SH, menu_focus="crafting")
        assert result.shape == (_SH, _SW, 4)
