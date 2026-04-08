"""Tests for the UI scaling infrastructure.

Validates ScaledCanvas coordinate transforms and theme.apply_scale
without requiring a live pygame display.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pygame
import pytest

from factoriax.ui import theme
from factoriax.ui.scaling import ScaledCanvas


@pytest.fixture(autouse=True)
def _reset_theme_scale() -> None:
    """Reset theme scale to 1 after each test."""
    yield  # type: ignore[misc]
    theme.apply_scale(1)


# ---------------------------------------------------------------------------
# theme.apply_scale
# ---------------------------------------------------------------------------


class TestApplyScale:
    """Tests for theme.apply_scale updating size constants."""

    def test_scale_1_keeps_defaults(self) -> None:
        """Scale factor 1 leaves constants at base values."""
        theme.apply_scale(1)
        assert theme.UI_SCALE == 1
        assert theme.FONT_HEADER == 26
        assert theme.FONT_BODY == 20
        assert theme.BORDER_PX == 4

    def test_scale_2_doubles_sizes(self) -> None:
        """Scale factor 2 doubles all size constants."""
        theme.apply_scale(2)
        assert theme.UI_SCALE == 2
        assert theme.FONT_HEADER == 52
        assert theme.FONT_BODY == 40
        assert theme.FONT_HINT == 28
        assert theme.HEADER_H == 88
        assert theme.BORDER_PX == 8
        assert theme.SCROLLBAR_W == 16

    def test_scale_3_triples_sizes(self) -> None:
        """Scale factor 3 triples all size constants."""
        theme.apply_scale(3)
        assert theme.UI_SCALE == 3
        assert theme.FONT_HEADER == 78
        assert theme.FONT_BODY == 60

    def test_colors_unchanged(self) -> None:
        """Color constants are not affected by scaling."""
        original_border = theme.BORDER
        original_text = theme.TEXT_COLOR
        theme.apply_scale(2)
        assert theme.BORDER == original_border
        assert theme.TEXT_COLOR == original_text


# ---------------------------------------------------------------------------
# ScaledCanvas
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _init_pygame() -> None:
    """Initialise pygame display for surface creation."""
    pygame.display.init()
    pygame.display.set_mode((800, 600))
    yield  # type: ignore[misc]
    pygame.display.quit()


class TestScaledCanvasInit:
    """Tests for ScaledCanvas creation."""

    def test_canvas_size_at_scale_1(self) -> None:
        """Canvas is base_size * 1 at ui_scale=1."""
        screen = pygame.display.get_surface()
        canvas = ScaledCanvas(1024, 1, screen)
        assert canvas.width == 1024
        assert canvas.height == 1024

    def test_canvas_size_at_scale_2(self) -> None:
        """Canvas is base_size * 2 at ui_scale=2."""
        screen = pygame.display.get_surface()
        canvas = ScaledCanvas(1024, 2, screen)
        assert canvas.width == 2048
        assert canvas.height == 2048

    def test_surface_matches_dimensions(self) -> None:
        """The pygame surface has the correct size."""
        screen = pygame.display.get_surface()
        canvas = ScaledCanvas(1024, 1, screen)
        assert canvas.surface.get_size() == (1024, 1024)


class TestToCanvas:
    """Tests for ScaledCanvas.to_canvas coordinate transform."""

    def test_identity_at_scale_1(self) -> None:
        """At 1x window scale, coords adjust for centering offset."""
        screen = pygame.display.get_surface()
        # Window is 800x600, canvas is 800x800, scale=1.
        # ox = (800-800)//2 = 0, oy = (600-800)//2 = -100.
        canvas = ScaledCanvas(800, 1, screen)
        cx, cy = canvas.to_canvas(100, 100)
        # cx = (100 - 0) // 1 = 100, cy = (100 - (-100)) // 1 = 200
        assert cx == 100
        assert cy == 200

    def test_with_offset(self) -> None:
        """Canvas coords account for centering offset."""
        screen = pygame.display.get_surface()
        # 400x400 canvas in 800x600 window -> scale=1, ox=200, oy=100
        canvas = ScaledCanvas(400, 1, screen)
        cx, cy = canvas.to_canvas(200, 100)
        assert cx == 0
        assert cy == 0


class TestHandleResize:
    """Tests for ScaledCanvas.handle_resize."""

    def test_resize_updates_scale(self) -> None:
        """Doubling window size doubles the scale factor."""
        screen = pygame.display.get_surface()
        canvas = ScaledCanvas(400, 1, screen)
        # Resize to 1200x1200 -> scale = min(1200//400, 1200//400) = 3
        canvas.handle_resize(1200, 1200)
        cx, cy = canvas.to_canvas(600, 600)
        # ox = (1200 - 400*3)//2 = 0, scale=3
        assert cx == 200
        assert cy == 200
