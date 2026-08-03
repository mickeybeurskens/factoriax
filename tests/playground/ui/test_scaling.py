"""Tests for :mod:`factoriax.playground.ui.scaling`.

:class:`ScaledCanvas` maps a window coordinate to a canvas coordinate and
back. A resize changes the mapping, so the transform tests and the resize
tests share one surface.
"""

from __future__ import annotations

import pygame
import pytest

from factoriax.playground.ui import theme
from factoriax.playground.ui.scaling import ScaledCanvas


@pytest.fixture(autouse=True)
def _reset_theme_scale() -> None:
    """Reset theme scale to 1 after each test."""
    yield  # type: ignore[misc]
    theme.apply_scale(1)


# ---------------------------------------------------------------------------
# ScaledCanvas
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _scaling_display_surface(pygame_display: None) -> None:
    """Re-assert an 800x600 display surface for every scaling test.

    The root ``pygame_display`` fixture initialises the display and sets an
    800x600 surface. Other modules, and ``tests/play/test_launch_screen.py``
    above all, then call ``set_mode`` with other dimensions, and pygame keeps
    that surface for the rest of the session. The tests below hardcode the
    math for an 800x600 window, so this module re-asserts that size one time
    when it loads.
    """
    pygame.display.set_mode((800, 600))


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
