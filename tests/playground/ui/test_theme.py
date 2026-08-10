"""Tests for :mod:`factoriax.playground.ui.theme`.

``apply_scale`` rewrites the module's size constants in place. Every UI
module reads those constants at draw time, so the scale factor reaches the
whole playground through this one call.
"""

from __future__ import annotations

import pytest

from factoriax.playground.ui import theme


@pytest.fixture(autouse=True)
def _reset_theme_scale() -> None:
    """Reset the theme scale to 1 after each test."""
    yield  # type: ignore[misc]
    theme.apply_scale(1)


class TestApplyScale:
    """Tests for theme.apply_scale updating size constants."""

    def test_scale_1_keeps_defaults(self) -> None:
        """Scale factor 1 leaves constants at base values."""
        theme.apply_scale(1)
        assert theme.UI_SCALE == 1
        assert theme.FONT_HEADER == 24
        assert theme.FONT_BODY == 18
        assert theme.BORDER_PX == 4

    def test_scale_2_doubles_sizes(self) -> None:
        """Scale factor 2 doubles all size constants."""
        theme.apply_scale(2)
        assert theme.UI_SCALE == 2
        assert theme.FONT_HEADER == 48
        assert theme.FONT_BODY == 36
        assert theme.FONT_HINT == 24
        assert theme.HEADER_H == 88
        assert theme.BORDER_PX == 8
        assert theme.SCROLLBAR_W == 16
