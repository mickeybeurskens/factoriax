"""Shared UI primitives for all pygame-based FactoriaX applications.

This package provides the common building blocks used by the editor,
play mode, and inspector. Importing from here (rather than from
``play.ui`` or ``play.main``) keeps the dependency graph clean and
ensures consistent styling across all interactive tools.
"""

from factoriax.playground.ui.compositing import (
    blit_rgba,
    blit_scroll_view,
    clip_scroll_offset,
    composite_rgba_over_rgb,
)
from factoriax.playground.ui.fonts import get_pixel_font, render_text_rgba
from factoriax.playground.ui.primitives import draw_panel, hit_test_regions
from factoriax.playground.ui.scaling import ScaledCanvas
from factoriax.playground.ui.theme import (
    BORDER,
    BORDER_PX,
    FONT_BODY,
    FONT_HEADER,
    FONT_HINT,
    HEADER_H,
    HINT_COLOR,
    HINT_HEIGHT,
    PANEL_BG,
    SCROLL_STEP,
    SCROLLBAR_BG,
    SCROLLBAR_THUMB,
    SCROLLBAR_W,
    SEP_H,
    SLOT_COUNT_COLOR,
    TEXT_COLOR,
)
from factoriax.playground.ui.window import auto_ui_scale, calculate_window_size

__all__ = [
    "ScaledCanvas",
    "auto_ui_scale",
    "blit_rgba",
    "blit_scroll_view",
    "calculate_window_size",
    "clip_scroll_offset",
    "composite_rgba_over_rgb",
    "draw_panel",
    "get_pixel_font",
    "hit_test_regions",
    "render_text_rgba",
    "BORDER",
    "BORDER_PX",
    "FONT_BODY",
    "FONT_HEADER",
    "FONT_HINT",
    "HEADER_H",
    "HINT_COLOR",
    "HINT_HEIGHT",
    "PANEL_BG",
    "SCROLL_STEP",
    "SCROLLBAR_BG",
    "SCROLLBAR_THUMB",
    "SCROLLBAR_W",
    "SEP_H",
    "SLOT_COUNT_COLOR",
    "TEXT_COLOR",
]
