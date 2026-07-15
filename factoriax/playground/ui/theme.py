"""Shared color and size constants for all FactoriaX UI applications.

All size values are defined at 1x scale (32 px-per-block base resolution).
Call :func:`apply_scale` once at startup to multiply them by the chosen
UI scale factor. Color constants are scale-independent.
"""

# ---------------------------------------------------------------------------
# Scale factor (set once at startup via apply_scale)
# ---------------------------------------------------------------------------

UI_SCALE: int = 1

# Base render resolution: pixels per block at 1x scale. Default tile and
# texture size for the renderer and icon generation.
BLOCK_PIXEL_SIZE: int = 32

# ---------------------------------------------------------------------------
# Base values at 1x scale (private, never imported directly)
# ---------------------------------------------------------------------------

_BASE_BORDER_PX: int = 4
_BASE_FONT_HEADER: int = 24
_BASE_FONT_BODY: int = 18
_BASE_FONT_HINT: int = 12
_BASE_HEADER_H: int = 44
_BASE_SEP_H: int = 4
_BASE_HINT_HEIGHT: int = 24
_BASE_SCROLL_STEP: int = 24
_BASE_SCROLLBAR_W: int = 8
_BASE_PAGE_PAD: int = 32
_BASE_PANEL_GUTTER: int = 24
_BASE_PANEL_TITLE_GAP: int = 8
_BASE_ROW_H: int = 36
_BASE_ROW_PAD: int = 16
_BASE_DOC_PAD: int = 20
_BASE_DOC_LINE_GAP: int = 4
_BASE_BORDER_RADIUS: int = 12

# ---------------------------------------------------------------------------
# Color constants (scale-independent)
# ---------------------------------------------------------------------------

PAGE_BG: tuple[int, int, int] = (20, 20, 25)
PANEL_BG: tuple[int, int, int, int] = (22, 22, 22, 228)
MENU_PANEL_BG: tuple[int, int, int] = (28, 28, 32)
BUTTON_FILL: tuple[int, int, int] = (35, 35, 40)
BUTTON_HOVER: tuple[int, int, int] = (55, 55, 60)
BORDER: tuple[int, int, int, int] = (190, 165, 55, 255)
BORDER_INACTIVE: tuple[int, int, int] = (
    int(BORDER[0] * 0.55),
    int(BORDER[1] * 0.55),
    int(BORDER[2] * 0.55),
)
TEXT_COLOR: tuple[int, int, int] = (220, 215, 180)
HINT_COLOR: tuple[int, int, int] = (180, 175, 140)
SLOT_COUNT_COLOR: tuple[int, int, int] = (220, 215, 180)
SCROLLBAR_BG: tuple[int, int, int, int] = (40, 40, 40, 200)
SCROLLBAR_THUMB: tuple[int, int, int, int] = (140, 130, 80, 255)

# ---------------------------------------------------------------------------
# Scaled size constants (updated by apply_scale)
# ---------------------------------------------------------------------------

BORDER_PX: int = _BASE_BORDER_PX
FONT_HEADER: int = _BASE_FONT_HEADER
FONT_BODY: int = _BASE_FONT_BODY
FONT_HINT: int = _BASE_FONT_HINT
HEADER_H: int = _BASE_HEADER_H
SEP_H: int = _BASE_SEP_H
HINT_HEIGHT: int = _BASE_HINT_HEIGHT
SCROLL_STEP: int = _BASE_SCROLL_STEP
SCROLLBAR_W: int = _BASE_SCROLLBAR_W
PAGE_PAD: int = _BASE_PAGE_PAD
PANEL_GUTTER: int = _BASE_PANEL_GUTTER
PANEL_TITLE_GAP: int = _BASE_PANEL_TITLE_GAP
ROW_H: int = _BASE_ROW_H
ROW_PAD: int = _BASE_ROW_PAD
DOC_PAD: int = _BASE_DOC_PAD
DOC_LINE_GAP: int = _BASE_DOC_LINE_GAP
BORDER_RADIUS: int = _BASE_BORDER_RADIUS


def apply_scale(factor: int) -> None:
    """Set the UI scale factor and recompute all size constants.

    Must be called once at startup before any UI rendering occurs.
    Color constants are not affected.

    Parameters
    ----------
    factor :
        Integer scale multiplier (1, 2, or 3).
    """
    global UI_SCALE
    global BORDER_PX, FONT_HEADER, FONT_BODY, FONT_HINT
    global HEADER_H, SEP_H, HINT_HEIGHT, SCROLL_STEP, SCROLLBAR_W
    global PAGE_PAD, PANEL_GUTTER, PANEL_TITLE_GAP
    global ROW_H, ROW_PAD, DOC_PAD, DOC_LINE_GAP, BORDER_RADIUS

    UI_SCALE = factor
    BORDER_PX = _BASE_BORDER_PX * factor
    FONT_HEADER = _BASE_FONT_HEADER * factor
    FONT_BODY = _BASE_FONT_BODY * factor
    FONT_HINT = _BASE_FONT_HINT * factor
    HEADER_H = _BASE_HEADER_H * factor
    SEP_H = _BASE_SEP_H * factor
    HINT_HEIGHT = _BASE_HINT_HEIGHT * factor
    SCROLL_STEP = _BASE_SCROLL_STEP * factor
    SCROLLBAR_W = _BASE_SCROLLBAR_W * factor
    PAGE_PAD = _BASE_PAGE_PAD * factor
    PANEL_GUTTER = _BASE_PANEL_GUTTER * factor
    PANEL_TITLE_GAP = _BASE_PANEL_TITLE_GAP * factor
    ROW_H = _BASE_ROW_H * factor
    ROW_PAD = _BASE_ROW_PAD * factor
    DOC_PAD = _BASE_DOC_PAD * factor
    DOC_LINE_GAP = _BASE_DOC_LINE_GAP * factor
    BORDER_RADIUS = _BASE_BORDER_RADIUS * factor
