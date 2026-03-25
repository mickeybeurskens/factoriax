"""Shared color and size constants for all FactoriaX UI applications.

All values are sized for the 32 px-per-block base resolution. Edit here
to restyle every menu, toolbar, and inspector panel at once.
"""

# Panel backgrounds and borders.
PANEL_BG: tuple[int, int, int, int] = (22, 22, 22, 228)
BORDER: tuple[int, int, int, int] = (190, 165, 55, 255)
BORDER_PX: int = 4

# Text colors.
TEXT_COLOR: tuple[int, int, int] = (220, 215, 180)
HINT_COLOR: tuple[int, int, int] = (180, 175, 140)
SLOT_COUNT_COLOR: tuple[int, int, int] = (220, 215, 180)

# Font sizes.
FONT_HEADER: int = 26
FONT_BODY: int = 20
FONT_HINT: int = 14

# Section layout.
HEADER_H: int = 44
SEP_H: int = 4
HINT_HEIGHT: int = 24

# Scroll system.
SCROLL_STEP: int = 24
SCROLLBAR_W: int = 8
SCROLLBAR_BG: tuple[int, int, int, int] = (40, 40, 40, 200)
SCROLLBAR_THUMB: tuple[int, int, int, int] = (140, 130, 80, 255)
