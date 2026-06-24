"""Shared visual constants and drawing primitives for menu/launch forms.

Holds the colour palette, base layout sizes, and small drawing helpers
used by multiple menu screens (the play launch screen and the controls
screen). Sizes are expressed in *base* pixels and multiplied by
``factoriax.playground.ui.theme.UI_SCALE`` at runtime by each caller.
"""

from __future__ import annotations

import pygame

from factoriax.playground.ui import theme as _theme
from factoriax.playground.ui.fonts import get_pixel_font

# -- Colors ---------------------------------------------------------------
BG: tuple[int, int, int] = (20, 20, 25)
GOLD: tuple[int, int, int] = (190, 165, 55)
SECTION_RULE: tuple[int, int, int] = (80, 70, 30)
LABEL_COLOR: tuple[int, int, int] = (200, 200, 200)
LABEL_DISABLED: tuple[int, int, int] = (90, 90, 90)
INPUT_BG: tuple[int, int, int] = (35, 35, 40)
INPUT_BORDER: tuple[int, int, int] = (80, 80, 80)
INPUT_TEXT: tuple[int, int, int] = (220, 220, 220)
INPUT_TEXT_DISABLED: tuple[int, int, int] = (90, 90, 90)
BTN_BG: tuple[int, int, int] = (30, 30, 35)
BTN_HOVER: tuple[int, int, int] = (45, 45, 50)
BTN_TEXT: tuple[int, int, int] = (220, 215, 180)
ERROR_BORDER: tuple[int, int, int] = (200, 50, 50)

# -- Base layout constants (multiplied by UI_SCALE at runtime) ------------
BASE_TOP_BAR_H: int = 80
BASE_SECTION_PAD_TOP: int = 24
BASE_SECTION_HEADER_H: int = 44
BASE_SECTION_RULE_H: int = 2
BASE_SECTION_GAP: int = 16
BASE_ROW_H: int = 48
BASE_ROW_GAP: int = 8
BASE_COL_GAP: int = 24
BASE_SIDE_PAD: int = 32
BASE_INPUT_W: int = 140
BASE_INPUT_H: int = 40
BASE_CHECKBOX_SIZE: int = 30
BASE_BTN_W: int = 160
BASE_BTN_H: int = 56
BASE_SCROLL_STEP: int = 24
FPS: int = 30
ERROR_FLASH_FRAMES: int = 15


def draw_button(
    surface: pygame.Surface,
    rect: pygame.Rect,
    text: str,
    font: pygame.font.Font,
    hovered: bool,
) -> None:
    """Draw a button with gold border, optional hover highlight.

    Parameters
    ----------
    surface :
        Destination surface.
    rect :
        Button bounding rectangle.
    text :
        Button label.
    font :
        Font for the label.
    hovered :
        Whether the mouse is over the button.
    surface: pygame.Surface :
        
    rect: pygame.Rect :
        
    text: str :
        
    font: pygame.font.Font :
        
    hovered: bool :
        

    Returns
    -------

    """
    bg = BTN_HOVER if hovered else BTN_BG
    pygame.draw.rect(surface, bg, rect)
    pygame.draw.rect(surface, GOLD, rect, 2)
    txt_surf = font.render(text, False, BTN_TEXT)
    tx = rect.x + (rect.width - txt_surf.get_width()) // 2
    ty = rect.y + (rect.height - txt_surf.get_height()) // 2
    surface.blit(txt_surf, (tx, ty))


def draw_checkbox(
    surface: pygame.Surface,
    x: int,
    y: int,
    size: int,
    checked: bool,
    hovered: bool,
) -> pygame.Rect:
    """Draw a checkbox square and return its bounding rect.

    Parameters
    ----------
    surface :
        Destination surface.
    x :
        Left edge in pixels.
    y :
        Top edge in pixels.
    size :
        Side length of the checkbox in pixels.
    checked :
        Whether the box is checked.
    hovered :
        Whether the mouse is over the checkbox area.
    surface: pygame.Surface :
        
    x: int :
        
    y: int :
        
    size: int :
        
    checked: bool :
        
    hovered: bool :
        

    Returns
    -------
    
        The bounding rectangle of the checkbox.

    """
    rect = pygame.Rect(x, y, size, size)
    bg = BTN_HOVER if hovered else INPUT_BG
    pygame.draw.rect(surface, bg, rect)
    pygame.draw.rect(surface, GOLD, rect, 1)
    if checked:
        font = get_pixel_font(24 * _theme.UI_SCALE)
        x_surf = font.render("X", False, GOLD)
        cx = rect.x + (rect.width - x_surf.get_width()) // 2
        cy = rect.y + (rect.height - x_surf.get_height()) // 2
        surface.blit(x_surf, (cx, cy))
    return rect


def draw_scrollbar(
    surface: pygame.Surface,
    scroll_offset: int,
    max_scroll: int,
    content_h: int,
    top_bar_h: int,
) -> None:
    """Draw a vertical scrollbar on the right edge when content overflows.

    Parameters
    ----------
    surface :
        Target surface.
    scroll_offset :
        Current scroll position.
    max_scroll :
        Maximum scroll offset.
    content_h :
        Total content height.
    top_bar_h :
        Height of the top bar (scrollbar starts below it).
    surface: pygame.Surface :
        
    scroll_offset: int :
        
    max_scroll: int :
        
    content_h: int :
        
    top_bar_h: int :
        

    Returns
    -------

    """
    if max_scroll <= 0:
        return
    sh = surface.get_height()
    bar_area_h = sh - top_bar_h
    thumb_h = max(20, int(bar_area_h * bar_area_h / content_h))
    thumb_y = top_bar_h + int(scroll_offset / max_scroll * (bar_area_h - thumb_h))
    bar_x = surface.get_width() - 8
    pygame.draw.rect(
        surface,
        (40, 40, 40),
        pygame.Rect(bar_x, top_bar_h, 8, bar_area_h),
    )
    pygame.draw.rect(
        surface,
        (140, 130, 80),
        pygame.Rect(bar_x, thumb_y, 8, thumb_h),
    )
