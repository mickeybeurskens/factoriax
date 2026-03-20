"""Player-only UI components using pygame for pixel-font text rendering.

This module is intentionally separate from renderer.py so that the RL
environment never pulls in a pygame dependency.  Only the play subpackage
(and any other human-facing entry points) should import from here.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
import pygame

from factoriax.achievements import ACHIEVEMENT_INFO, NUM_ACHIEVEMENTS
from factoriax.constants import (
    ITEM_COLORS,
    MACHINE_NUM_SLOTS,
    MACHINE_SLOT_ROLES,
    MACHINE_TYPE_NAMES,
    NUM_INVENTORY_SLOTS,
    NUM_RECIPES,
    RECIPE_NAMES,
    RECIPES,
    SLOT_ROLE_COLORS,
    SLOT_ROLE_LABELS,
    ItemType,
)
from factoriax.crafting import can_afford_recipe, count_item_in_inventory
from factoriax.renderer import render_item_icon
from factoriax.state import EnvState


@dataclass(frozen=True, slots=True)
class ClickRegion:
    """Represents a clickable rectangular area in the UI.

    Coordinates are in base render resolution (before window scaling).

    Attributes:
        x: Left edge of the region in pixels.
        y: Top edge of the region in pixels.
        w: Width of the region in pixels.
        h: Height of the region in pixels.
        action: Type of action to perform ("select_slot", "select_recipe",
            "focus_inventory", "focus_crafting", "pause_option").
        param: Action-specific parameter (slot/recipe index, option index).
    """

    x: int
    y: int
    w: int
    h: int
    action: str
    param: int

# ---------------------------------------------------------------------------
# Style constants — edit here to restyle every menu at once.
# All values are sized for the 32 px-per-block base resolution.
# ---------------------------------------------------------------------------

_PANEL_BG: tuple[int, int, int, int] = (22, 22, 22, 228)
_PAUSE_OPTION_NORMAL: tuple[int, int, int, int] = (45, 45, 45, 255)
_PAUSE_OPTION_SELECTED: tuple[int, int, int, int] = (75, 75, 75, 255)
_BORDER: tuple[int, int, int, int] = (190, 165, 55, 255)
_BORDER_PX: int = 4
_FOCUS_STRIP: tuple[int, int, int, int] = (55, 130, 55, 255)
_HEADER_H: int = 44  # height reserved for each section label row
_SEP_H: int = 4  # height of the gold separator beneath labels
_FONT_HEADER: int = 26  # section label font size
_FONT_BODY: int = 20  # item names, counts, recipe info font size
_FONT_HINT: int = 14  # control hint font size
_HINT_HEIGHT: int = 24  # height reserved for hint bar at bottom of menus
_HINT_COLOR: tuple[int, int, int] = (120, 115, 90)

# Scroll system constants — shared by every scrollable menu.
SCROLL_STEP: int = 24
_SCROLLBAR_W: int = 8
_SCROLLBAR_BG: tuple[int, int, int, int] = (40, 40, 40, 200)
_SCROLLBAR_THUMB: tuple[int, int, int, int] = (140, 130, 80, 255)

# Crafting ingredient affordability colours.
_AFFORD_COLOR: tuple[int, int, int] = (110, 220, 110)
_CANNOT_AFFORD_COLOR: tuple[int, int, int] = (200, 80, 80)

# Item display names keyed by ItemType value.
_ITEM_NAMES: dict[int, str] = {
    ItemType.COAL: "Coal",
    ItemType.IRON: "Iron",
    ItemType.COPPER: "Copper",
    ItemType.MINER: "Miner",
    ItemType.CHEST: "Chest",
    ItemType.CONVEYOR_BELT: "Belt",
    ItemType.ARM: "Arm",
}

# Comma-separated preference list for pygame.font.SysFont.  Terminus is a
# 1:1 pixel bitmap font common on Linux; the rest are fallbacks.
_PIXEL_FONT_PREFERENCE = "terminus,fixedsys excelsior,courier new,monospace,courier"


# ---------------------------------------------------------------------------
# Core drawing primitives
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=16)
def get_pixel_font(size: int) -> pygame.font.Font:
    """Return a pixel-style monospace font at the requested size.

    Results are cached so font objects are created at most once per unique
    size, regardless of how many frames are rendered.  Requires
    pygame.font to be initialised before the first call.

    Args:
        size: Desired font height in pixels.

    Returns:
        A pygame.font.Font instance.
    """
    font = pygame.font.SysFont(_PIXEL_FONT_PREFERENCE, size)
    if font is not None:
        return font
    return pygame.font.Font(pygame.font.get_default_font(), size)


def draw_panel(
    overlay: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    bg: tuple[int, int, int, int] = _PANEL_BG,
    border: tuple[int, int, int, int] = _BORDER,
    border_px: int = _BORDER_PX,
) -> None:
    """Draw a rectangular panel: solid background with a uniform border.

    All menus call this first so their look is controlled by the style
    constants above.  Modifies *overlay* in place.

    Args:
        overlay: Destination RGBA array of shape (H, W, 4).
        x: Left column of the panel.
        y: Top row of the panel.
        w: Panel width in pixels.
        h: Panel height in pixels.
        bg: Background RGBA colour.
        border: Border RGBA colour.
        border_px: Border thickness in pixels.
    """
    overlay[y : y + h, x : x + w] = bg
    for i in range(border_px):
        overlay[y + i, x : x + w] = border
        overlay[y + h - 1 - i, x : x + w] = border
        overlay[y : y + h, x + i] = border
        overlay[y : y + h, x + w - 1 - i] = border


# Cache for rendered text arrays, keyed by (font identity, text, color).
# Font objects come from get_pixel_font (lru_cached), so their id() is
# stable for the lifetime of the process.  The cache avoids repeated
# pygame font rasterisation for text that hasn't changed between frames.
_text_rgba_cache: dict[
    tuple[int, str, tuple[int, int, int]], np.ndarray
] = {}


def _render_text_rgba(
    text: str,
    font: pygame.font.Font,
    color: tuple[int, int, int],
) -> np.ndarray:
    """Render text to an RGBA array with a fully transparent background.

    Results are cached by ``(id(font), text, color)`` so that identical
    text drawn on consecutive frames is rasterised at most once.
    antialias=False keeps every pixel either the exact glyph colour or
    transparent, which is what gives the pixelated look.

    Args:
        text: String to render.
        font: pygame Font to use.
        color: RGB glyph colour.

    Returns:
        RGBA numpy array of shape (H, W, 4).
    """
    key = (id(font), text, color)
    cached = _text_rgba_cache.get(key)
    if cached is not None:
        return cached

    surface = font.render(text, False, color)
    w, h = surface.get_size()
    # surfarray returns (W, H, 3); transpose to (H, W, 3).
    rgb = pygame.surfarray.array3d(surface).transpose(1, 0, 2)
    result = np.zeros((h, w, 4), dtype=np.uint8)
    result[:, :, :3] = rgb
    result[:, :, 3] = np.where(np.any(rgb != 0, axis=2), 255, 0)

    _text_rgba_cache[key] = result
    return result


def _blit_rgba(
    overlay: np.ndarray,
    src: np.ndarray,
    y: int,
    x: int,
) -> None:
    """Alpha-composite *src* onto *overlay* at (y, x), clipping to bounds.

    Args:
        overlay: Destination RGBA array of shape (H, W, 4); modified in place.
        src: Source RGBA array of shape (h, w, 4).
        y: Top row in *overlay*.
        x: Left column in *overlay*.
    """
    oh, ow = overlay.shape[:2]
    sh, sw = src.shape[:2]

    src_y0 = max(0, -y)
    src_x0 = max(0, -x)
    dst_y0, dst_x0 = max(0, y), max(0, x)
    dst_y1 = min(oh, y + sh)
    dst_x1 = min(ow, x + sw)

    if dst_y1 <= dst_y0 or dst_x1 <= dst_x0:
        return

    crop_h = dst_y1 - dst_y0
    crop_w = dst_x1 - dst_x0
    src_crop = src[src_y0 : src_y0 + crop_h, src_x0 : src_x0 + crop_w]
    dst = overlay[dst_y0:dst_y1, dst_x0:dst_x1]

    alpha = src_crop[:, :, 3:4].astype(np.float32) / 255.0
    dst[:, :, :3] = (
        src_crop[:, :, :3].astype(np.float32) * alpha
        + dst[:, :, :3].astype(np.float32) * (1.0 - alpha)
    ).astype(np.uint8)
    dst[:, :, 3] = np.maximum(dst[:, :, 3], src_crop[:, :, 3])


def _draw_section_header(
    overlay: np.ndarray,
    x: int,
    y: int,
    w: int,
    text: str,
    font: pygame.font.Font,
    is_focused: bool,
) -> int:
    """Draw a labelled section header and return the y where content begins.

    Draws an optional focus strip, a centred section label, and a gold
    separator line.  The panel border at (y, x) is already drawn by the
    caller via draw_panel.

    Args:
        overlay: Destination RGBA array; modified in place.
        x: Left column of the section (panel border column).
        y: Top row of the panel (border row).
        w: Section width in pixels.
        text: Section label.
        font: pygame Font for the label.
        is_focused: Whether to draw the focus strip.

    Returns:
        Y coordinate of the first content pixel below the header.
    """
    content_y = y + _BORDER_PX

    if is_focused:
        overlay[
            content_y : content_y + _BORDER_PX,
            x + _BORDER_PX : x + w - _BORDER_PX,
        ] = _FOCUS_STRIP
        content_y += _BORDER_PX

    label = _render_text_rgba(text, font, (215, 195, 65))
    label_x = x + (w - label.shape[1]) // 2
    label_y = content_y + (_HEADER_H - label.shape[0]) // 2
    _blit_rgba(overlay, label, label_y, label_x)

    sep_y = content_y + _HEADER_H
    overlay[sep_y : sep_y + _SEP_H, x + _BORDER_PX + 4 : x + w - _BORDER_PX - 4] = (
        _BORDER
    )
    return sep_y + _SEP_H + 8


def _render_control_hints(
    overlay: np.ndarray,
    hints: str,
    x: int,
    y: int,
    w: int,
) -> None:
    """Render a control hint bar centered in the given region.

    Args:
        overlay: Destination RGBA array; modified in place.
        hints: Hint text to render (e.g. "[TAB] Next | [E] Craft").
        x: Left edge of the hint region.
        y: Top edge of the hint region.
        w: Width of the hint region.
    """
    font = get_pixel_font(_FONT_HINT)
    hint_arr = _render_text_rgba(hints, font, _HINT_COLOR)
    hint_x = x + (w - hint_arr.shape[1]) // 2
    hint_y = y + (_HINT_HEIGHT - hint_arr.shape[0]) // 2
    _blit_rgba(overlay, hint_arr, hint_y, hint_x)


# ---------------------------------------------------------------------------
# Scroll system
# ---------------------------------------------------------------------------


def clip_scroll_offset(offset: int, content_h: int, viewport_h: int) -> int:
    """Clamp a scroll offset to the valid range for the given content and viewport.

    Args:
        offset: Proposed scroll offset in pixels.
        content_h: Total height of the scrollable content in pixels.
        viewport_h: Height of the visible viewport in pixels.

    Returns:
        Clamped offset in ``[0, max(0, content_h - viewport_h)]``.
    """
    return max(0, min(offset, max(0, content_h - viewport_h)))


def blit_scroll_view(
    overlay: np.ndarray,
    content: np.ndarray,
    vp_x: int,
    vp_y: int,
    vp_w: int,
    vp_h: int,
    scroll_offset: int,
) -> None:
    """Composite a scrollable content canvas into a viewport on *overlay*.

    When the content is taller than the viewport a scrollbar is drawn along
    the right edge of the viewport.  The scrollbar appearance is fully
    controlled by this function so every menu looks identical — callers only
    choose the viewport geometry.

    Args:
        overlay: Destination RGBA array; modified in place.
        content: Full content RGBA canvas of shape ``(content_h, vp_w, 4)``.
        vp_x: Left edge of the viewport in overlay coordinates.
        vp_y: Top edge of the viewport in overlay coordinates.
        vp_w: Viewport width in pixels (includes scrollbar when shown).
        vp_h: Viewport height in pixels.
        scroll_offset: Number of content pixels scrolled off the top.
    """
    content_h = content.shape[0]
    needs_bar = content_h > vp_h
    render_w = vp_w - (_SCROLLBAR_W if needs_bar else 0)

    visible = content[scroll_offset : scroll_offset + vp_h, :render_w]
    _blit_rgba(overlay, visible, vp_y, vp_x)

    if needs_bar:
        bar_x = vp_x + vp_w - _SCROLLBAR_W
        overlay[vp_y : vp_y + vp_h, bar_x : bar_x + _SCROLLBAR_W] = _SCROLLBAR_BG
        thumb_h = max(12, vp_h * vp_h // content_h)
        max_scroll = content_h - vp_h
        thumb_y = vp_y + int((vp_h - thumb_h) * scroll_offset / max(1, max_scroll))
        overlay[
            thumb_y : thumb_y + thumb_h, bar_x : bar_x + _SCROLLBAR_W
        ] = _SCROLLBAR_THUMB


def scroll_adjust_regions(
    regions: list[ClickRegion],
    vp_x: int,
    vp_y: int,
    vp_h: int,
    scroll_offset: int,
) -> list[ClickRegion]:
    """Translate content-space click regions to screen-space, clipping to viewport.

    Content-space coordinates have ``x=0`` at the viewport's left edge and
    ``y=0`` at the top of the full (unscrolled) content canvas.

    Args:
        regions: Click regions in content-space coordinates.
        vp_x: Left edge of the viewport in screen coordinates.
        vp_y: Top edge of the viewport in screen coordinates.
        vp_h: Viewport height in pixels (used to discard off-screen regions).
        scroll_offset: Pixels of content scrolled off the top.

    Returns:
        Filtered list of :class:`ClickRegion` objects in screen coordinates.
    """
    result: list[ClickRegion] = []
    for r in regions:
        screen_y = r.y - scroll_offset + vp_y
        if screen_y + r.h <= vp_y or screen_y >= vp_y + vp_h:
            continue
        result.append(
            ClickRegion(
                x=r.x + vp_x,
                y=screen_y,
                w=r.w,
                h=r.h,
                action=r.action,
                param=r.param,
            )
        )
    return result


# ---------------------------------------------------------------------------
# Public menu renderers
# ---------------------------------------------------------------------------


def render_achievement_menu(
    state: EnvState,
    screen_width: int,
    screen_height: int,
    scroll_offset: int = 0,
) -> np.ndarray:
    """Render the achievement menu as a scrollable RGBA overlay.

    Achievements are laid out in a fixed-height scroll view so the list
    remains comfortable even as more achievements are added.  A scrollbar
    appears automatically when the content overflows the viewport.  A footer
    shows the overall completion count.

    Args:
        state: Current environment state.
        screen_width: Total render width in pixels.
        screen_height: Total render height in pixels.
        scroll_offset: Pixels of content scrolled off the top.

    Returns:
        RGBA numpy array of shape ``(screen_height, screen_width, 4)``.
    """
    overlay = np.zeros((screen_height, screen_width, 4), dtype=np.uint8)

    menu_w = int(screen_width * 0.52)
    menu_h = int(screen_height * 0.68)
    menu_x = (screen_width - menu_w) // 2
    menu_y = (screen_height - menu_h) // 2

    draw_panel(overlay, menu_x, menu_y, menu_w, menu_h)

    title_font = get_pixel_font(32)
    body_font = get_pixel_font(_FONT_BODY)

    title_arr = _render_text_rgba("ACHIEVEMENTS", title_font, (215, 195, 65))
    title_x = menu_x + (menu_w - title_arr.shape[1]) // 2
    _blit_rgba(overlay, title_arr, menu_y + _BORDER_PX + 16, title_x)

    sep_y = menu_y + _BORDER_PX + 16 + title_arr.shape[0] + 12
    overlay[sep_y : sep_y + _SEP_H, menu_x + 20 : menu_x + menu_w - 20] = _BORDER

    unlocked = np.array(state.achievements_unlocked)
    n_unlocked = int(np.sum(unlocked))

    # Fixed-height rows make the scroll math simple and the list readable.
    row_h = 36
    footer_reserve = _HINT_HEIGHT + _BORDER_PX + 40

    vp_x = menu_x + _BORDER_PX
    vp_y = sep_y + _SEP_H + 8
    vp_w = menu_w - 2 * _BORDER_PX
    vp_h = menu_y + menu_h - footer_reserve - vp_y

    content_h = NUM_ACHIEVEMENTS * row_h
    scroll_offset = clip_scroll_offset(scroll_offset, content_h, vp_h)

    icon_size = 20
    content = np.zeros((content_h, vp_w, 4), dtype=np.uint8)
    for i, info in enumerate(ACHIEVEMENT_INFO):
        is_unlocked = bool(unlocked[i])
        row_y = i * row_h

        if is_unlocked:
            content[row_y : row_y + row_h - 4, 8 : vp_w - 8] = (42, 68, 42, 210)

        icon_x = 24
        icon_y = row_y + (row_h - icon_size) // 2
        icon_color: tuple[int, int, int, int] = (
            (75, 215, 75, 255) if is_unlocked else (65, 65, 65, 255)
        )
        content[icon_y : icon_y + icon_size, icon_x : icon_x + icon_size] = icon_color

        text_color: tuple[int, int, int] = (
            (235, 228, 185) if is_unlocked else (105, 105, 88)
        )
        name_arr = _render_text_rgba(info.name, body_font, text_color)
        name_y = row_y + (row_h - name_arr.shape[0]) // 2
        _blit_rgba(content, name_arr, name_y, icon_x + icon_size + 12)

    blit_scroll_view(overlay, content, vp_x, vp_y, vp_w, vp_h, scroll_offset)

    footer_arr = _render_text_rgba(
        f"{n_unlocked} / {NUM_ACHIEVEMENTS} unlocked",
        body_font,
        (148, 140, 98),
    )
    fx = menu_x + (menu_w - footer_arr.shape[1]) // 2
    fy = menu_y + menu_h - footer_arr.shape[0] - 20 - _HINT_HEIGHT
    _blit_rgba(overlay, footer_arr, fy, fx)
    overlay[fy - 8 : fy - 6, menu_x + 20 : menu_x + menu_w - 20] = (80, 75, 40, 255)

    hint_y = menu_y + menu_h - _HINT_HEIGHT - _BORDER_PX
    _render_control_hints(
        overlay,
        "[UP/DOWN] Scroll  [ESC] Close",
        menu_x + _BORDER_PX,
        hint_y,
        menu_w - 2 * _BORDER_PX,
    )

    return overlay


def render_pause_menu(
    screen_width: int,
    screen_height: int,
    selected_option: int = 0,
) -> tuple[np.ndarray, list[ClickRegion]]:
    """Render the pause menu as an RGBA overlay.

    Displays a centered panel with Resume and Quit Game options. The selected
    option is highlighted with a brighter background.

    Args:
        screen_width: Total render width in pixels.
        screen_height: Total render height in pixels.
        selected_option: Currently selected option index (0=Resume, 1=Quit).

    Returns:
        Tuple of (RGBA overlay array, list of click regions).
    """
    overlay = np.zeros((screen_height, screen_width, 4), dtype=np.uint8)
    click_regions: list[ClickRegion] = []

    menu_w = int(screen_width * 0.35)
    menu_h = int(screen_height * 0.35)
    menu_x = (screen_width - menu_w) // 2
    menu_y = (screen_height - menu_h) // 2

    draw_panel(overlay, menu_x, menu_y, menu_w, menu_h)

    title_font = get_pixel_font(28)
    body_font = get_pixel_font(_FONT_BODY)

    title_arr = _render_text_rgba("PAUSED", title_font, (215, 195, 65))
    title_x = menu_x + (menu_w - title_arr.shape[1]) // 2
    _blit_rgba(overlay, title_arr, menu_y + _BORDER_PX + 16, title_x)

    sep_y = menu_y + _BORDER_PX + 16 + title_arr.shape[0] + 12
    overlay[sep_y : sep_y + _SEP_H, menu_x + 20 : menu_x + menu_w - 20] = _BORDER

    options = ["Resume", "Quit Game"]
    option_h = 40
    options_start_y = sep_y + _SEP_H + 24

    for i, option_text in enumerate(options):
        option_y = options_start_y + i * (option_h + 12)
        option_bg = (
            _PAUSE_OPTION_SELECTED if i == selected_option else _PAUSE_OPTION_NORMAL
        )

        option_x = menu_x + 24
        option_w = menu_w - 48
        overlay[
            option_y : option_y + option_h,
            option_x : option_x + option_w,
        ] = option_bg

        click_regions.append(
            ClickRegion(
                x=option_x,
                y=option_y,
                w=option_w,
                h=option_h,
                action="pause_option",
                param=i,
            )
        )

        if i == selected_option:
            white = (255, 255, 255, 255)
            overlay[option_y, option_x : option_x + option_w] = white
            overlay[option_y + option_h - 1, option_x : option_x + option_w] = white
            overlay[option_y : option_y + option_h, option_x] = white
            overlay[option_y : option_y + option_h, option_x + option_w - 1] = white

        text_color = (235, 228, 185) if i == selected_option else (160, 155, 130)
        text_arr = _render_text_rgba(option_text, body_font, text_color)
        text_x = option_x + (option_w - text_arr.shape[1]) // 2
        text_y = option_y + (option_h - text_arr.shape[0]) // 2
        _blit_rgba(overlay, text_arr, text_y, text_x)

    hint_y = menu_y + menu_h - _HINT_HEIGHT - _BORDER_PX
    _render_control_hints(
        overlay,
        "[UP/DOWN] Select | [ENTER/E] Confirm | [ESC] Back",
        menu_x + _BORDER_PX,
        hint_y,
        menu_w - 2 * _BORDER_PX,
    )

    return overlay, click_regions


def render_welcome_screen(
    screen_width: int,
    screen_height: int,
) -> np.ndarray:
    """Render the one-time welcome screen shown at game start.

    Displays a brief narrative hook, a compact control reference, and a
    prompt to dismiss.  Returned as a fully-opaque RGBA overlay so it
    completely covers the world behind it.

    Args:
        screen_width: Total render width in pixels.
        screen_height: Total render height in pixels.

    Returns:
        RGBA numpy array of shape ``(screen_height, screen_width, 4)``.
    """
    overlay = np.zeros((screen_height, screen_width, 4), dtype=np.uint8)
    overlay[:, :, :3] = 10
    overlay[:, :, 3] = 255

    menu_w = int(screen_width * 0.68)
    menu_x = (screen_width - menu_w) // 2

    title_font = get_pixel_font(28)
    body_font = get_pixel_font(_FONT_BODY)
    hint_font = get_pixel_font(_FONT_HINT)
    line_h = body_font.get_height()
    hint_h = hint_font.get_height()

    story = "You crashed on an unknown planet."
    goal = "Mine ore, build machines, launch a spaceship."

    controls = [
        ("WASD", "Move"),
        ("SPACE", "Mine ore"),
        ("E", "Place machine"),
        ("I", "Inventory & crafting"),
        ("F", "Inspect machine"),
    ]

    key_col_w = 64
    control_row_h = hint_h + 6
    controls_h = len(controls) * control_row_h

    inner_h = (
        48  # title
        + _SEP_H
        + 8  # separator
        + line_h
        + 6  # story line
        + line_h
        + 16  # goal line
        + _SEP_H
        + 8  # separator
        + controls_h  # control rows
        + 16  # gap before hint
        + _HINT_HEIGHT
    )
    menu_h = inner_h + 2 * (_BORDER_PX + 16)
    menu_y = (screen_height - menu_h) // 2

    draw_panel(overlay, menu_x, menu_y, menu_w, menu_h, bg=(22, 22, 22, 255))

    cy = menu_y + _BORDER_PX + 16

    # Title
    title_arr = _render_text_rgba("CRASH LANDED", title_font, (215, 195, 65))
    _blit_rgba(overlay, title_arr, cy, menu_x + (menu_w - title_arr.shape[1]) // 2)
    cy += title_arr.shape[0] + 8

    # Separator
    overlay[cy : cy + _SEP_H, menu_x + 20 : menu_x + menu_w - 20] = _BORDER
    cy += _SEP_H + 8

    # Story lines
    for text in (story, goal):
        arr = _render_text_rgba(text, body_font, (200, 195, 160))
        _blit_rgba(overlay, arr, cy, menu_x + (menu_w - arr.shape[1]) // 2)
        cy += line_h + 6
    cy += 10

    # Separator
    overlay[cy : cy + _SEP_H, menu_x + 20 : menu_x + menu_w - 20] = _BORDER
    cy += _SEP_H + 8

    # Controls
    controls_x = menu_x + (menu_w - key_col_w - 16 - 120) // 2
    for key, desc in controls:
        key_arr = _render_text_rgba(key, hint_font, (215, 195, 65))
        desc_arr = _render_text_rgba(desc, hint_font, (160, 155, 130))
        row_y = cy + (control_row_h - hint_h) // 2
        _blit_rgba(overlay, key_arr, row_y, controls_x + key_col_w - key_arr.shape[1])
        _blit_rgba(overlay, desc_arr, row_y, controls_x + key_col_w + 16)
        cy += control_row_h

    # Dismiss hint
    cy += 16
    hint_arr = _render_text_rgba("[SPACE / ENTER]  Start", hint_font, _HINT_COLOR)
    _blit_rgba(overlay, hint_arr, cy, menu_x + (menu_w - hint_arr.shape[1]) // 2)

    return overlay


def render_machine_menu(
    state: EnvState,
    screen_width: int,
    screen_height: int,
    tx: int,
    ty: int,
    machine_panel_active: bool = True,
) -> tuple[np.ndarray, list[ClickRegion]]:
    """Render the machine inventory inspection menu as an RGBA overlay.

    Displays all active slots for the machine at tile ``(tx, ty)`` in a grid
    of up to four columns.  Each slot shows a colour-coded role badge (IN /
    OUT / STORE), a filled colour swatch when the slot is occupied, a stack
    count, and the item name.  Empty slots show a dim placeholder.

    The focused slot in the *active* panel is highlighted with a bright white
    border; the focused slot in the *inactive* panel uses a dim gray border
    so the player can see both positions at a glance.  Press Tab to toggle
    which panel is active; press E to transfer between the two focused slots.

    Args:
        state: Current environment state.
        screen_width: Total render width in pixels.
        screen_height: Total render height in pixels.
        tx: X tile coordinate of the machine to inspect.
        ty: Y tile coordinate of the machine to inspect.
        machine_panel_active: Whether the machine slot grid has active focus
            (``True``) or the player inventory strip does (``False``).

    Returns:
        Tuple of (RGBA overlay array of shape ``(screen_height, screen_width,
        4)``, list of click regions).  Machine slot regions carry
        ``action="select_machine_slot"`` with ``param=slot_index``.  Player
        inventory regions carry ``action="select_slot"`` with
        ``param=slot_index``.
    """
    overlay = np.zeros((screen_height, screen_width, 4), dtype=np.uint8)
    click_regions: list[ClickRegion] = []

    machine_type = int(state.machine_types[ty, tx])
    num_slots = int(MACHINE_NUM_SLOTS[machine_type])
    slot_roles = MACHINE_SLOT_ROLES[machine_type]
    inv_items = np.array(state.machine_inventory_items[ty, tx])
    inv_counts = np.array(state.machine_inventory_counts[ty, tx])
    focused_slot = int(state.machine_selected_slot[ty, tx])

    selected_player = int(state.selected_player)
    focused_player_slot = int(state.selected_slots[selected_player])
    player_items = np.array(state.inventory_items[selected_player])
    player_counts = np.array(state.inventory_counts[selected_player])

    header_font = get_pixel_font(_FONT_HEADER)
    body_font = get_pixel_font(_FONT_BODY)
    hint_font = get_pixel_font(_FONT_HINT)
    line_h = body_font.get_height()

    # --- Layout constants ---
    padding = 20
    slot_cols = min(4, num_slots) if num_slots > 0 else 1
    slot_rows = (num_slots + slot_cols - 1) // slot_cols if num_slots > 0 else 0
    cell_gap = 8
    badge_h = 20
    icon_size = 40
    player_icon = 26

    # separator + gap + strip label + gap + icon row
    player_strip_h = _SEP_H + 8 + line_h + 6 + player_icon

    # _BORDER_PX + _HEADER_H + _SEP_H + 8 — matches _draw_section_header offset
    header_offset = _BORDER_PX + _HEADER_H + _SEP_H + 8

    # Fixed overhead: everything except the slot grid itself.
    overhead_h = (
        header_offset
        + padding
        + (padding if slot_rows > 0 else 0)
        + player_strip_h
        + padding // 2
        + _HINT_HEIGHT
        + _BORDER_PX
    )

    # Ideal cell height; shrink to fit if the screen is small.
    # badge + gap + icon + gap + count text + gap + name text + bottom margin
    cell_h = badge_h + 4 + icon_size + 4 + line_h + 2 + line_h + 10
    if slot_rows > 0:
        max_slot_h = max(0, int(screen_height * 0.92) - overhead_h)
        ideal_slot_h = slot_rows * cell_h + max(0, slot_rows - 1) * cell_gap
        if ideal_slot_h > max_slot_h:
            cell_h = max(
                badge_h + 20,
                (max_slot_h - max(0, slot_rows - 1) * cell_gap) // slot_rows,
            )
            icon_size = max(16, cell_h - badge_h - 4 - line_h * 2 - 14)

    slot_area_h = (
        slot_rows * cell_h + max(0, slot_rows - 1) * cell_gap if slot_rows > 0 else 0
    )

    menu_w = int(screen_width * 0.72)
    menu_h = overhead_h + slot_area_h
    menu_x = (screen_width - menu_w) // 2
    menu_y = (screen_height - menu_h) // 2

    draw_panel(overlay, menu_x, menu_y, menu_w, menu_h)

    machine_name = MACHINE_TYPE_NAMES.get(machine_type, "Machine")
    content_y = _draw_section_header(
        overlay, menu_x, menu_y, menu_w, machine_name, header_font, False
    )

    # --- Slot grid ---
    grid_w = menu_w - 2 * padding
    cell_w = (
        (grid_w - (slot_cols - 1) * cell_gap) // slot_cols if slot_cols > 0 else grid_w
    )
    grid_x = menu_x + padding

    for slot_idx in range(num_slots):
        row = slot_idx // slot_cols
        col = slot_idx % slot_cols
        cell_x = grid_x + col * (cell_w + cell_gap)
        cell_y = content_y + padding // 2 + row * (cell_h + cell_gap)

        role = int(slot_roles[slot_idx])
        role_label = SLOT_ROLE_LABELS.get(role, "")
        role_color = SLOT_ROLE_COLORS.get(role, (60, 60, 60))
        is_focused = slot_idx == focused_slot

        # Cell background + selection border
        cell_bg: tuple[int, int, int, int] = (
            (90, 90, 90, 255) if is_focused else (55, 55, 55, 255)
        )
        overlay[cell_y : cell_y + cell_h, cell_x : cell_x + cell_w] = cell_bg
        if is_focused:
            sel: tuple[int, int, int, int] = (
                (255, 255, 255, 255) if machine_panel_active else (100, 100, 100, 200)
            )
            overlay[cell_y, cell_x : cell_x + cell_w] = sel
            overlay[cell_y + cell_h - 1, cell_x : cell_x + cell_w] = sel
            overlay[cell_y : cell_y + cell_h, cell_x] = sel
            overlay[cell_y : cell_y + cell_h, cell_x + cell_w - 1] = sel

        click_regions.append(
            ClickRegion(
                x=cell_x,
                y=cell_y,
                w=cell_w,
                h=cell_h,
                action="select_machine_slot",
                param=slot_idx,
            )
        )

        # Role badge strip across the top of the cell
        overlay[cell_y : cell_y + badge_h, cell_x : cell_x + cell_w] = (
            *role_color,
            220,
        )
        if role_label:
            badge_arr = _render_text_rgba(role_label, hint_font, (230, 230, 230))
            badge_x = cell_x + (cell_w - badge_arr.shape[1]) // 2
            badge_y = cell_y + (badge_h - badge_arr.shape[0]) // 2
            _blit_rgba(overlay, badge_arr, badge_y, badge_x)

        # Item swatch, count, and name
        item_type = int(inv_items[slot_idx])
        count = int(inv_counts[slot_idx])
        icon_x = cell_x + (cell_w - icon_size) // 2
        icon_y = cell_y + badge_h + 4

        if item_type != 0 and count > 0:
            pad = 6
            icon_s = icon_size - 2 * pad
            if icon_s > 0:
                icon = render_item_icon(item_type, icon_s)
                overlay[
                    icon_y + pad : icon_y + pad + icon_s,
                    icon_x + pad : icon_x + pad + icon_s,
                ] = icon

            rgb = ITEM_COLORS.get(item_type, (128, 128, 128))
            count_arr = _render_text_rgba(f"x{count}", body_font, rgb)
            count_x = cell_x + (cell_w - count_arr.shape[1]) // 2
            count_y = icon_y + icon_size + 4
            _blit_rgba(overlay, count_arr, count_y, count_x)

            name = _ITEM_NAMES.get(item_type, "")
            if name:
                name_color: tuple[int, int, int] = (
                    (235, 228, 185) if is_focused else (160, 155, 130)
                )
                name_arr = _render_text_rgba(name, body_font, name_color)
                name_x = cell_x + (cell_w - name_arr.shape[1]) // 2
                _blit_rgba(overlay, name_arr, count_y + line_h + 2, name_x)
        else:
            pad = 6
            overlay[
                icon_y + pad : icon_y + icon_size - pad,
                icon_x + pad : icon_x + icon_size - pad,
            ] = (45, 45, 45, 255)
            empty_arr = _render_text_rgba("empty", body_font, (70, 70, 70))
            empty_x = cell_x + (cell_w - empty_arr.shape[1]) // 2
            _blit_rgba(overlay, empty_arr, icon_y + icon_size + 4, empty_x)

    # --- Player inventory strip ---
    slot_area_bottom = content_y + padding // 2 + slot_area_h
    sep_y = slot_area_bottom + (padding if slot_area_h > 0 else 0)
    overlay[
        sep_y : sep_y + _SEP_H,
        menu_x + _BORDER_PX + 4 : menu_x + menu_w - _BORDER_PX - 4,
    ] = _BORDER

    label_y = sep_y + _SEP_H + 8
    label_arr = _render_text_rgba("Player Inventory", body_font, (148, 140, 98))
    _blit_rgba(overlay, label_arr, label_y, menu_x + padding)

    strip_y = label_y + line_h + 6
    strip_x = menu_x + padding
    strip_cell_w = (menu_w - 2 * padding) // NUM_INVENTORY_SLOTS

    for slot_idx in range(NUM_INVENTORY_SLOTS):
        icon_x = strip_x + slot_idx * strip_cell_w
        icon_w = strip_cell_w - 4
        is_focused_player = slot_idx == focused_player_slot
        overlay[strip_y : strip_y + player_icon, icon_x : icon_x + icon_w] = (
            55,
            55,
            55,
            255,
        )
        click_regions.append(
            ClickRegion(
                x=icon_x,
                y=strip_y,
                w=icon_w,
                h=player_icon,
                action="select_slot",
                param=slot_idx,
            )
        )

        if is_focused_player:
            psel: tuple[int, int, int, int] = (
                (255, 255, 255, 255)
                if not machine_panel_active
                else (100, 100, 100, 200)
            )
            overlay[strip_y, icon_x : icon_x + icon_w] = psel
            overlay[strip_y + player_icon - 1, icon_x : icon_x + icon_w] = psel
            overlay[strip_y : strip_y + player_icon, icon_x] = psel
            overlay[strip_y : strip_y + player_icon, icon_x + icon_w - 1] = psel

        p_item = int(player_items[slot_idx])
        p_count = int(player_counts[slot_idx])
        if p_item != 0 and p_count > 0:
            pad = 3
            icon_h = player_icon - 2 * pad
            icon_w_inner = icon_w - 2 * pad
            icon_s = min(icon_h, icon_w_inner)
            icon = render_item_icon(p_item, icon_s)
            iy = strip_y + pad + (icon_h - icon_s) // 2
            ix = icon_x + pad + (icon_w_inner - icon_s) // 2
            overlay[iy : iy + icon_s, ix : ix + icon_s] = icon
            cnt_arr = _render_text_rgba(str(p_count), hint_font, (220, 220, 220))
            cnt_y = strip_y + player_icon - hint_font.get_height()
            _blit_rgba(overlay, cnt_arr, cnt_y, icon_x)

    # --- Hint bar ---
    hint_y = menu_y + menu_h - _HINT_HEIGHT - _BORDER_PX
    _render_control_hints(
        overlay,
        "[E] Transfer  [TAB] Switch panel  [◄/►] Select  [ESC] Close",
        menu_x + _BORDER_PX,
        hint_y,
        menu_w - 2 * _BORDER_PX,
    )

    return overlay, click_regions


def render_inventory_menu(
    state: EnvState,
    screen_width: int,
    screen_height: int,
    menu_focus: str = "inventory",
) -> tuple[np.ndarray, list[ClickRegion]]:
    """Render the inventory and crafting menu as an RGBA overlay.

    Left section: 2x5 inventory grid with item icon, count, and name per
    slot.  Right section: recipe list showing output, name, and per-
    ingredient have/need counts coloured by affordability.

    Args:
        state: Current environment state.
        screen_width: Total screen width in pixels.
        screen_height: Total screen height in pixels.
        menu_focus: Focused section — "inventory" or "crafting".

    Returns:
        Tuple of (RGBA overlay array, list of click regions).
    """
    overlay = np.zeros((screen_height, screen_width, 4), dtype=np.uint8)
    click_regions: list[ClickRegion] = []

    menu_w = int(screen_width * 0.75)
    menu_h = int(screen_height * 0.5)
    menu_x = (screen_width - menu_w) // 2
    menu_y = (screen_height - menu_h) // 2

    draw_panel(overlay, menu_x, menu_y, menu_w, menu_h)

    inv_w = int(menu_w * 0.6)
    craft_w = menu_w - inv_w
    div_x = menu_x + inv_w

    # Gold vertical divider between sections.
    for i in range(_BORDER_PX):
        overlay[menu_y : menu_y + menu_h, div_x + i] = _BORDER

    header_font = get_pixel_font(_FONT_HEADER)
    body_font = get_pixel_font(_FONT_BODY)

    inv_content_y = _draw_section_header(
        overlay,
        menu_x,
        menu_y,
        inv_w,
        "INVENTORY",
        header_font,
        menu_focus == "inventory",
    )
    craft_content_y = _draw_section_header(
        overlay,
        div_x,
        menu_y,
        craft_w,
        "CRAFTING",
        header_font,
        menu_focus == "crafting",
    )

    click_regions.append(
        ClickRegion(
            x=menu_x,
            y=menu_y,
            w=inv_w,
            h=inv_content_y - menu_y,
            action="focus_inventory",
            param=0,
        )
    )
    click_regions.append(
        ClickRegion(
            x=div_x,
            y=menu_y,
            w=craft_w,
            h=craft_content_y - menu_y,
            action="focus_crafting",
            param=0,
        )
    )

    selected_player = int(state.selected_player)
    selected_slot = int(state.selected_slots[selected_player])
    selected_recipe = int(state.selected_recipes[selected_player])
    craft_progress = int(state.craft_progress[selected_player])
    inventory_items = np.array(state.inventory_items[selected_player])
    inventory_counts = np.array(state.inventory_counts[selected_player])

    # ------------------------------------------------------------------
    # Inventory grid — 2 rows × 5 columns
    # ------------------------------------------------------------------
    cols = 5
    rows = 2
    padding = 20
    grid_x = menu_x + padding
    grid_w = inv_w - 2 * padding

    cell_w = grid_w // cols
    icon_size = min(cell_w - 12, 56)
    line_h = body_font.get_height()
    # Each cell: icon + 4px gap + count text + 2px gap + name text + 8px margin
    cell_h = icon_size + 4 + line_h + 2 + line_h + 8

    grid_available_h = menu_h - (inv_content_y - menu_y) - padding
    row_gap = max(8, (grid_available_h - rows * cell_h) // (rows + 1))
    grid_y = inv_content_y + row_gap

    for slot_idx in range(NUM_INVENTORY_SLOTS):
        row = slot_idx // cols
        col = slot_idx % cols

        cell_x = grid_x + col * cell_w
        cell_y = grid_y + row * (cell_h + row_gap)
        icon_x = cell_x + (cell_w - icon_size) // 2

        is_selected = (slot_idx == selected_slot) and (menu_focus == "inventory")
        slot_bg: tuple[int, int, int, int] = (
            (90, 90, 90, 255) if is_selected else (55, 55, 55, 255)
        )
        overlay[cell_y : cell_y + icon_size, icon_x : icon_x + icon_size] = slot_bg

        if is_selected:
            white = (255, 255, 255, 255)
            overlay[cell_y, icon_x : icon_x + icon_size] = white
            overlay[cell_y + icon_size - 1, icon_x : icon_x + icon_size] = white
            overlay[cell_y : cell_y + icon_size, icon_x] = white
            overlay[cell_y : cell_y + icon_size, icon_x + icon_size - 1] = white

        click_regions.append(
            ClickRegion(
                x=icon_x,
                y=cell_y,
                w=icon_size,
                h=cell_h,
                action="select_slot",
                param=slot_idx,
            )
        )

        item_type = int(inventory_items[slot_idx])
        count = int(inventory_counts[slot_idx])

        if item_type != 0 and count > 0:
            pad = 8
            icon_s = icon_size - 2 * pad
            if icon_s > 0:
                icon = render_item_icon(item_type, icon_s)
                overlay[
                    cell_y + pad : cell_y + pad + icon_s,
                    icon_x + pad : icon_x + pad + icon_s,
                ] = icon

            rgb = ITEM_COLORS.get(item_type, (128, 128, 128))
            count_arr = _render_text_rgba(f"x{count}", body_font, rgb)
            count_y = cell_y + icon_size + 4
            count_x = cell_x + (cell_w - count_arr.shape[1]) // 2
            _blit_rgba(overlay, count_arr, count_y, count_x)

            name = _ITEM_NAMES.get(item_type, "")
            if name:
                name_color: tuple[int, int, int] = (
                    (235, 228, 185) if is_selected else (160, 155, 130)
                )
                name_arr = _render_text_rgba(name, body_font, name_color)
                name_y = cell_y + icon_size + 4 + line_h + 2
                name_x = cell_x + (cell_w - name_arr.shape[1]) // 2
                _blit_rgba(overlay, name_arr, name_y, name_x)

    # ------------------------------------------------------------------
    # Crafting recipes list — rendered into a scroll view
    # ------------------------------------------------------------------
    hint_y = menu_y + menu_h - _HINT_HEIGHT - _BORDER_PX

    vp_x_craft = div_x + _BORDER_PX
    vp_y_craft = craft_content_y
    vp_w_craft = craft_w - _BORDER_PX
    vp_h_craft = hint_y - craft_content_y

    recipe_h = 80
    craft_pad = 8
    out_icon = 32
    inp_icon = 20

    content_h_craft = NUM_RECIPES * recipe_h
    craft_scroll = clip_scroll_offset(
        selected_recipe * recipe_h - (vp_h_craft - recipe_h) // 2,
        content_h_craft,
        vp_h_craft,
    )

    recipe_content = np.zeros((max(content_h_craft, 1), vp_w_craft, 4), dtype=np.uint8)
    recipe_regions: list[ClickRegion] = []

    for recipe_idx in range(NUM_RECIPES):
        recipe = RECIPES[recipe_idx]
        is_selected_recipe = (recipe_idx == selected_recipe) and (
            menu_focus == "crafting"
        )
        can_afford = bool(can_afford_recipe(state, selected_player, recipe_idx))
        ry = recipe_idx * recipe_h

        recipe_regions.append(
            ClickRegion(
                x=0,
                y=ry,
                w=vp_w_craft,
                h=recipe_h,
                action="select_recipe",
                param=recipe_idx,
            )
        )

        if is_selected_recipe:
            recipe_content[ry : ry + recipe_h - 4, 0:vp_w_craft] = (65, 65, 65, 255)
            white = (255, 255, 255, 255)
            recipe_content[ry, 0:vp_w_craft] = white
            recipe_content[ry + recipe_h - 5, 0:vp_w_craft] = white
            recipe_content[ry : ry + recipe_h - 4, 0] = white
            recipe_content[ry : ry + recipe_h - 4, vp_w_craft - 1] = white

        out_item = recipe["output"]
        out_s = min(out_icon, vp_w_craft - craft_pad)
        if out_s > 0:
            out_icon_arr = render_item_icon(out_item, out_s)
            recipe_content[
                ry + craft_pad : ry + craft_pad + out_s,
                craft_pad : craft_pad + out_s,
            ] = out_icon_arr

        name_color = (230, 225, 180) if can_afford else (150, 145, 120)
        name_arr = _render_text_rgba(RECIPE_NAMES[recipe_idx], body_font, name_color)
        name_y = ry + craft_pad + (out_icon - name_arr.shape[0]) // 2
        _blit_rgba(recipe_content, name_arr, name_y, craft_pad + out_icon + 12)

        inp_y = ry + craft_pad + out_icon + craft_pad
        inp_x = craft_pad

        for item_type, required in recipe["inputs"]:
            have = int(count_item_in_inventory(state, selected_player, item_type))
            inp_s = min(inp_icon, vp_w_craft - inp_x)
            if inp_s > 0:
                inp_icon_arr = render_item_icon(item_type, inp_s)
                recipe_content[
                    inp_y : inp_y + inp_s, inp_x : inp_x + inp_s
                ] = inp_icon_arr
            count_color: tuple[int, int, int] = (
                _AFFORD_COLOR if have >= required else _CANNOT_AFFORD_COLOR
            )
            ratio_arr = _render_text_rgba(f"{have}/{required}", body_font, count_color)
            _blit_rgba(recipe_content, ratio_arr, inp_y, inp_x + inp_icon + 6)
            inp_x += inp_icon + 6 + ratio_arr.shape[1] + 12

        if craft_progress > 0 and is_selected_recipe:
            bar_y = ry + recipe_h - 20
            bar_w = vp_w_craft - 16
            filled = int(bar_w * (1 - craft_progress / recipe["ticks"]))
            recipe_content[bar_y : bar_y + 8, craft_pad : craft_pad + bar_w] = (
                35, 35, 35, 255,
            )
            if filled > 0:
                recipe_content[
                    bar_y : bar_y + 8, craft_pad : craft_pad + filled
                ] = (100, 200, 100, 255)

    blit_scroll_view(
        overlay, recipe_content, vp_x_craft, vp_y_craft, vp_w_craft, vp_h_craft,
        craft_scroll,
    )
    click_regions.extend(
        scroll_adjust_regions(
            recipe_regions, vp_x_craft, vp_y_craft, vp_h_craft, craft_scroll
        )
    )

    if menu_focus == "crafting":
        hints = "[UP/DOWN] Select | [TAB] Inventory | [E] Craft | [ESC] Close"
    else:
        hints = "[LEFT/RIGHT] Select | [TAB] Crafting | [E] Place | [ESC] Close"
    _render_control_hints(
        overlay, hints, menu_x + _BORDER_PX, hint_y, menu_w - 2 * _BORDER_PX
    )

    return overlay, click_regions
