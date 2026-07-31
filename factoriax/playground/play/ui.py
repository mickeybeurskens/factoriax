"""Player-only UI components using pygame for pixel-font text rendering.

This module is intentionally separate from the JAX renderer so that the
RL environment never pulls in a pygame dependency. Only the play
subpackage (and any other human-facing entry points) should import from
here.
"""

from __future__ import annotations

import functools

import numpy as np
import pygame

from factoriax.engine.constants import (
    BLOCK_MAX_RESOURCES,
    BLOCK_TO_ITEM,
    NUM_ITEM_TYPES,
    PLACEABLE_ITEM_LIST,
    RESOURCE_ITEM_LIST,
    BlockType,
    Direction,
    ItemType,
    Machine,
)
from factoriax.engine.crafting import can_afford_recipe, count_item_in_inventory
from factoriax.engine.state import EnvParams, EnvState
from factoriax.engine.tables import PLAYER_MAX_STACK
from factoriax.playground.play.achievements import FREE_PLAY_ACHIEVEMENTS
from factoriax.playground.ui import theme as _theme
from factoriax.playground.ui.compositing import blit_rgba as _blit_rgba
from factoriax.playground.ui.compositing import blit_scroll_view, clip_scroll_offset
from factoriax.playground.ui.fonts import get_pixel_font
from factoriax.playground.ui.fonts import render_text_rgba as _render_text_rgba
from factoriax.playground.ui.icons import ITEM_COLORS, PLAYER_COLORS, render_item_icon
from factoriax.playground.ui.labels import MACHINE_TYPE_NAMES
from factoriax.playground.ui.primitives import ClickRegion, draw_panel

SCROLL_STEP = _theme.SCROLL_STEP

# Play-specific style constants not shared with other apps.
_PAUSE_OPTION_NORMAL: tuple[int, int, int, int] = (45, 45, 45, 255)
_PAUSE_OPTION_SELECTED: tuple[int, int, int, int] = (75, 75, 75, 255)
_FOCUS_STRIP: tuple[int, int, int, int] = (55, 130, 55, 255)

# Crafting ingredient affordability colours.
_AFFORD_COLOR: tuple[int, int, int] = (110, 220, 110)
_CANNOT_AFFORD_COLOR: tuple[int, int, int] = (200, 80, 80)

# Names that differ from the title-cased enum name, because the play UI draws
# them in narrow slots and the full name does not fit.
_ITEM_NAME_OVERRIDES: dict[int, str] = {
    int(ItemType.IRON_ORE): "Iron",
    int(ItemType.COPPER_ORE): "Copper",
    int(ItemType.TIN_ORE): "Tin",
    int(ItemType.CONVEYOR_BELT): "Belt",
    int(ItemType.TIER1_SCIENCE_PACK): "T1 Sci",
    int(ItemType.TIER2_SCIENCE_PACK): "T2 Sci",
    int(ItemType.TIER3_SCIENCE_PACK): "T3 Sci",
}

# Item display names keyed by ItemType value. The enum supplies the entries, so
# a new item gets a name without an edit here.
_ITEM_NAMES: dict[int, str] = {
    int(it): _ITEM_NAME_OVERRIDES.get(int(it), it.name.replace("_", " ").title())
    for it in ItemType
    if it.name != "EMPTY"
}

# ---------------------------------------------------------------------------
# Entity-based inventory helper
# ---------------------------------------------------------------------------


def _entity_inventory(state: EnvState, ty: int, tx: int) -> np.ndarray:
    """Build a per-item-type count array from entity buffers at a tile.

    Parameters
    ----------
    state :
        Current environment state.
    ty :
        Tile Y coordinate.
    tx :
        Tile X coordinate.
    state : EnvState :

    ty : int :

    tx : int :

    state: EnvState :

    ty: int :

    tx: int :


    Returns
    -------


    """
    eidx = int(state.tile_entity[ty, tx])
    inv = np.zeros(NUM_ITEM_TYPES, dtype=np.int32)
    if eidx < 0:
        return inv
    buf_t = int(state.ent_buf_type[eidx])
    buf_c = int(state.ent_buf_count[eidx])
    if buf_t > 0:
        inv[buf_t] += buf_c
    for s in range(2):
        at = int(state.ent_asm_in_type[eidx, s])
        ac = int(state.ent_asm_in_count[eidx, s])
        if at > 0:
            inv[at] += ac
    aot = int(state.ent_asm_out_type[eidx])
    aoc = int(state.ent_asm_out_count[eidx])
    if aot > 0:
        inv[aot] += aoc
    return inv


# ---------------------------------------------------------------------------
# Core drawing primitives (re-exported from factoriax.playground.ui)
# ---------------------------------------------------------------------------


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

    Parameters
    ----------
    overlay :
        Destination RGBA array; modified in place.
    x :
        Left column of the section (panel border column).
    y :
        Top row of the panel (border row).
    w :
        Section width in pixels.
    text :
        Section label.
    font :
        pygame Font for the label.
    is_focused :
        Whether to draw the focus strip.
    overlay : np.ndarray :

    x : int :

    y : int :

    w : int :

    text : str :

    font : pygame.font.Font :

    is_focused : bool :

    overlay: np.ndarray :

    x: int :

    y: int :

    w: int :

    text: str :

    font: pygame.font.Font :

    is_focused: bool :


    Returns
    -------


    """
    content_y = y + _theme.BORDER_PX

    if is_focused:
        overlay[
            content_y : content_y + _theme.BORDER_PX,
            x + _theme.BORDER_PX : x + w - _theme.BORDER_PX,
        ] = _FOCUS_STRIP
        content_y += _theme.BORDER_PX

    label = _render_text_rgba(text, font, (215, 195, 65))
    label_x = x + (w - label.shape[1]) // 2
    label_y = content_y + (_theme.HEADER_H - label.shape[0]) // 2
    _blit_rgba(overlay, label, label_y, label_x)

    sep_y = content_y + _theme.HEADER_H
    overlay[
        sep_y : sep_y + _theme.SEP_H,
        x + _theme.BORDER_PX + 4 : x + w - _theme.BORDER_PX - 4,
    ] = _theme.BORDER
    return sep_y + _theme.SEP_H + 8


def _render_control_hints(
    overlay: np.ndarray,
    hints: str,
    x: int,
    y: int,
    w: int,
) -> None:
    """Render a control hint bar centered in the given region.

    Parameters
    ----------
    overlay :
        Destination RGBA array; modified in place.
    hints :
        Hint text to render (e.g. "[TAB] Next | [E] Craft").
    x :
        Left edge of the hint region.
    y :
        Top edge of the hint region.
    w :
        Width of the hint region.
    overlay : np.ndarray :

    hints : str :

    x : int :

    y : int :

    w : int :

    overlay: np.ndarray :

    hints: str :

    x: int :

    y: int :

    w: int :


    Returns
    -------


    """
    font = get_pixel_font(_theme.FONT_HINT)
    hint_arr = _render_text_rgba(hints, font, _theme.HINT_COLOR)
    hint_x = x + (w - hint_arr.shape[1]) // 2
    hint_y = y + (_theme.HINT_HEIGHT - hint_arr.shape[0]) // 2
    _blit_rgba(overlay, hint_arr, hint_y, hint_x)


# ---------------------------------------------------------------------------
# Scroll system
# ---------------------------------------------------------------------------


# clip_scroll_offset and blit_scroll_view are re-exported from factoriax.playground.ui
# at the top of this file.


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

    Parameters
    ----------
    regions :
        Click regions in content-space coordinates.
    vp_x :
        Left edge of the viewport in screen coordinates.
    vp_y :
        Top edge of the viewport in screen coordinates.
    vp_h :
        Viewport height in pixels (used to discard off-screen regions).
    scroll_offset :
        Pixels of content scrolled off the top.
    regions : list[ClickRegion] :

    vp_x : int :

    vp_y : int :

    vp_h : int :

    scroll_offset : int :

    regions: list[ClickRegion] :

    vp_x: int :

    vp_y: int :

    vp_h: int :

    scroll_offset: int :


    Returns
    -------


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
    achievements: np.ndarray,
    screen_width: int,
    screen_height: int,
    scroll_offset: int = 0,
    selected_index: int = 0,
) -> np.ndarray:
    """Render the achievement menu as a scrollable RGBA overlay.

    Achievements are laid out in a fixed-height scroll view so the list
    remains comfortable even as more achievements are added.  A scrollbar
    appears automatically when the content overflows the viewport.  The
    selected row is highlighted with a border, and a hint string for the
    selected achievement is shown above the footer.

    Parameters
    ----------
    achievements :
        Boolean array of unlocked achievement flags.
    screen_width :
        Total render width in pixels.
    screen_height :
        Total render height in pixels.
    scroll_offset :
        Pixels of content scrolled off the top.
    selected_index :
        Currently selected achievement row index.
    achievements : np.ndarray :

    screen_width : int :

    screen_height : int :

    scroll_offset : int :
        (Default value = 0)
    selected_index : int :
        (Default value = 0)
    achievements: np.ndarray :

    screen_width: int :

    screen_height: int :

    scroll_offset: int :
         (Default value = 0)
    selected_index: int :
         (Default value = 0)

    Returns
    -------


    """
    overlay = np.zeros((screen_height, screen_width, 4), dtype=np.uint8)

    menu_w = int(screen_width * 0.52)
    menu_h = int(screen_height * 0.68)
    menu_x = (screen_width - menu_w) // 2
    menu_y = (screen_height - menu_h) // 2

    draw_panel(overlay, menu_x, menu_y, menu_w, menu_h)

    title_font = get_pixel_font(32)
    body_font = get_pixel_font(_theme.FONT_BODY)

    title_arr = _render_text_rgba("ACHIEVEMENTS", title_font, (215, 195, 65))
    title_x = menu_x + (menu_w - title_arr.shape[1]) // 2
    _blit_rgba(overlay, title_arr, menu_y + _theme.BORDER_PX + 16, title_x)

    sep_y = menu_y + _theme.BORDER_PX + 16 + title_arr.shape[0] + 12
    overlay[sep_y : sep_y + _theme.SEP_H, menu_x + 20 : menu_x + menu_w - 20] = (
        _theme.BORDER
    )

    unlocked = np.array(achievements)
    n_unlocked = int(np.sum(unlocked))

    hint_font = get_pixel_font(_theme.FONT_HINT)

    # Fixed-height rows make the scroll math simple and the list readable.
    row_h = 36
    hint_text_h = hint_font.get_height() + 12
    footer_reserve = _theme.HINT_HEIGHT + _theme.BORDER_PX + 40 + hint_text_h

    vp_x = menu_x + _theme.BORDER_PX
    vp_y = sep_y + _theme.SEP_H + 8
    vp_w = menu_w - 2 * _theme.BORDER_PX
    vp_h = menu_y + menu_h - footer_reserve - vp_y

    content_h = len(FREE_PLAY_ACHIEVEMENTS) * row_h
    scroll_offset = clip_scroll_offset(scroll_offset, content_h, vp_h)

    icon_size = 20
    content = np.zeros((content_h, vp_w, 4), dtype=np.uint8)
    for i, info in enumerate(FREE_PLAY_ACHIEVEMENTS):
        is_unlocked = bool(unlocked[i])
        row_y = i * row_h

        if is_unlocked:
            content[row_y : row_y + row_h - 4, 8 : vp_w - 8] = (42, 68, 42, 210)

        if i == selected_index:
            sel_t = 2
            content[row_y : row_y + sel_t, 8 : vp_w - 8] = _theme.BORDER
            bot = row_y + row_h - 4
            content[bot - sel_t : bot, 8 : vp_w - 8] = _theme.BORDER
            content[row_y : row_y + row_h - 4, 8 : 8 + sel_t] = _theme.BORDER
            content[row_y : row_y + row_h - 4, vp_w - 8 - sel_t : vp_w - 8] = (
                _theme.BORDER
            )

        icon_x = 24
        icon_y = row_y + (row_h - icon_size) // 2
        icon_color: tuple[int, int, int, int] = (
            (75, 215, 75, 255) if is_unlocked else (65, 65, 65, 255)
        )
        content[icon_y : icon_y + icon_size, icon_x : icon_x + icon_size] = icon_color

        text_color: tuple[int, int, int] = (
            (235, 228, 185) if is_unlocked else (150, 148, 125)
        )
        name_arr = _render_text_rgba(info.name, body_font, text_color)
        name_y = row_y + (row_h - name_arr.shape[0]) // 2
        _blit_rgba(content, name_arr, name_y, icon_x + icon_size + 12)

    blit_scroll_view(overlay, content, vp_x, vp_y, vp_w, vp_h, scroll_offset)

    # Achievement hint for the selected row.
    sel_hint = FREE_PLAY_ACHIEVEMENTS[selected_index].hint
    hint_arr = _render_text_rgba(sel_hint, hint_font, _theme.HINT_COLOR)
    hx = menu_x + (menu_w - hint_arr.shape[1]) // 2
    hy = menu_y + menu_h - _theme.HINT_HEIGHT - _theme.BORDER_PX - 40 - hint_text_h + 4
    _blit_rgba(overlay, hint_arr, hy, hx)

    footer_arr = _render_text_rgba(
        f"{n_unlocked} / {len(FREE_PLAY_ACHIEVEMENTS)} unlocked",
        body_font,
        (180, 172, 130),
    )
    fx = menu_x + (menu_w - footer_arr.shape[1]) // 2
    fy = hy + hint_arr.shape[0] + 8
    _blit_rgba(overlay, footer_arr, fy, fx)
    overlay[fy - 4 : fy - 2, menu_x + 20 : menu_x + menu_w - 20] = (80, 75, 40, 255)

    hint_y = menu_y + menu_h - _theme.HINT_HEIGHT - _theme.BORDER_PX
    _render_control_hints(
        overlay,
        "[W/S] Select  [ESC] Close",
        menu_x + _theme.BORDER_PX,
        hint_y,
        menu_w - 2 * _theme.BORDER_PX,
    )

    return overlay


def render_pause_menu(
    screen_width: int,
    screen_height: int,
    selected_option: int = 0,
) -> tuple[np.ndarray, list[ClickRegion]]:
    """Render the pause menu as an RGBA overlay.

    Displays a centered panel with Resume, Reset, and Quit Game options.
    The selected option is highlighted with a brighter background.

    Parameters
    ----------
    screen_width :
        Total render width in pixels.
    screen_height :
        Total render height in pixels.
    selected_option :
        Currently selected option index
        (0=Resume, 1=Reset, 2=Quit).
    screen_width : int :

    screen_height : int :

    selected_option : int :
        (Default value = 0)
    screen_width: int :

    screen_height: int :

    selected_option: int :
         (Default value = 0)

    Returns
    -------


    """
    overlay = np.zeros((screen_height, screen_width, 4), dtype=np.uint8)
    click_regions: list[ClickRegion] = []

    menu_w = int(screen_width * 0.35)
    menu_h = int(screen_height * 0.42)
    menu_x = (screen_width - menu_w) // 2
    menu_y = (screen_height - menu_h) // 2

    draw_panel(overlay, menu_x, menu_y, menu_w, menu_h)

    title_font = get_pixel_font(28)
    body_font = get_pixel_font(_theme.FONT_BODY)

    title_arr = _render_text_rgba("PAUSED", title_font, (215, 195, 65))
    title_x = menu_x + (menu_w - title_arr.shape[1]) // 2
    _blit_rgba(overlay, title_arr, menu_y + _theme.BORDER_PX + 16, title_x)

    sep_y = menu_y + _theme.BORDER_PX + 16 + title_arr.shape[0] + 12
    overlay[sep_y : sep_y + _theme.SEP_H, menu_x + 20 : menu_x + menu_w - 20] = (
        _theme.BORDER
    )

    options = ["Resume", "Controls", "Reset", "Quit Game"]
    option_h = 40
    options_start_y = sep_y + _theme.SEP_H + 24

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

    hint_y = menu_y + menu_h - _theme.HINT_HEIGHT - _theme.BORDER_PX
    _render_control_hints(
        overlay,
        "[W/S] Select | [ENTER/E] Confirm | [ESC] Back",
        menu_x + _theme.BORDER_PX,
        hint_y,
        menu_w - 2 * _theme.BORDER_PX,
    )

    return overlay, click_regions


def render_welcome_screen(
    screen_width: int,
    screen_height: int,
    record_enabled: bool = False,
) -> tuple[np.ndarray, list[ClickRegion]]:
    """Render the one-time welcome screen shown at game start.

    Displays a brief narrative hook, a compact control reference, and a
    prompt to dismiss.

    Parameters
    ----------
    screen_width :
        Total render width in pixels.
    screen_height :
        Total render height in pixels.
    record_enabled :
        Unused, kept for API compat.
    screen_width : int :

    screen_height : int :

    record_enabled : bool :
        (Default value = False)
    screen_width: int :

    screen_height: int :

    record_enabled: bool :
         (Default value = False)

    Returns
    -------


    """
    overlay = np.zeros((screen_height, screen_width, 4), dtype=np.uint8)
    overlay[:, :, :3] = 10
    overlay[:, :, 3] = 255

    menu_w = int(screen_width * 0.68)
    menu_x = (screen_width - menu_w) // 2

    title_font = get_pixel_font(28)
    body_font = get_pixel_font(_theme.FONT_BODY)
    hint_font = get_pixel_font(_theme.FONT_HINT)
    line_h = body_font.get_height()
    hint_h = hint_font.get_height()

    story = "You crashed on an unknown planet."
    goal = "Mine ore, build machines, launch a spaceship."

    controls = [
        ("WASD", "Move"),
        ("SPACE", "Mine the tile you face"),
        ("E", "Place / pick up / craft"),
        ("I", "Inventory & crafting"),
        ("F", "Inspect machine"),
        ("?", "Full controls list"),
    ]

    key_col_w = 64
    control_row_h = hint_h + 6
    controls_h = len(controls) * control_row_h

    inner_h = (
        48  # title
        + _theme.SEP_H
        + 8  # separator
        + line_h
        + 6  # story line
        + line_h
        + 16  # goal line
        + _theme.SEP_H
        + 8  # separator
        + controls_h  # control rows
        + 16  # gap before hint
        + _theme.HINT_HEIGHT
    )
    menu_h = min(inner_h + 2 * (_theme.BORDER_PX + 16), screen_height - 4)
    menu_y = (screen_height - menu_h) // 2

    draw_panel(overlay, menu_x, menu_y, menu_w, menu_h, bg=(22, 22, 22, 255))

    cy = menu_y + _theme.BORDER_PX + 16

    # Title
    title_arr = _render_text_rgba("CRASH LANDED", title_font, (215, 195, 65))
    _blit_rgba(overlay, title_arr, cy, menu_x + (menu_w - title_arr.shape[1]) // 2)
    cy += title_arr.shape[0] + 8

    # Separator
    overlay[cy : cy + _theme.SEP_H, menu_x + 20 : menu_x + menu_w - 20] = _theme.BORDER
    cy += _theme.SEP_H + 8

    # Story lines
    for text in (story, goal):
        arr = _render_text_rgba(text, body_font, (200, 195, 160))
        _blit_rgba(overlay, arr, cy, menu_x + (menu_w - arr.shape[1]) // 2)
        cy += line_h + 6
    cy += 10

    # Separator
    overlay[cy : cy + _theme.SEP_H, menu_x + 20 : menu_x + menu_w - 20] = _theme.BORDER
    cy += _theme.SEP_H + 8

    # Controls
    controls_x = menu_x + (menu_w - key_col_w - 16 - 120) // 2
    for key, desc in controls:
        key_arr = _render_text_rgba(key, hint_font, (215, 195, 65))
        desc_arr = _render_text_rgba(desc, hint_font, (190, 185, 155))
        row_y = cy + (control_row_h - hint_h) // 2
        _blit_rgba(overlay, key_arr, row_y, controls_x + key_col_w - key_arr.shape[1])
        _blit_rgba(overlay, desc_arr, row_y, controls_x + key_col_w + 16)
        cy += control_row_h

    regions: list[ClickRegion] = []

    # Dismiss hint.
    cy += 16
    hint_arr = _render_text_rgba("[SPACE / ENTER]  Start", hint_font, _theme.HINT_COLOR)
    _blit_rgba(overlay, hint_arr, cy, menu_x + (menu_w - hint_arr.shape[1]) // 2)

    return overlay, regions


def render_victory_screen(
    screen_width: int,
    screen_height: int,
) -> np.ndarray:
    """Render a celebration overlay when the player builds a rocket.

    Parameters
    ----------
    screen_width :
        Total render width in pixels.
    screen_height :
        Total render height in pixels.
    screen_width : int :

    screen_height : int :

    screen_width: int :

    screen_height: int :


    Returns
    -------


    """
    overlay = np.zeros((screen_height, screen_width, 4), dtype=np.uint8)
    overlay[:, :] = (0, 0, 0, 200)

    menu_w = int(screen_width * 0.5)
    menu_h = 180
    menu_x = (screen_width - menu_w) // 2
    menu_y = (screen_height - menu_h) // 2

    draw_panel(overlay, menu_x, menu_y, menu_w, menu_h)

    title_font = get_pixel_font(28)
    body_font = get_pixel_font(_theme.FONT_BODY)
    hint_font = get_pixel_font(_theme.FONT_HINT)

    title = _render_text_rgba("ROCKET LAUNCHED", title_font, (215, 195, 65))
    _blit_rgba(
        overlay,
        title,
        menu_y + _theme.BORDER_PX + 16,
        menu_x + (menu_w - title.shape[1]) // 2,
    )

    sep_y = menu_y + _theme.BORDER_PX + 16 + title.shape[0] + 12
    overlay[
        sep_y : sep_y + _theme.SEP_H,
        menu_x + 20 : menu_x + menu_w - 20,
    ] = _theme.BORDER

    msg = _render_text_rgba(
        "You escaped the planet.",
        body_font,
        (200, 195, 160),
    )
    _blit_rgba(
        overlay,
        msg,
        sep_y + _theme.SEP_H + 16,
        menu_x + (menu_w - msg.shape[1]) // 2,
    )

    hint = _render_text_rgba(
        "[SPACE / ENTER]  Continue",
        hint_font,
        _theme.HINT_COLOR,
    )
    _blit_rgba(
        overlay,
        hint,
        menu_y + menu_h - _theme.BORDER_PX - hint.shape[0] - 8,
        menu_x + (menu_w - hint.shape[1]) // 2,
    )

    return overlay


def render_machine_menu(
    state: EnvState,
    params: EnvParams,
    screen_width: int,
    screen_height: int,
    tx: int,
    ty: int,
    machine_panel_active: bool = True,
    selected_item: int = 0,
    focused_machine_item: int = 0,
) -> tuple[np.ndarray, list[ClickRegion]]:
    """Render the machine inventory inspection menu as an RGBA overlay.

    Displays non-empty item types held by the machine at tile ``(tx, ty)``
    in a grid of up to four columns.  Each cell shows an item icon, stack
    count, and name.  Empty machines show a placeholder message.

    The focused item in the *active* panel is highlighted with a bright white
    border; the focused item in the *inactive* panel uses a dim gray border
    so the player can see both positions at a glance.

    Parameters
    ----------
    state :
        Current environment state.
    params :
        Environment parameters, read for this scenario's recipe table.
    screen_width :
        Total render width in pixels.
    screen_height :
        Total render height in pixels.
    tx :
        X tile coordinate of the machine to inspect.
    ty :
        Y tile coordinate of the machine to inspect.
    machine_panel_active :
        Whether the machine slot grid has active focus
        (``True``) or the player inventory strip does (``False``).
    selected_item :
        Currently selected player inventory item type.
    focused_machine_item :
        Currently focused machine item type.
    state : EnvState :

    screen_width : int :

    screen_height : int :

    tx : int :

    ty : int :

    machine_panel_active : bool :
        (Default value = True)
    selected_item : int :
        (Default value = 0)
    focused_machine_item : int :
        (Default value = 0)
    state: EnvState :

    screen_width: int :

    screen_height: int :

    tx: int :

    ty: int :

    machine_panel_active: bool :
         (Default value = True)
    selected_item: int :
         (Default value = 0)
    focused_machine_item: int :
         (Default value = 0)

    Returns
    -------


    """
    overlay = np.zeros((screen_height, screen_width, 4), dtype=np.uint8)
    click_regions: list[ClickRegion] = []

    machine_type = int(state.machine_types[ty, tx])
    machine_inv = _entity_inventory(state, ty, tx)
    active_types = [i for i in range(1, NUM_ITEM_TYPES) if machine_inv[i] > 0]
    num_slots = len(active_types)

    selected_player = int(state.selected_player)
    player_inv = np.array(state.player_inventory[selected_player])

    header_font = get_pixel_font(_theme.FONT_HEADER)
    body_font = get_pixel_font(_theme.FONT_BODY)
    hint_font = get_pixel_font(_theme.FONT_HINT)
    line_h = body_font.get_height()

    # --- Layout constants ---
    padding = 20
    slot_cols = min(4, num_slots) if num_slots > 0 else 1
    slot_rows = (num_slots + slot_cols - 1) // slot_cols if num_slots > 0 else 0
    cell_gap = 8
    badge_h = 20
    icon_size = 40
    player_icon = 26

    # separator + gap + label + gap + machines row + gap + resources row
    player_strip_h = _theme.SEP_H + 8 + line_h + 6 + player_icon + 4 + player_icon

    # Matches _draw_section_header offset.
    header_offset = _theme.BORDER_PX + _theme.HEADER_H + _theme.SEP_H + 8

    # Fixed overhead: everything except the slot grid itself.
    overhead_h = (
        header_offset
        + padding
        + (padding if slot_rows > 0 else 0)
        + player_strip_h
        + padding // 2
        + _theme.HINT_HEIGHT
        + _theme.BORDER_PX
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
    menu_h = min(overhead_h + slot_area_h, screen_height - 4)
    menu_x = (screen_width - menu_w) // 2
    menu_y = max(2, (screen_height - menu_h) // 2)

    draw_panel(overlay, menu_x, menu_y, menu_w, menu_h)

    machine_name = MACHINE_TYPE_NAMES.get(machine_type, "Machine")
    content_y = _draw_section_header(
        overlay, menu_x, menu_y, menu_w, machine_name, header_font, False
    )

    # --- Assembler recipe subtitle ---
    if machine_type == int(Machine.ASSEMBLER):
        eidx = int(state.tile_entity[ty, tx])
        if eidx >= 0:
            out_type = int(state.ent_asm_out_type[eidx])
            asm_power = int(state.ent_power[eidx])
            if out_type > 0:
                table = params.recipe_table
                recipe_idx = int(table.output_to_recipe[out_type])
                recipe_name = table.names[recipe_idx] if recipe_idx >= 0 else "Unknown"
            else:
                recipe_name = "Auto"
            status = "Idle" if asm_power == 0 else f"{asm_power} ticks left"
        else:
            recipe_name = "Auto"
            status = "Idle"
        subtitle = f"Recipe: {recipe_name}  |  {status}"
        sub_arr = _render_text_rgba(subtitle, body_font, (180, 170, 130))
        sub_x = menu_x + (menu_w - sub_arr.shape[1]) // 2
        _blit_rgba(overlay, sub_arr, content_y, sub_x)
        content_y += sub_arr.shape[0] + 6

    # --- Slot grid ---
    grid_w = menu_w - 2 * padding
    cell_w = (
        (grid_w - (slot_cols - 1) * cell_gap) // slot_cols if slot_cols > 0 else grid_w
    )
    grid_x = menu_x + padding

    for slot_idx, item_type in enumerate(active_types):
        row = slot_idx // slot_cols
        col = slot_idx % slot_cols
        cell_x = grid_x + col * (cell_w + cell_gap)
        cell_y = content_y + padding // 2 + row * (cell_h + cell_gap)

        is_focused = item_type == focused_machine_item

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
                param=item_type,
            )
        )

        count = int(machine_inv[item_type])
        icon_x = cell_x + (cell_w - icon_size) // 2
        icon_y = cell_y + badge_h + 4

        pad = 6
        icon_s = icon_size - 2 * pad
        if icon_s > 0:
            icon = render_item_icon(item_type, icon_s)
            overlay[
                icon_y + pad : icon_y + pad + icon_s,
                icon_x + pad : icon_x + pad + icon_s,
            ] = icon

        count_arr = _render_text_rgba(
            f"x{count}",
            body_font,
            _theme.SLOT_COUNT_COLOR,
        )
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

    # --- Player inventory strip (two rows: machines + resources) ---
    slot_area_bottom = content_y + padding // 2 + slot_area_h
    sep_y = slot_area_bottom + (padding if slot_area_h > 0 else 0)
    overlay[
        sep_y : sep_y + _theme.SEP_H,
        menu_x + _theme.BORDER_PX + 4 : menu_x + menu_w - _theme.BORDER_PX - 4,
    ] = _theme.BORDER

    label_y = sep_y + _theme.SEP_H + 8
    label_arr = _render_text_rgba("Player", body_font, (148, 140, 98))
    _blit_rgba(overlay, label_arr, label_y, menu_x + padding)

    strip_x = menu_x + padding
    strip_w = menu_w - 2 * padding

    def _draw_strip_row(
        items: tuple[int, ...] | list[int],
        row_y: int,
    ) -> None:
        """

        Parameters
        ----------
        items : tuple[int :

        ...] | list[int] :

        row_y : int :

        items: tuple[int :

        row_y: int :


        Returns
        -------


        """
        n = len(items)
        if n == 0:
            return
        scw = strip_w // n
        for idx, item_type in enumerate(items):
            ix = strip_x + idx * scw
            iw = scw - 4
            is_fp = item_type == selected_item
            oh, ow = overlay.shape[:2]

            # Background.
            y0 = max(0, row_y)
            y1 = min(oh, row_y + player_icon)
            x0 = max(0, ix)
            x1 = min(ow, ix + iw)
            if y0 < y1 and x0 < x1:
                overlay[y0:y1, x0:x1] = (55, 55, 55, 255)

            click_regions.append(
                ClickRegion(
                    x=ix,
                    y=row_y,
                    w=iw,
                    h=player_icon,
                    action="select_slot",
                    param=item_type,
                ),
            )

            # Selection border.
            if is_fp and y0 < y1 and x0 < x1:
                psel: tuple[int, int, int, int] = (
                    (255, 255, 255, 255)
                    if not machine_panel_active
                    else (100, 100, 100, 200)
                )
                if 0 <= row_y < oh:
                    overlay[row_y, x0:x1] = psel
                if 0 <= row_y + player_icon - 1 < oh:
                    overlay[row_y + player_icon - 1, x0:x1] = psel
                if 0 <= ix < ow:
                    overlay[y0:y1, ix] = psel
                if 0 <= ix + iw - 1 < ow:
                    overlay[y0:y1, ix + iw - 1] = psel

            # Icon or ghost + count.
            p_count = int(player_inv[item_type])
            spad = 3
            icon_h_inner = player_icon - 2 * spad
            icon_w_inner = iw - 2 * spad
            icon_s = min(icon_h_inner, icon_w_inner)
            if icon_s > 0:
                iiy = row_y + spad + (icon_h_inner - icon_s) // 2
                iix = ix + spad + (icon_w_inner - icon_s) // 2
                if p_count > 0:
                    ic = render_item_icon(item_type, icon_s)
                    _blit_rgba(overlay, ic, iiy, iix)
                else:
                    _render_ghost_icon(
                        overlay,
                        iix,
                        iiy,
                        icon_s,
                        item_type,
                    )
            if p_count > 0:
                cnt_arr = _render_text_rgba(
                    str(p_count),
                    hint_font,
                    (220, 220, 220),
                )
                cnt_y = row_y + player_icon - hint_font.get_height()
                _blit_rgba(overlay, cnt_arr, cnt_y, ix)

    machines_y = label_y + line_h + 6
    _draw_strip_row(PLACEABLE_ITEM_LIST, machines_y)

    resources_y = machines_y + player_icon + 4
    _draw_strip_row(RESOURCE_ITEM_LIST, resources_y)

    # --- Hint bar ---
    hint_y = menu_y + menu_h - _theme.HINT_HEIGHT - _theme.BORDER_PX
    hints = "[E] Transfer  [W/S] Switch panel  [A/D] Select  [ESC] Close"
    if machine_type == int(Machine.ASSEMBLER):
        hints = "[Q] Recipe  " + hints
    _render_control_hints(
        overlay,
        hints,
        menu_x + _theme.BORDER_PX,
        hint_y,
        menu_w - 2 * _theme.BORDER_PX,
    )

    return overlay, click_regions


# ---------------------------------------------------------------------------
# Pocket slot rendering
# ---------------------------------------------------------------------------

_POCKET_BG: tuple[int, int, int, int] = (38, 38, 38, 255)
_POCKET_SELECTED_BG: tuple[int, int, int, int] = (75, 75, 75, 255)
_POCKET_LIGHT_EDGE: tuple[int, int, int, int] = (55, 55, 55, 255)
_POCKET_DARK_EDGE: tuple[int, int, int, int] = (28, 28, 28, 255)
_POCKET_SELECTED_LIGHT: tuple[int, int, int, int] = (110, 110, 110, 255)
_POCKET_SELECTED_DARK: tuple[int, int, int, int] = (55, 55, 55, 255)
_CHIP_DIM: tuple[int, int, int, int] = (32, 32, 32, 255)
_CHIP_COUNT: int = 8
_CHIP_W: int = 3
_CHIP_H: int = 3
_CHIP_GAP: int = 1
_BADGE_BG: tuple[int, int, int, int] = (25, 25, 25, 220)
_GHOST_ALPHA: float = 0.15
_PANEL_BG_RGB: tuple[int, int, int] = (30, 30, 30)


def _render_pocket_bg(
    overlay: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    selected: bool,
    building_color: tuple[int, int, int] | None = None,
    frame_tick: int = 0,
) -> None:
    """Draw an inset pocket frame with optional pulsing border.

    Parameters
    ----------
    overlay :
        RGBA overlay to draw on.
    x :
        Left edge.
    y :
        Top edge.
    w :
        Width.
    h :
        Height.
    selected :
        Whether the pocket is currently selected.
    building_color :
        Item RGB color for pulsing border, or None
        for a static white border when selected.
    frame_tick :
        Frame counter for pulse animation.
    overlay : np.ndarray :

    x : int :

    y : int :

    w : int :

    h : int :

    selected : bool :

    building_color : tuple[int :

    int :

    int] | None :
        (Default value = None)
    frame_tick : int :
        (Default value = 0)
    overlay: np.ndarray :

    x: int :

    y: int :

    w: int :

    h: int :

    selected: bool :

    building_color: tuple[int :

    frame_tick: int :
         (Default value = 0)

    Returns
    -------


    """
    oh, ow = overlay.shape[:2]
    if y >= oh or x >= ow or y + h <= 0 or x + w <= 0:
        return

    # Interior fill.
    bg = _POCKET_SELECTED_BG if selected else _POCKET_BG
    y0, y1 = max(0, y), min(oh, y + h)
    x0, x1 = max(0, x), min(ow, x + w)
    overlay[y0:y1, x0:x1] = bg

    # Inset bevel (1px light top/left, dark bottom/right).
    if selected:
        light, dark = _POCKET_SELECTED_LIGHT, _POCKET_SELECTED_DARK
    else:
        light, dark = _POCKET_LIGHT_EDGE, _POCKET_DARK_EDGE
    if 0 <= y < oh:
        overlay[y, x0:x1] = light
    if 0 <= x < ow:
        overlay[y0:y1, x] = light
    if 0 <= y + h - 1 < oh:
        overlay[y + h - 1, x0:x1] = dark
    if 0 <= x + w - 1 < ow:
        overlay[y0:y1, x + w - 1] = dark

    # Round corners by setting to panel bg.
    panel_bg = (*_PANEL_BG_RGB, 228)
    for corner in [(y, x), (y, x + w - 1), (y + h - 1, x), (y + h - 1, x + w - 1)]:
        if 0 <= corner[0] < overlay.shape[0] and 0 <= corner[1] < overlay.shape[1]:
            overlay[corner[0], corner[1]] = panel_bg

    if not selected:
        return

    # Selection border (bounds-safe).
    if building_color is not None:
        phase = (frame_tick // 15) % 2
        color = (255, 255, 255, 255) if phase == 0 else (*building_color, 255)
        for bw in range(2):
            row_t, row_b = y + bw, y + h - 1 - bw
            col_l, col_r = x + bw, x + w - 1 - bw
            if 0 <= row_t < oh:
                overlay[row_t, x0:x1] = color
            if 0 <= row_b < oh:
                overlay[row_b, x0:x1] = color
            if 0 <= col_l < ow:
                overlay[y0:y1, col_l] = color
            if 0 <= col_r < ow:
                overlay[y0:y1, col_r] = color
    else:
        white = (255, 255, 255, 255)
        if 0 <= y < oh:
            overlay[y, x0:x1] = white
        if 0 <= y + h - 1 < oh:
            overlay[y + h - 1, x0:x1] = white
        if 0 <= x < ow:
            overlay[y0:y1, x] = white
        if 0 <= x + w - 1 < ow:
            overlay[y0:y1, x + w - 1] = white


def _render_chips(
    overlay: np.ndarray,
    x: int,
    y: int,
    h: int,
    count: int,
    max_stack: int,
    item_color: tuple[int, int, int],
    num_chips: int = _CHIP_COUNT,
) -> None:
    """Draw stack-fill chips along the right edge of a pocket.

    Chips fill from bottom to top. Lit chips use a desaturated
    version of the item color.

    Parameters
    ----------
    overlay :
        RGBA overlay to draw on.
    x :
        Right inner edge x of the pocket (chips drawn left of this).
    y :
        Top of the chip column area.
    h :
        Height of the chip column area.
    count :
        Current item count.
    max_stack :
        Maximum stack size for this item type.
    item_color :
        RGB tuple for lit chips.
    num_chips :
        Number of chip slots to draw.
    overlay : np.ndarray :

    x : int :

    y : int :

    h : int :

    count : int :

    max_stack : int :

    item_color : tuple[int :

    int :

    int] :

    num_chips : int :
        (Default value = _CHIP_COUNT)
    overlay: np.ndarray :

    x: int :

    y: int :

    h: int :

    count: int :

    max_stack: int :

    item_color: tuple[int :

    num_chips: int :
         (Default value = _CHIP_COUNT)

    Returns
    -------


    """
    if max_stack <= 0:
        return

    chip_x = x - _CHIP_W - 1
    total_chip_h = num_chips * (_CHIP_H + _CHIP_GAP) - _CHIP_GAP
    chip_y0 = y + (h - total_chip_h) // 2

    lit = min(num_chips, int(count / max_stack * num_chips + 0.5))
    is_full = count >= max_stack

    # Desaturate item color to 60% for partial, 100% for full.
    if is_full:
        lit_r, lit_g, lit_b = item_color
    else:
        lit_r = int(item_color[0] * 0.6 + 128 * 0.4)
        lit_g = int(item_color[1] * 0.6 + 128 * 0.4)
        lit_b = int(item_color[2] * 0.6 + 128 * 0.4)

    for i in range(num_chips):
        # Chips fill from bottom (index 0 = bottom).
        chip_idx = num_chips - 1 - i
        cy = chip_y0 + i * (_CHIP_H + _CHIP_GAP)
        if cy + _CHIP_H > overlay.shape[0] or chip_x + _CHIP_W > overlay.shape[1]:
            continue
        if chip_x < 0 or cy < 0:
            continue

        if chip_idx < lit:
            overlay[cy : cy + _CHIP_H, chip_x : chip_x + _CHIP_W] = (
                lit_r,
                lit_g,
                lit_b,
                255,
            )
        else:
            overlay[cy : cy + _CHIP_H, chip_x : chip_x + _CHIP_W] = _CHIP_DIM


def _render_count_badge(
    overlay: np.ndarray,
    x: int,
    y: int,
    count: int,
    font: pygame.font.Font,
) -> int:
    """Draw a pill-shaped count badge.

    Parameters
    ----------
    overlay :
        RGBA overlay to draw on.
    x :
        Right edge x for badge alignment.
    y :
        Bottom edge y for badge alignment.
    count :
        Item count to display.
    font :
        Font for the count text.
    overlay : np.ndarray :

    x : int :

    y : int :

    count : int :

    font : pygame.font.Font :

    overlay: np.ndarray :

    x: int :

    y: int :

    count: int :

    font: pygame.font.Font :


    Returns
    -------


    """
    text_arr = _render_text_rgba(str(count), font, _theme.SLOT_COUNT_COLOR)
    th, tw = int(text_arr.shape[0]), int(text_arr.shape[1])
    pad_x, pad_y = 3, 1
    bw = tw + 2 * pad_x
    bh = th + 2 * pad_y

    # Position: right-aligned to x, bottom-aligned to y.
    bx = x - bw
    by = y - bh

    if bx < 0 or by < 0 or bx + bw > overlay.shape[1] or by + bh > overlay.shape[0]:
        return bw

    # Dark pill background.
    overlay[by : by + bh, bx : bx + bw] = _BADGE_BG

    # Round corners.
    for cy, cx in [
        (by, bx),
        (by, bx + bw - 1),
        (by + bh - 1, bx),
        (by + bh - 1, bx + bw - 1),
    ]:
        if 0 <= cy < overlay.shape[0] and 0 <= cx < overlay.shape[1]:
            overlay[cy, cx, 3] = 0

    # Text.
    _blit_rgba(overlay, text_arr, by + pad_y, bx + pad_x)
    return bw


@functools.lru_cache(maxsize=64)
def _ghost_icon(item_type: int, size: int) -> np.ndarray:
    """Return a cached ghost-opacity version of an item icon.

    Parameters
    ----------
    item_type :
        ItemType value
    size :
        Icon size in pixels
    item_type : int :

    size : int :

    item_type: int :

    size: int :


    Returns
    -------


    """
    icon = render_item_icon(item_type, size)
    ghost = icon.copy()
    ghost[:, :, 3] = (ghost[:, :, 3].astype(np.float32) * _GHOST_ALPHA).astype(
        np.uint8,
    )
    return ghost


def _render_ghost_icon(
    overlay: np.ndarray,
    x: int,
    y: int,
    size: int,
    item_type: int,
) -> None:
    """Render an item icon at ghost opacity.

    Parameters
    ----------
    overlay :
        RGBA overlay to draw on.
    x :
        Left edge x.
    y :
        Top edge y.
    size :
        Icon size in pixels.
    item_type :
        ItemType value.
    overlay : np.ndarray :

    x : int :

    y : int :

    size : int :

    item_type : int :

    overlay: np.ndarray :

    x: int :

    y: int :

    size: int :

    item_type: int :


    Returns
    -------


    """
    _blit_rgba(overlay, _ghost_icon(item_type, size), y, x)


def _render_placement_triangle(
    overlay: np.ndarray,
    cx: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    """Draw a small downward-pointing triangle.

    Parameters
    ----------
    overlay :
        RGBA overlay to draw on.
    cx :
        Center x position.
    y :
        Top edge y of the triangle.
    color :
        RGB fill color.
    overlay : np.ndarray :

    cx : int :

    y : int :

    color : tuple[int :

    int :

    int] :

    overlay: np.ndarray :

    cx: int :

    y: int :

    color: tuple[int :


    Returns
    -------


    """
    # 5px wide, 3px tall filled triangle.
    for row in range(3):
        half = 2 - row  # 2, 1, 0
        for dx in range(-half, half + 1):
            px = cx + dx
            py = y + row
            if 0 <= py < overlay.shape[0] and 0 <= px < overlay.shape[1]:
                overlay[py, px] = (*color, 255)


_BASE_HOTBAR_H: int = 154
"""Base height of the persistent bottom bar at 1x scale."""


def _hotbar_h() -> int:
    """ """
    return _BASE_HOTBAR_H * _theme.UI_SCALE


_DIRECTION_LETTERS: dict[int, str] = {
    int(Direction.LEFT): "W",
    int(Direction.RIGHT): "E",
    int(Direction.UP): "N",
    int(Direction.DOWN): "S",
}

# Gold border for the held-slot highlight in inventory swap mode.
_HELD_BORDER: tuple[int, int, int, int] = (210, 180, 50, 255)


def render_hotbar(
    state: EnvState,
    screen_width: int,
    screen_height: int,
    selected_item: int = 0,
    frame_tick: int = 0,
) -> tuple[np.ndarray, list[ClickRegion]]:
    """Render the building tool belt in the left 2/3 of the bottom bar.

    Shows a player badge on the left and 6 machine pockets in the
    centre. Each pocket uses the inset pocket style with stack chips,
    ghost icons when empty, count badges, and a pulsing border when
    selected for placement.

    Parameters
    ----------
    state :
        Current environment state.
    screen_width :
        Total render width in pixels.
    screen_height :
        Total render height in pixels.
    selected_item :
        Currently selected item type for highlighting.
    frame_tick :
        Frame counter for pulsing animation.
    state : EnvState :

    screen_width : int :

    screen_height : int :

    selected_item : int :
        (Default value = 0)
    frame_tick : int :
        (Default value = 0)
    state: EnvState :

    screen_width: int :

    screen_height: int :

    selected_item: int :
         (Default value = 0)
    frame_tick: int :
         (Default value = 0)

    Returns
    -------


    """
    overlay = np.zeros((screen_height, screen_width, 4), dtype=np.uint8)
    regions: list[ClickRegion] = []

    bar_y = screen_height - _hotbar_h()
    bar_w = screen_width * 2 // 3

    # Full-width background and top border (shared with info panel).
    overlay[bar_y:, :screen_width] = (22, 22, 22, 228)
    overlay[bar_y : bar_y + 2, :screen_width] = _theme.BORDER
    overlay[bar_y + 2 :, bar_w : bar_w + 2] = _theme.BORDER

    selected_player = int(state.selected_player)
    direction = int(state.player_directions[selected_player])
    player_inv = np.array(state.player_inventory[selected_player])

    hint_font = get_pixel_font(_theme.FONT_HINT)

    content_h = 48
    content_y = bar_y + (_hotbar_h() - content_h) // 2

    # --- Player badge (~60px) ---
    badge_x = 8
    badge_cy = content_y + content_h // 2
    circle_r = 14
    colors = PLAYER_COLORS[selected_player % len(PLAYER_COLORS)]
    for dy in range(-circle_r, circle_r + 1):
        half_w = int((circle_r * circle_r - dy * dy) ** 0.5)
        row = badge_cy + dy
        if 0 <= row < screen_height:
            c0 = max(0, badge_x + circle_r - half_w)
            c1 = min(screen_width, badge_x + circle_r + half_w + 1)
            overlay[row, c0:c1, :3] = colors[0]
            overlay[row, c0:c1, 3] = 255

    label = _render_text_rgba(
        f"P{selected_player + 1}",
        hint_font,
        (255, 255, 255),
    )
    _blit_rgba(
        overlay,
        label,
        badge_cy - label.shape[0] // 2 - 1,
        badge_x + circle_r - label.shape[1] // 2,
    )
    dir_letter = _DIRECTION_LETTERS.get(direction, "?")
    dir_arr = _render_text_rgba(dir_letter, hint_font, (200, 195, 160))
    _blit_rgba(
        overlay,
        dir_arr,
        badge_cy - dir_arr.shape[0] // 2 - 1,
        badge_x + circle_r * 2 + 6,
    )

    # --- Machine pockets with [<] [>] [Place] buttons ---
    resource_area_w = 150
    n_slots = len(PLACEABLE_ITEM_LIST)
    arrow_w = 20
    place_btn_w = 44
    btn_gap = 4
    slot_area_x = 64 + arrow_w + btn_gap
    slot_area_w = (
        bar_w
        - 64
        - resource_area_w
        - 8
        - 2 * (arrow_w + btn_gap)
        - place_btn_w
        - btn_gap
    )
    slot_w = slot_area_w // max(n_slots, 1)
    icon_size = min(40, slot_w - 12)
    name_h = hint_font.get_height()
    pocket_h = icon_size
    pocket_y = content_y + (content_h - pocket_h) // 2
    name_y = pocket_y + pocket_h + 2

    # [<] cycle-prev arrow button.
    arrow_x = 64
    arrow_cy = pocket_y + pocket_h // 2
    for dy in range(-5, 6):
        half = 5 - abs(dy)
        lx = arrow_x + (arrow_w // 2) + half
        rx = arrow_x + (arrow_w // 2) - half
        if 0 <= arrow_cy + dy < screen_height and rx <= lx:
            overlay[arrow_cy + dy, rx:lx, :3] = 160
            overlay[arrow_cy + dy, rx:lx, 3] = 255
    regions.append(
        ClickRegion(
            x=arrow_x,
            y=pocket_y,
            w=arrow_w,
            h=pocket_h,
            action="cycle_prev",
            param=0,
        ),
    )

    max_stack_arr = np.array(PLAYER_MAX_STACK)

    for i, item_type in enumerate(PLACEABLE_ITEM_LIST):
        sx = slot_area_x + i * slot_w
        pocket_x = sx + (slot_w - icon_size) // 2

        is_selected = item_type == selected_item
        item_color = ITEM_COLORS.get(item_type, (128, 128, 128))

        _render_pocket_bg(
            overlay,
            pocket_x,
            pocket_y,
            icon_size,
            pocket_h,
            selected=is_selected,
            building_color=item_color if is_selected else None,
            frame_tick=frame_tick,
        )

        count = int(player_inv[item_type])
        max_stack = int(max_stack_arr[item_type])

        icon_pad = 4
        icon_s = icon_size - 2 * icon_pad - _CHIP_W - 1
        if icon_s > 0:
            icon_x = pocket_x + icon_pad
            icon_iy = pocket_y + (pocket_h - icon_s) // 2
            if count > 0:
                icon_arr = render_item_icon(item_type, icon_s)
                overlay[
                    icon_iy : icon_iy + icon_s,
                    icon_x : icon_x + icon_s,
                ] = icon_arr
            else:
                _render_ghost_icon(
                    overlay,
                    icon_x,
                    icon_iy,
                    icon_s,
                    item_type,
                )

        chip_edge_x = pocket_x + icon_size - 1
        _render_chips(
            overlay,
            chip_edge_x,
            pocket_y + 2,
            pocket_h - 4,
            count,
            max_stack,
            item_color,
        )

        if count > 0:
            _render_count_badge(
                overlay,
                pocket_x + icon_size - _CHIP_W - 2,
                pocket_y + pocket_h - 1,
                count,
                hint_font,
            )

        name = _ITEM_NAMES.get(item_type, "")
        name_color = (220, 215, 180) if is_selected else (130, 125, 100)
        if count == 0:
            name_color = (80, 78, 65)
        name_arr = _render_text_rgba(name, hint_font, name_color)
        nx = sx + (slot_w - name_arr.shape[1]) // 2
        _blit_rgba(overlay, name_arr, name_y, nx)

        if is_selected:
            tri_y = name_y + name_h + 1
            tri_cx = pocket_x + icon_size // 2
            _render_placement_triangle(
                overlay,
                tri_cx,
                tri_y,
                item_color,
            )

        regions.append(
            ClickRegion(
                x=pocket_x,
                y=pocket_y,
                w=icon_size,
                h=pocket_h + name_h + 4,
                action="select_slot",
                param=item_type,
            ),
        )

    # [>] cycle-next arrow button.
    arrow_rx = slot_area_x + n_slots * slot_w + btn_gap
    for dy in range(-5, 6):
        half = 5 - abs(dy)
        lx = arrow_rx + (arrow_w // 2) - half
        rx = arrow_rx + (arrow_w // 2) + half
        if 0 <= arrow_cy + dy < screen_height and lx <= rx:
            overlay[arrow_cy + dy, lx:rx, :3] = 160
            overlay[arrow_cy + dy, lx:rx, 3] = 255
    regions.append(
        ClickRegion(
            x=arrow_rx,
            y=pocket_y,
            w=arrow_w,
            h=pocket_h,
            action="cycle_next",
            param=0,
        ),
    )

    # [Place] button.
    place_x = arrow_rx + arrow_w + btn_gap
    place_bg: tuple[int, int, int, int] = (45, 45, 40, 255)
    place_border: tuple[int, int, int, int] = (130, 120, 60, 255)
    overlay[pocket_y : pocket_y + pocket_h, place_x : place_x + place_btn_w] = place_bg
    overlay[pocket_y, place_x : place_x + place_btn_w] = place_border
    overlay[pocket_y + pocket_h - 1, place_x : place_x + place_btn_w] = place_border
    overlay[pocket_y : pocket_y + pocket_h, place_x] = place_border
    overlay[pocket_y : pocket_y + pocket_h, place_x + place_btn_w - 1] = place_border
    place_txt = _render_text_rgba("Place", hint_font, (200, 195, 160))
    ptx = place_x + (place_btn_w - place_txt.shape[1]) // 2
    pty = pocket_y + (pocket_h - place_txt.shape[0]) // 2
    _blit_rgba(overlay, place_txt, pty, ptx)
    regions.append(
        ClickRegion(
            x=place_x,
            y=pocket_y,
            w=place_btn_w,
            h=pocket_h,
            action="place_selected",
            param=0,
        ),
    )

    # --- Resource counts (two-column layout, no research) ---
    res_x = bar_w - resource_area_w
    res_icon_s = 12
    col_w = resource_area_w // 2

    overlay[content_y : content_y + content_h, res_x - 3] = (
        50,
        50,
        45,
        180,
    )

    ore_items = list(BLOCK_TO_ITEM.values())
    res_count_color = (160, 155, 135)
    for oi, item_type in enumerate(ore_items):
        col = oi // 3  # 0 for first 3, 1 for rest
        row = oi % 3
        rx = res_x + col * col_w
        ry = content_y + 2 + row * (res_icon_s + 2)
        count = int(player_inv[int(item_type)])
        icon_arr = render_item_icon(int(item_type), res_icon_s)
        _blit_rgba(overlay, icon_arr, ry, rx)
        count_arr = _render_text_rgba(
            str(count),
            hint_font,
            res_count_color,
        )
        _blit_rgba(overlay, count_arr, ry, rx + res_icon_s + 3)

    return overlay, regions


# Display names for block types shown in the info panel.
# Ore names are derived from BLOCK_TO_ITEM + _ITEM_NAMES so new
# ores are included automatically.
_BLOCK_NAMES: dict[int, str] = {
    int(BlockType.DIRT): "Dirt",
    int(BlockType.WATER): "Water",
    int(BlockType.OUT_OF_BOUNDS): "Out of Bounds",
    **{
        int(bt): f"{_ITEM_NAMES.get(int(it), '?')} Deposit"
        for bt, it in BLOCK_TO_ITEM.items()
    },
}

# Colors for resource block names, derived from ITEM_COLORS so new
# ores get their color automatically.
_BLOCK_COLORS: dict[int, tuple[int, int, int]] = {
    int(bt): ITEM_COLORS.get(int(it), (220, 215, 180))
    for bt, it in BLOCK_TO_ITEM.items()
}

# Items that correspond to mineable blocks (re-exported from constants
# as int keys for the info panel renderer).
_BLOCK_TO_ITEM: dict[int, int] = {int(bt): int(it) for bt, it in BLOCK_TO_ITEM.items()}


def render_info_panel(
    state: EnvState,
    screen_width: int,
    screen_height: int,
    hover_tx: int,
    hover_ty: int,
) -> np.ndarray:
    """Render the tile info panel in the right 1/3 of the bottom bar.

    Shows contextual information about the tile the mouse is hovering
    over: machine details (name, inventory), resource stats, or plain
    block type. When no tile is hovered the panel shows a hint.

    Parameters
    ----------
    state :
        Current environment state.
    screen_width :
        Total render width in pixels.
    screen_height :
        Total render height in pixels.
    hover_tx :
        Hovered tile X coordinate, or -1 if none.
    hover_ty :
        Hovered tile Y coordinate, or -1 if none.
    state : EnvState :

    screen_width : int :

    screen_height : int :

    hover_tx : int :

    hover_ty : int :

    state: EnvState :

    screen_width: int :

    screen_height: int :

    hover_tx: int :

    hover_ty: int :


    Returns
    -------


    """
    overlay = np.zeros((screen_height, screen_width, 4), dtype=np.uint8)

    bar_y = screen_height - _hotbar_h()
    panel_x = screen_width * 2 // 3 + 2  # after separator
    panel_w = screen_width - panel_x
    pad = 10
    cx = panel_x + pad
    cy = bar_y + pad
    max_x = panel_x + panel_w - pad

    hint_font = get_pixel_font(_theme.FONT_HINT)
    body_font = get_pixel_font(_theme.FONT_BODY)
    header_font = get_pixel_font(_theme.FONT_HEADER)

    map_h, map_w = state.map.shape[:2]
    valid = 0 <= hover_tx < map_w and 0 <= hover_ty < map_h

    if not valid:
        # No tile hovered: show dim hint, centred in upper area.
        hint = _render_text_rgba("Hover over the world", hint_font, (100, 95, 70))
        _blit_rgba(
            overlay,
            hint,
            bar_y + 20,
            panel_x + (panel_w - hint.shape[1]) // 2,
        )
    else:
        block_type = int(state.map[hover_ty, hover_tx])
        machine_type = int(state.machine_types[hover_ty, hover_tx])
        has_machine = machine_type != int(Machine.NONE)

        if has_machine:
            cy = _render_info_machine(
                overlay,
                state,
                cx,
                cy,
                max_x,
                hover_tx,
                hover_ty,
                machine_type,
                header_font,
                body_font,
                hint_font,
            )
        else:
            cy = _render_info_terrain(
                overlay,
                state,
                cx,
                cy,
                max_x,
                hover_tx,
                hover_ty,
                block_type,
                header_font,
                body_font,
                hint_font,
            )

    return overlay


def _render_info_machine(
    overlay: np.ndarray,
    state: EnvState,
    cx: int,
    cy: int,
    max_x: int,
    tx: int,
    ty: int,
    machine_type: int,
    header_font: pygame.font.Font,
    body_font: pygame.font.Font,
    hint_font: pygame.font.Font,
) -> int:
    """Draw machine info into the info panel overlay.

    Parameters
    ----------
    overlay :
        Destination RGBA array, modified in place.
    state :
        Current environment state.
    cx :
        Content left X.
    cy :
        Content top Y.
    max_x :
        Content right edge X.
    tx :
        Machine tile X.
    ty :
        Machine tile Y.
    machine_type :
        Machine int value.
    header_font :
        Font for the machine name header.
    body_font :
        Font for body text.
    hint_font :
        Font for small labels.
    overlay : np.ndarray :

    state : EnvState :

    cx : int :

    cy : int :

    max_x : int :

    tx : int :

    ty : int :

    machine_type : int :

    header_font : pygame.font.Font :

    body_font : pygame.font.Font :

    hint_font : pygame.font.Font :

    overlay: np.ndarray :

    state: EnvState :

    cx: int :

    cy: int :

    max_x: int :

    tx: int :

    ty: int :

    machine_type: int :

    header_font: pygame.font.Font :

    body_font: pygame.font.Font :

    hint_font: pygame.font.Font :


    Returns
    -------


    """
    name = MACHINE_TYPE_NAMES.get(machine_type, "Unknown")
    icon_s = 24
    machine_inv = _entity_inventory(state, ty, tx)
    first_item = next(
        (i for i in range(1, NUM_ITEM_TYPES) if machine_inv[i] > 0),
        machine_type,
    )
    icon_arr = render_item_icon(first_item, icon_s)

    # Header: icon + name.
    name_surf = _render_text_rgba(name, header_font, (220, 215, 180))
    _blit_rgba(overlay, icon_arr, cy, cx)
    _blit_rgba(
        overlay,
        name_surf,
        cy + (icon_s - name_surf.shape[0]) // 2,
        cx + icon_s + 6,
    )
    cy += max(icon_s, name_surf.shape[0]) + 6

    # Facing direction.
    eidx_dir = int(state.tile_entity[ty, tx])
    direction = int(state.ent_direction[eidx_dir]) if eidx_dir >= 0 else 0
    dir_letter = _DIRECTION_LETTERS.get(direction, "?")
    dir_txt = _render_text_rgba(
        f"Facing: {dir_letter}",
        hint_font,
        (180, 175, 140),
    )
    _blit_rgba(overlay, dir_txt, cy, cx)
    cy += dir_txt.shape[0] + 6

    # Inventory summary: show non-empty item types as icon + count.
    ix = cx
    slot_icon_s = 20
    for item_t in range(1, NUM_ITEM_TYPES):
        count = int(machine_inv[item_t])
        if count == 0:
            continue
        if ix + slot_icon_s + 30 > max_x:
            break
        slot_arr = render_item_icon(item_t, slot_icon_s)
        _blit_rgba(overlay, slot_arr, cy, ix)
        cnt = _render_text_rgba(
            f"x{count}",
            hint_font,
            (220, 215, 180),
        )
        _blit_rgba(overlay, cnt, cy + 3, ix + slot_icon_s + 2)
        ix += slot_icon_s + cnt.shape[1] + 8

    return cy


def _render_info_terrain(
    overlay: np.ndarray,
    state: EnvState,
    cx: int,
    cy: int,
    max_x: int,
    tx: int,
    ty: int,
    block_type: int,
    header_font: pygame.font.Font,
    body_font: pygame.font.Font,
    hint_font: pygame.font.Font,
) -> int:
    """Draw terrain tile info into the info panel overlay.

    Parameters
    ----------
    overlay :
        Destination RGBA array, modified in place.
    state :
        Current environment state.
    cx :
        Content left X.
    cy :
        Content top Y.
    max_x :
        Content right edge X.
    tx :
        Tile X.
    ty :
        Tile Y.
    block_type :
        BlockType int value.
    header_font :
        Font for the tile name header.
    body_font :
        Font for body text.
    hint_font :
        Font for small labels.
    overlay : np.ndarray :

    state : EnvState :

    cx : int :

    cy : int :

    max_x : int :

    tx : int :

    ty : int :

    block_type : int :

    header_font : pygame.font.Font :

    body_font : pygame.font.Font :

    hint_font : pygame.font.Font :

    overlay: np.ndarray :

    state: EnvState :

    cx: int :

    cy: int :

    max_x: int :

    tx: int :

    ty: int :

    block_type: int :

    header_font: pygame.font.Font :

    body_font: pygame.font.Font :

    hint_font: pygame.font.Font :


    Returns
    -------


    """
    name = _BLOCK_NAMES.get(block_type, "Unknown")
    color = _BLOCK_COLORS.get(block_type, (220, 215, 180))

    # Header: icon (if resource) + name.
    item_type = _BLOCK_TO_ITEM.get(block_type)
    icon_s = 24
    if item_type is not None:
        icon_arr = render_item_icon(item_type, icon_s)
        _blit_rgba(overlay, icon_arr, cy, cx)
        name_x = cx + icon_s + 6
    else:
        name_x = cx

    name_surf = _render_text_rgba(name, header_font, color)
    _blit_rgba(
        overlay,
        name_surf,
        cy + (icon_s - name_surf.shape[0]) // 2 if item_type else cy,
        name_x,
    )
    cy += max(icon_s if item_type else name_surf.shape[0], name_surf.shape[0]) + 6

    # Resource remaining (for mineable blocks).
    if item_type is not None:
        remaining = int(state.block_resources[ty, tx])
        res_txt = _render_text_rgba(
            f"Resources: {remaining} / {BLOCK_MAX_RESOURCES}",
            body_font,
            (180, 175, 140),
        )
        _blit_rgba(overlay, res_txt, cy, cx)
        cy += res_txt.shape[0] + 4

    # Position.
    pos_txt = _render_text_rgba(
        f"({tx}, {ty})",
        hint_font,
        (130, 125, 100),
    )
    _blit_rgba(overlay, pos_txt, cy, cx)
    cy += pos_txt.shape[0] + 4

    return cy


def render_inventory_menu(
    state: EnvState,
    params: EnvParams,
    screen_width: int,
    screen_height: int,
    selected_recipe: int = 0,
) -> tuple[np.ndarray, list[ClickRegion]]:
    """Render the crafting menu as an RGBA overlay.

    Shows a scrollable recipe list with output icons, names, and
    per-ingredient have/need counts coloured by affordability. The hotbar
    selects a machine, so this menu holds no machine selection.

    Parameters
    ----------
    state
        Current environment state.
    params
        Environment parameters, read for this scenario's recipe table.
    screen_width
        Total screen width in pixels.
    screen_height
        Total screen height in pixels.
    selected_recipe
        Index of the focused recipe.

    Returns
    -------
    tuple[numpy.ndarray, list[ClickRegion]]
        The RGBA overlay, and the click regions it drew.
    """
    overlay = np.zeros((screen_height, screen_width, 4), dtype=np.uint8)
    click_regions: list[ClickRegion] = []

    menu_w = int(screen_width * 0.45)
    menu_h = int(screen_height * 0.55)
    menu_x = (screen_width - menu_w) // 2
    menu_y = (screen_height - menu_h) // 2

    draw_panel(overlay, menu_x, menu_y, menu_w, menu_h)

    header_font = get_pixel_font(_theme.FONT_HEADER)
    body_font = get_pixel_font(_theme.FONT_BODY)

    craft_content_y = _draw_section_header(
        overlay,
        menu_x,
        menu_y,
        menu_w,
        "CRAFTING",
        header_font,
        True,
    )

    selected_player = int(state.selected_player)
    craft_progress = 0

    # ------------------------------------------------------------------
    # Crafting recipes list — rendered into a scroll view
    # ------------------------------------------------------------------
    hint_y = menu_y + menu_h - _theme.HINT_HEIGHT - _theme.BORDER_PX

    vp_x_craft = menu_x + _theme.BORDER_PX
    vp_y_craft = craft_content_y
    vp_w_craft = menu_w - 2 * _theme.BORDER_PX
    vp_h_craft = hint_y - craft_content_y

    recipe_h = 80
    craft_pad = 8
    out_icon = 32
    inp_icon = 20

    # This scenario's book, not BASE_RECIPES.
    table = params.recipe_table
    num_recipes = int(table.outputs.shape[0])

    content_h_craft = num_recipes * recipe_h
    craft_scroll = clip_scroll_offset(
        selected_recipe * recipe_h - (vp_h_craft - recipe_h) // 2,
        content_h_craft,
        vp_h_craft,
    )

    recipe_content = np.zeros((max(content_h_craft, 1), vp_w_craft, 4), dtype=np.uint8)
    recipe_regions: list[ClickRegion] = []

    for recipe_idx in range(num_recipes):
        is_selected_recipe = recipe_idx == selected_recipe
        can_afford = bool(can_afford_recipe(state, params, selected_player, recipe_idx))
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

        out_item = int(table.outputs[recipe_idx])
        out_s = min(out_icon, vp_w_craft - craft_pad)
        if out_s > 0:
            out_icon_arr = render_item_icon(out_item, out_s)
            recipe_content[
                ry + craft_pad : ry + craft_pad + out_s,
                craft_pad : craft_pad + out_s,
            ] = out_icon_arr

        name_color = (230, 225, 180) if can_afford else (150, 145, 120)
        name_arr = _render_text_rgba(table.names[recipe_idx], body_font, name_color)
        name_y = ry + craft_pad + (out_icon - name_arr.shape[0]) // 2
        _blit_rgba(recipe_content, name_arr, name_y, craft_pad + out_icon + 12)

        inp_y = ry + craft_pad + out_icon + craft_pad
        inp_x = craft_pad

        for slot in range(int(table.input_items.shape[1])):
            item_type = int(table.input_items[recipe_idx, slot])
            required = int(table.input_counts[recipe_idx, slot])
            if item_type == int(ItemType.EMPTY):  # padded slot
                continue
            have = int(count_item_in_inventory(state, selected_player, item_type))
            inp_s = min(inp_icon, vp_w_craft - inp_x)
            if inp_s > 0:
                inp_icon_arr = render_item_icon(item_type, inp_s)
                recipe_content[inp_y : inp_y + inp_s, inp_x : inp_x + inp_s] = (
                    inp_icon_arr
                )
            count_color: tuple[int, int, int] = (
                _AFFORD_COLOR if have >= required else _CANNOT_AFFORD_COLOR
            )
            ratio_arr = _render_text_rgba(f"{have}/{required}", body_font, count_color)
            _blit_rgba(recipe_content, ratio_arr, inp_y, inp_x + inp_icon + 6)
            inp_x += inp_icon + 6 + ratio_arr.shape[1] + 12

        recipe_ticks = int(table.ticks[recipe_idx])
        if craft_progress > 0 and is_selected_recipe and recipe_ticks > 0:
            bar_y = ry + recipe_h - 20
            bar_w = vp_w_craft - 16
            filled = int(bar_w * (1 - craft_progress / recipe_ticks))
            recipe_content[bar_y : bar_y + 8, craft_pad : craft_pad + bar_w] = (
                35,
                35,
                35,
                255,
            )
            if filled > 0:
                recipe_content[bar_y : bar_y + 8, craft_pad : craft_pad + filled] = (
                    100,
                    200,
                    100,
                    255,
                )

    blit_scroll_view(
        overlay,
        recipe_content,
        vp_x_craft,
        vp_y_craft,
        vp_w_craft,
        vp_h_craft,
        craft_scroll,
    )
    click_regions.extend(
        scroll_adjust_regions(
            recipe_regions, vp_x_craft, vp_y_craft, vp_h_craft, craft_scroll
        )
    )

    hints = "[W/S] Select | [E] Craft | [ESC] Close"
    _render_control_hints(
        overlay,
        hints,
        menu_x + _theme.BORDER_PX,
        hint_y,
        menu_w - 2 * _theme.BORDER_PX,
    )

    return overlay, click_regions


# ---------------------------------------------------------------------------
# Help overlay
# ---------------------------------------------------------------------------

_HELP_LINES: list[str] = [
    "-- Movement --",
    "WASD / D-pad    Move player",
    "Ctrl+1-9        Switch active player",
    "",
    "-- Actions --",
    "SPACE           Mine ore at current tile",
    "E / A-button    Place / pick up machine",
    "R               Rotate machine in front",
    "G               Repair machine in front",
    "1-5             Quick-select machine",
    "",
    "-- Crafting --",
    "I               Open crafting menu",
    "W/S             Select recipe",
    "E               Craft selected recipe",
    "",
    "-- Machine Transfer --",
    "F               Inspect machine in front",
    "W/S             Switch panel",
    "A/D             Select item",
    "E               Transfer items",
    "",
    "-- Research --",
    "T               Research menu",
    "W/S             Select technology",
    "E               Spend science pack",
    "",
    "-- Other --",
    "P               Achievements",
    "ESC / B-button  Close menu / pause",
]


def render_help_overlay(
    screen_width: int,
    screen_height: int,
) -> np.ndarray:
    """Render a controls reference overlay for play mode.

    Displays all keybindings grouped by category. Press any key
    to dismiss.

    Parameters
    ----------
    screen_width :
        Total render width in pixels.
    screen_height :
        Total render height in pixels.
    screen_width : int :

    screen_height : int :

    screen_width: int :

    screen_height: int :


    Returns
    -------


    """
    overlay = np.zeros((screen_height, screen_width, 4), dtype=np.uint8)
    overlay[:, :] = (0, 0, 0, 180)

    font = get_pixel_font(_theme.FONT_HINT)
    title_font = get_pixel_font(_theme.FONT_HEADER)
    line_h = font.get_height() + 4

    total_h = len(_HELP_LINES) * line_h + 48
    panel_w = min(420, screen_width - 40)
    panel_h = min(total_h, screen_height - 40)
    px = (screen_width - panel_w) // 2
    py = (screen_height - panel_h) // 2

    draw_panel(overlay, px, py, panel_w, panel_h)

    title = _render_text_rgba("Controls", title_font, (215, 195, 65))
    _blit_rgba(
        overlay,
        title,
        py + _theme.BORDER_PX + 6,
        px + (panel_w - title.shape[1]) // 2,
    )

    cy = py + _theme.BORDER_PX + 6 + title.shape[0] + 8
    overlay[cy : cy + _theme.SEP_H, px + 16 : px + panel_w - 16] = _theme.BORDER
    cy += _theme.SEP_H + 6

    for line in _HELP_LINES:
        if not line:
            cy += line_h // 2
            continue
        if line.startswith("--"):
            arr = _render_text_rgba(line, font, (215, 195, 65))
        else:
            arr = _render_text_rgba(line, font, (180, 175, 150))
        _blit_rgba(overlay, arr, cy, px + 16)
        cy += line_h

    dismiss = _render_text_rgba("Press any key to close", font, _theme.HINT_COLOR)
    _blit_rgba(
        overlay,
        dismiss,
        py + panel_h - _theme.BORDER_PX - dismiss.shape[0] - 4,
        px + (panel_w - dismiss.shape[1]) // 2,
    )

    return overlay
