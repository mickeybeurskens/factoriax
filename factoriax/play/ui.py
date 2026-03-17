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


from factoriax.constants import (
    ITEM_COLORS,
    NUM_INVENTORY_SLOTS,
    NUM_RECIPES,
    RECIPE_NAMES,
    RECIPES,
    ItemType,
)
from factoriax.crafting import can_afford_recipe, count_item_in_inventory
from factoriax.state import EnvState

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
_HEADER_H: int = 44       # height reserved for each section label row
_SEP_H: int = 4           # height of the gold separator beneath labels
_FONT_HEADER: int = 26    # section label font size
_FONT_BODY: int = 20      # item names, counts, recipe info font size
_FONT_HINT: int = 14      # control hint font size
_HINT_HEIGHT: int = 24    # height reserved for hint bar at bottom of menus
_HINT_COLOR: tuple[int, int, int] = (120, 115, 90)

# Crafting ingredient affordability colours.
_AFFORD_COLOR: tuple[int, int, int] = (110, 220, 110)
_CANNOT_AFFORD_COLOR: tuple[int, int, int] = (200, 80, 80)

# Item display names keyed by ItemType value.
_ITEM_NAMES: dict[int, str] = {
    ItemType.COAL: "Coal",
    ItemType.IRON: "Iron",
    ItemType.COPPER: "Copper",
    ItemType.MINER: "Miner",
}

# Comma-separated preference list for pygame.font.SysFont.  Terminus is a
# 1:1 pixel bitmap font common on Linux; the rest are fallbacks.
_PIXEL_FONT_PREFERENCE = (
    "terminus,fixedsys excelsior,courier new,monospace,courier"
)


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


def _render_text_rgba(
    text: str,
    font: pygame.font.Font,
    color: tuple[int, int, int],
) -> np.ndarray:
    """Render text to an RGBA array with a fully transparent background.

    antialias=False keeps every pixel either the exact glyph colour or
    transparent, which is what gives the pixelated look.

    Args:
        text: String to render.
        font: pygame Font to use.
        color: RGB glyph colour.

    Returns:
        RGBA numpy array of shape (H, W, 4).
    """
    surface = font.render(text, False, color)
    w, h = surface.get_size()
    # surfarray returns (W, H, 3); transpose to (H, W, 3).
    rgb = pygame.surfarray.array3d(surface).transpose(1, 0, 2)
    result = np.zeros((h, w, 4), dtype=np.uint8)
    result[:, :, :3] = rgb
    result[:, :, 3] = np.where(np.any(rgb != 0, axis=2), 255, 0)
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
# Public menu renderers
# ---------------------------------------------------------------------------


def render_achievement_menu(
    state: EnvState,
    screen_width: int,
    screen_height: int,
) -> np.ndarray:
    """Render the achievement menu as an RGBA overlay.

    Displays every achievement with a colour-coded status icon (bright green
    = unlocked, dark grey = locked) and its name in a pixely monospace font.
    A footer shows the overall completion count.

    Args:
        state: Current environment state.
        screen_width: Total render width in pixels.
        screen_height: Total render height in pixels.

    Returns:
        RGBA numpy array of shape (screen_height, screen_width, 4).
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

    footer_reserve = 56
    available_h = menu_h - (sep_y - menu_y) - 16 - footer_reserve
    row_h = min(76, available_h // max(NUM_ACHIEVEMENTS, 1))
    list_y = sep_y + _SEP_H + 16

    for i, info in enumerate(ACHIEVEMENT_INFO):
        is_unlocked = bool(unlocked[i])
        row_y = list_y + i * row_h

        if is_unlocked:
            overlay[
                row_y : row_y + row_h - 4,
                menu_x + 8 : menu_x + menu_w - 8,
            ] = (42, 68, 42, 210)

        icon_size = 20
        icon_x = menu_x + 32
        icon_y = row_y + (row_h - icon_size) // 2
        icon_color: tuple[int, int, int, int] = (
            (75, 215, 75, 255) if is_unlocked else (65, 65, 65, 255)
        )
        overlay[icon_y : icon_y + icon_size, icon_x : icon_x + icon_size] = icon_color

        text_color: tuple[int, int, int] = (
            (235, 228, 185) if is_unlocked else (105, 105, 88)
        )
        name_arr = _render_text_rgba(info.name, body_font, text_color)
        name_y = row_y + (row_h - name_arr.shape[0]) // 2
        _blit_rgba(overlay, name_arr, name_y, icon_x + icon_size + 20)

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
        overlay, "[ESC] Close", menu_x + _BORDER_PX, hint_y, menu_w - 2 * _BORDER_PX
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
        option_bg = _PAUSE_OPTION_SELECTED if i == selected_option else _PAUSE_OPTION_NORMAL

        option_x = menu_x + 24
        option_w = menu_w - 48
        overlay[
            option_y : option_y + option_h,
            option_x : option_x + option_w,
        ] = option_bg

        click_regions.append(ClickRegion(
            x=option_x, y=option_y, w=option_w, h=option_h,
            action="pause_option", param=i,
        ))

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
        overlay, "[UP/DOWN] Select | [ENTER/E] Confirm | [ESC] Back",
        menu_x + _BORDER_PX, hint_y, menu_w - 2 * _BORDER_PX,
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
        overlay, menu_x, menu_y, inv_w,
        "INVENTORY", header_font, menu_focus == "inventory",
    )
    craft_content_y = _draw_section_header(
        overlay, div_x, menu_y, craft_w,
        "CRAFTING", header_font, menu_focus == "crafting",
    )

    click_regions.append(ClickRegion(
        x=menu_x, y=menu_y, w=inv_w, h=inv_content_y - menu_y,
        action="focus_inventory", param=0,
    ))
    click_regions.append(ClickRegion(
        x=div_x, y=menu_y, w=craft_w, h=craft_content_y - menu_y,
        action="focus_crafting", param=0,
    ))

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

        click_regions.append(ClickRegion(
            x=icon_x, y=cell_y, w=icon_size, h=cell_h,
            action="select_slot", param=slot_idx,
        ))

        item_type = int(inventory_items[slot_idx])
        count = int(inventory_counts[slot_idx])

        if item_type != 0 and count > 0:
            rgb = ITEM_COLORS.get(item_type, (128, 128, 128))
            pad = 8
            overlay[
                cell_y + pad : cell_y + icon_size - pad,
                icon_x + pad : icon_x + icon_size - pad,
            ] = (*rgb, 255)

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
    # Crafting recipes list
    # ------------------------------------------------------------------
    craft_pad = 20
    craft_x = div_x + craft_pad
    craft_available_w = craft_w - 2 * craft_pad
    craft_available_h = menu_h - (craft_content_y - menu_y) - craft_pad
    recipe_h = min(120, craft_available_h // max(NUM_RECIPES, 1))

    for recipe_idx in range(NUM_RECIPES):
        recipe = RECIPES[recipe_idx]
        is_selected_recipe = (recipe_idx == selected_recipe) and (
            menu_focus == "crafting"
        )
        can_afford = bool(can_afford_recipe(state, selected_player, recipe_idx))

        recipe_y = craft_content_y + recipe_idx * recipe_h

        click_regions.append(ClickRegion(
            x=craft_x, y=recipe_y, w=craft_available_w, h=recipe_h,
            action="select_recipe", param=recipe_idx,
        ))

        if is_selected_recipe:
            overlay[
                recipe_y : recipe_y + recipe_h - 4,
                craft_x : craft_x + craft_available_w,
            ] = (65, 65, 65, 255)
            white = (255, 255, 255, 255)
            overlay[recipe_y, craft_x : craft_x + craft_available_w] = white
            overlay[recipe_y + recipe_h - 5, craft_x : craft_x + craft_available_w] = white
            overlay[recipe_y : recipe_y + recipe_h - 4, craft_x] = white
            overlay[recipe_y : recipe_y + recipe_h - 4, craft_x + craft_available_w - 1] = white

        # Output icon.
        out_icon = 32
        out_item = recipe["output"]
        out_rgb = ITEM_COLORS.get(out_item, (128, 128, 128))
        overlay[
            recipe_y + 8 : recipe_y + 8 + out_icon,
            craft_x + 8 : craft_x + 8 + out_icon,
        ] = (*out_rgb, 255)

        # Recipe name to the right of the icon.
        name_color = (230, 225, 180) if can_afford else (150, 145, 120)
        name_arr = _render_text_rgba(RECIPE_NAMES[recipe_idx], body_font, name_color)
        name_y = recipe_y + 8 + (out_icon - name_arr.shape[0]) // 2
        _blit_rgba(overlay, name_arr, name_y, craft_x + 8 + out_icon + 12)

        # Ingredient row: [small icon] [have/need] for each input.
        inp_y = recipe_y + 8 + out_icon + 8
        inp_x = craft_x + 8
        inp_icon = 20

        for item_type, required in recipe["inputs"]:
            have = int(count_item_in_inventory(state, selected_player, item_type))
            inp_rgb = ITEM_COLORS.get(item_type, (128, 128, 128))
            overlay[inp_y : inp_y + inp_icon, inp_x : inp_x + inp_icon] = (*inp_rgb, 255)

            count_color: tuple[int, int, int] = (
                _AFFORD_COLOR if have >= required else _CANNOT_AFFORD_COLOR
            )
            ratio_arr = _render_text_rgba(f"{have}/{required}", body_font, count_color)
            _blit_rgba(overlay, ratio_arr, inp_y, inp_x + inp_icon + 6)
            inp_x += inp_icon + 6 + ratio_arr.shape[1] + 12

        # Progress bar on the selected recipe.
        if craft_progress > 0 and is_selected_recipe:
            bar_y = recipe_y + recipe_h - 20
            bar_w = craft_available_w - 16
            filled = int(bar_w * (1 - craft_progress / recipe["ticks"]))
            overlay[bar_y : bar_y + 8, craft_x + 8 : craft_x + 8 + bar_w] = (
                35, 35, 35, 255
            )
            if filled > 0:
                overlay[bar_y : bar_y + 8, craft_x + 8 : craft_x + 8 + filled] = (
                    100, 200, 100, 255
                )

    hint_y = menu_y + menu_h - _HINT_HEIGHT - _BORDER_PX
    if menu_focus == "crafting":
        hints = "[UP/DOWN] Select | [TAB] Inventory | [E] Craft | [ESC] Close"
    else:
        hints = "[LEFT/RIGHT] Select | [TAB] Crafting | [E] Place | [ESC] Close"
    _render_control_hints(
        overlay, hints, menu_x + _BORDER_PX, hint_y, menu_w - 2 * _BORDER_PX
    )

    return overlay, click_regions
