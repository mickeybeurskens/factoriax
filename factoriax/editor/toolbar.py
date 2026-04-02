"""Sidebar toolbar, menu bar, and status bar for the level editor.

Renders all chrome around the canvas: tool palette, block selector,
resource brush panel, machine palette, entity palette, top menu bar,
and bottom status bar.  Each render function returns an RGB or RGBA image and a list of
:class:`~factoriax.play.ui.ClickRegion` objects for hit-testing.
"""

from __future__ import annotations

import numpy as np
import pygame

from factoriax.constants import (
    ITEM_TO_MACHINE,
    MACHINE_SLOT_ROLES,
    MACHINE_TYPE_NAMES,
    BlockType,
    ItemType,
    MachineType,
    SlotRole,
)
from factoriax.renderer import (
    create_biter_texture,
    create_player_start_icon,
    get_textures,
    render_item_icon,
)
from factoriax.ui.fonts import get_pixel_font
from factoriax.ui.primitives import ClickRegion

TOOLBAR_WIDTH = 120
MENU_BAR_HEIGHT = 32
STATUS_BAR_HEIGHT = 24
MAX_EDITOR_PLAYERS = 8

_BG = (30, 30, 30)
_MENU_BG = (22, 22, 22)
_STATUS_BG = (22, 22, 22)
_ACTIVE_TOOL_BG = (60, 90, 60)
_SELECTED_BORDER = (255, 255, 255)
_TEXT_COLOR = (200, 200, 200)
_ACCENT_COLOR = (190, 165, 55)
_DIM_TEXT = (120, 120, 120)

TOOL_PAINT = "paint"
TOOL_FILL = "fill"
TOOL_ERASE = "erase"

# Auto-derive palette entries from the enum definitions so new types
# appear in the editor without manual updates.

BLOCK_ITEMS: list[tuple[int, str]] = [
    (int(b), b.name.capitalize())
    for b in BlockType
    if b not in (BlockType.INVALID, BlockType.OUT_OF_BOUNDS)
]

MACHINE_ITEMS: list[tuple[int, str]] = [
    (int(m), MACHINE_TYPE_NAMES.get(int(m), m.name.capitalize()))
    for m in MachineType
    if m != MachineType.NONE
]

MACHINE_TO_ITEM_MAP: dict[int, int] = {
    int(machine): int(item) for item, machine in ITEM_TO_MACHINE.items()
}


def _render_text(
    text: str, font: pygame.font.Font, color: tuple[int, int, int]
) -> np.ndarray:
    """Render text to an RGB numpy array.

    Args:
        text: String to render.
        font: Pygame font.
        color: RGB text colour.

    Returns:
        RGB uint8 array of shape ``(H, W, 3)``.
    """
    surface = font.render(text, False, color)
    return pygame.surfarray.array3d(surface).transpose(1, 0, 2)


def render_menu_bar(
    width: int,
) -> tuple[np.ndarray, list[ClickRegion]]:
    """Render the top menu bar with New / Load / Save / Play buttons.

    Args:
        width: Full window width in pixels.

    Returns:
        ``(image, regions)`` where *image* is RGB shape
        ``(MENU_BAR_HEIGHT, width, 3)`` and *regions* contains
        click regions for each button.
    """
    bar = np.full((MENU_BAR_HEIGHT, width, 3), _MENU_BG, dtype=np.uint8)
    font = get_pixel_font(14)

    buttons = ["New", "Load", "Save", "Play"]
    actions = ["new", "load", "save", "play"]
    regions: list[ClickRegion] = []

    btn_w = 60
    btn_h = 24
    btn_y = (MENU_BAR_HEIGHT - btn_h) // 2
    gap = 8

    for i, (label, action) in enumerate(zip(buttons, actions)):
        bx = gap + i * (btn_w + gap)
        bar[btn_y : btn_y + btn_h, bx : bx + btn_w] = (50, 50, 50)
        bar[btn_y, bx : bx + btn_w] = _ACCENT_COLOR
        bar[btn_y + btn_h - 1, bx : bx + btn_w] = _ACCENT_COLOR
        bar[btn_y : btn_y + btn_h, bx] = _ACCENT_COLOR
        bar[btn_y : btn_y + btn_h, bx + btn_w - 1] = _ACCENT_COLOR

        txt = _render_text(label, font, _TEXT_COLOR)
        th, tw = txt.shape[:2]
        tx = bx + (btn_w - tw) // 2
        ty = btn_y + (btn_h - th) // 2
        _blit_rgb(bar, txt, ty, tx)

        regions.append(
            ClickRegion(x=bx, y=btn_y, w=btn_w, h=btn_h, action=action, param=0)
        )

    return bar, regions


def render_toolbar(
    selected_tool: str,
    selected_block: int,
    selected_machine: int,
    direction: int,
    resource_brush: object,
    height: int,
    show_resources: bool = False,
    selected_entity: tuple[str, int] | None = None,
) -> tuple[np.ndarray, list[ClickRegion]]:
    """Render the sidebar toolbar with all palette sections.

    Args:
        selected_tool: Active tool name (``"paint"``, ``"fill"``, ``"erase"``).
        selected_block: Active ``BlockType`` value.
        selected_machine: Active ``MachineType`` value, or 0 for none.
        direction: Current machine placement direction.
        resource_brush: :class:`~factoriax.editor.state.ResourceBrush`.
        height: Available height for the toolbar in pixels.
        show_resources: Whether the resource overlay is active.
        selected_entity: Active entity selection as ``(kind, index)``
            where *kind* is ``"player"`` or ``"biter"``, or ``None``.

    Returns:
        ``(image, regions)`` where *image* is RGB shape
        ``(height, TOOLBAR_WIDTH, 3)``.
    """
    from factoriax.editor.state import ResourceBrush

    brush: ResourceBrush = resource_brush  # type: ignore[assignment]
    # Render at full content height so small maps don't clip icons.
    # The caller slices the visible portion using a scroll offset.
    render_h = max(height, 800)
    bar = np.full((render_h, TOOLBAR_WIDTH, 3), _BG, dtype=np.uint8)
    regions: list[ClickRegion] = []
    font = get_pixel_font(12)
    y = 4

    label_txt = _render_text("Tools", font, _ACCENT_COLOR)
    _blit_rgb(bar, label_txt, y, 4)
    y += label_txt.shape[0] + 4

    tools = [(TOOL_PAINT, "Paint"), (TOOL_FILL, "Fill"), (TOOL_ERASE, "Erase")]
    btn_h = 20
    for tool_id, label in tools:
        bg = _ACTIVE_TOOL_BG if tool_id == selected_tool else (50, 50, 50)
        bar[y : y + btn_h, 4 : TOOLBAR_WIDTH - 4] = bg
        txt = _render_text(label, font, _TEXT_COLOR)
        _blit_rgb(bar, txt, y + (btn_h - txt.shape[0]) // 2, 8)
        regions.append(
            ClickRegion(
                x=4,
                y=y,
                w=TOOLBAR_WIDTH - 8,
                h=btn_h,
                action="tool",
                param=tools.index((tool_id, label)),
            )
        )
        y += btn_h + 2

    y += 8
    label_txt = _render_text("Blocks", font, _ACCENT_COLOR)
    _blit_rgb(bar, label_txt, y, 4)
    y += label_txt.shape[0] + 4

    swatch_size = 18
    textures = get_textures(swatch_size)
    for idx, (block_id, name) in enumerate(BLOCK_ITEMS):
        tex = textures.get(block_id)
        bx = 4
        if tex is not None:
            bar[y : y + swatch_size, bx : bx + swatch_size] = tex[:, :, :3]
        if block_id == selected_block and selected_machine == 0:
            _draw_border(bar, bx, y, swatch_size, swatch_size, _SELECTED_BORDER)
        txt = _render_text(name, font, _TEXT_COLOR)
        _blit_rgb(bar, txt, y + (swatch_size - txt.shape[0]) // 2, bx + swatch_size + 4)
        regions.append(
            ClickRegion(
                x=4, y=y, w=TOOLBAR_WIDTH - 8, h=swatch_size, action="block", param=idx
            )
        )
        y += swatch_size + 2

    y += 8
    label_txt = _render_text("Resource", font, _ACCENT_COLOR)
    _blit_rgb(bar, label_txt, y, 4)
    y += label_txt.shape[0] + 4

    mode_label = "Exact" if brush.mode == "exact" else "Range"
    mode_txt = _render_text(mode_label, font, _TEXT_COLOR)
    _blit_rgb(bar, mode_txt, y, 8)
    regions.append(
        ClickRegion(
            x=4,
            y=y,
            w=TOOLBAR_WIDTH - 8,
            h=16,
            action="toggle_res_mode",
            param=0,
        )
    )
    y += 18

    field_h = 16
    if brush.mode == "exact":
        val_txt = _render_text(str(brush.exact_value), font, _TEXT_COLOR)
        bar[y : y + field_h, 8 : TOOLBAR_WIDTH - 8] = (45, 45, 45)
        _blit_rgb(bar, val_txt, y + 2, 12)
        regions.append(
            ClickRegion(
                x=8,
                y=y,
                w=TOOLBAR_WIDTH - 16,
                h=field_h,
                action="edit_res_exact",
                param=0,
            )
        )
        y += field_h + 4
    else:
        min_txt = _render_text(f"Min: {brush.range_min}", font, _TEXT_COLOR)
        bar[y : y + field_h, 8 : TOOLBAR_WIDTH - 8] = (45, 45, 45)
        _blit_rgb(bar, min_txt, y + 2, 12)
        regions.append(
            ClickRegion(
                x=8,
                y=y,
                w=TOOLBAR_WIDTH - 16,
                h=field_h,
                action="edit_res_min",
                param=0,
            )
        )
        y += field_h + 2

        max_txt = _render_text(f"Max: {brush.range_max}", font, _TEXT_COLOR)
        bar[y : y + field_h, 8 : TOOLBAR_WIDTH - 8] = (45, 45, 45)
        _blit_rgb(bar, max_txt, y + 2, 12)
        regions.append(
            ClickRegion(
                x=8,
                y=y,
                w=TOOLBAR_WIDTH - 16,
                h=field_h,
                action="edit_res_max",
                param=0,
            )
        )
        y += field_h + 4

    show_lbl = "Show" if show_resources else "Show"
    show_bg = _ACTIVE_TOOL_BG if show_resources else (50, 50, 50)
    bar[y : y + field_h, 4 : TOOLBAR_WIDTH - 4] = show_bg
    show_txt = _render_text(show_lbl, font, _TEXT_COLOR)
    _blit_rgb(bar, show_txt, y + 2, 8)
    regions.append(
        ClickRegion(
            x=4,
            y=y,
            w=TOOLBAR_WIDTH - 8,
            h=field_h,
            action="toggle_show_res",
            param=0,
        )
    )
    y += field_h + 4

    y += 8
    label_txt = _render_text("Machines", font, _ACCENT_COLOR)
    _blit_rgb(bar, label_txt, y, 4)
    y += label_txt.shape[0] + 4

    icon_size = 18
    for idx, (machine_id, name) in enumerate(MACHINE_ITEMS):
        item_type = MACHINE_TO_ITEM_MAP[machine_id]
        icon = render_item_icon(item_type, icon_size, direction)
        bx = 4
        bar[y : y + icon_size, bx : bx + icon_size] = icon[:, :, :3]
        if machine_id == selected_machine:
            _draw_border(bar, bx, y, icon_size, icon_size, _SELECTED_BORDER)
        txt = _render_text(name, font, _TEXT_COLOR)
        _blit_rgb(bar, txt, y + (icon_size - txt.shape[0]) // 2, bx + icon_size + 4)
        regions.append(
            ClickRegion(
                x=4, y=y, w=TOOLBAR_WIDTH - 8, h=icon_size, action="machine", param=idx
            )
        )
        y += icon_size + 2

    # -- Entities section --------------------------------------------------
    y += 8
    label_txt = _render_text("Entities", font, _ACCENT_COLOR)
    _blit_rgb(bar, label_txt, y, 4)
    y += label_txt.shape[0] + 4

    for player_idx in range(MAX_EDITOR_PLAYERS):
        icon = create_player_start_icon(player_idx, icon_size)
        bx = 4
        _blit_rgba(bar, icon, y, bx)
        if selected_entity == ("player", player_idx):
            _draw_border(bar, bx, y, icon_size, icon_size, _SELECTED_BORDER)
        txt = _render_text(f"Player {player_idx}", font, _TEXT_COLOR)
        _blit_rgb(bar, txt, y + (icon_size - txt.shape[0]) // 2, bx + icon_size + 4)
        regions.append(
            ClickRegion(
                x=4,
                y=y,
                w=TOOLBAR_WIDTH - 8,
                h=icon_size,
                action="entity",
                param=player_idx,
            )
        )
        y += icon_size + 2

    biter_icon = create_biter_texture(icon_size)
    bx = 4
    _blit_rgba(bar, biter_icon, y, bx)
    if selected_entity == ("biter", 0):
        _draw_border(bar, bx, y, icon_size, icon_size, _SELECTED_BORDER)
    txt = _render_text("Biter", font, _TEXT_COLOR)
    _blit_rgb(bar, txt, y + (icon_size - txt.shape[0]) // 2, bx + icon_size + 4)
    regions.append(
        ClickRegion(
            x=4,
            y=y,
            w=TOOLBAR_WIDTH - 8,
            h=icon_size,
            action="entity",
            param=MAX_EDITOR_PLAYERS,
        )
    )
    y += icon_size + 2

    return bar, regions


def render_status_bar(
    tool: str,
    brush_name: str,
    cursor_tile: tuple[int, int] | None,
    level_name: str,
    dirty: bool,
    resource_info: str,
    width: int,
    layer: str = "terrain",
) -> np.ndarray:
    """Render the bottom status bar.

    Args:
        tool: Active tool name.
        brush_name: Name of the selected block or machine.
        cursor_tile: ``(x, y)`` tile under the cursor, or ``None``.
        level_name: Current level name.
        dirty: Whether unsaved changes exist.
        resource_info: Resource brush summary (e.g. ``"Res:100"``).
        width: Full window width in pixels.
        layer: Active editing layer (``"terrain"``, ``"machine"``,
            or ``"entity"``).

    Returns:
        RGB uint8 array of shape ``(STATUS_BAR_HEIGHT, width, 3)``.
    """
    bar = np.full((STATUS_BAR_HEIGHT, width, 3), _STATUS_BG, dtype=np.uint8)
    font = get_pixel_font(12)

    layer_tags = {
        "terrain": "[Terrain]",
        "machine": "[Machine]",
        "entity": "[Entity]",
        "inventory": "[Inventory]",
    }
    layer_colors = {
        "terrain": (100, 180, 100),
        "machine": (100, 140, 200),
        "entity": (200, 140, 100),
        "inventory": (180, 120, 200),
    }
    layer_tag = layer_tags.get(layer, "[Terrain]")
    parts = [layer_tag, f"{tool.capitalize()}: {brush_name}"]
    parts.append(resource_info)
    if cursor_tile is not None:
        parts.append(f"({cursor_tile[0]}, {cursor_tile[1]})")
    name_display = level_name + (" *" if dirty else "")
    parts.append(name_display)

    help_txt = _render_text("? Help  I: Inspect", font, (100, 100, 100))
    y_center = (STATUS_BAR_HEIGHT - help_txt.shape[0]) // 2
    x = 4
    _blit_rgb(bar, help_txt, y_center, x)
    x += help_txt.shape[1] + 8

    layer_color = layer_colors.get(layer, (100, 180, 100))
    tag_txt = _render_text(layer_tag, font, layer_color)
    _blit_rgb(bar, tag_txt, y_center, x)
    x += tag_txt.shape[1]

    rest_txt = _render_text(
        "  " + "  ".join(parts[1:]),
        font,
        _DIM_TEXT,
    )
    _blit_rgb(bar, rest_txt, y_center, x)
    return bar


def _draw_border(
    img: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    color: tuple[int, int, int],
) -> None:
    """Draw a 1px border around a rectangle on an RGB image.

    Args:
        img: RGB image (mutated in place).
        x: Left column.
        y: Top row.
        w: Width.
        h: Height.
        color: RGB border colour.
    """
    ih, iw = img.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(iw, x + w)
    y1 = min(ih, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    img[y0, x0:x1] = color
    img[y1 - 1, x0:x1] = color
    img[y0:y1, x0] = color
    img[y0:y1, x1 - 1] = color


def _blit_rgb(dst: np.ndarray, src: np.ndarray, y: int, x: int) -> None:
    """Copy an RGB patch onto *dst* with clipping.

    Args:
        dst: Destination RGB array (mutated in place).
        src: Source RGB array.
        y: Top row.
        x: Left column.
    """
    dh, dw = dst.shape[:2]
    sh, sw = src.shape[:2]
    sy0 = max(0, -y)
    sx0 = max(0, -x)
    dy0 = max(0, y)
    dx0 = max(0, x)
    dy1 = min(dh, y + sh)
    dx1 = min(dw, x + sw)
    if dy1 <= dy0 or dx1 <= dx0:
        return
    ch = dy1 - dy0
    cw = dx1 - dx0
    dst[dy0:dy1, dx0:dx1] = src[sy0 : sy0 + ch, sx0 : sx0 + cw]


def _blit_rgba(dst: np.ndarray, src: np.ndarray, y: int, x: int) -> None:
    """Composite an RGBA patch onto an RGB *dst* using alpha blending.

    Pixels with zero alpha leave the destination unchanged. Fully opaque
    pixels overwrite directly.

    Args:
        dst: Destination RGB array (mutated in place).
        src: Source RGBA array.
        y: Top row.
        x: Left column.
    """
    dh, dw = dst.shape[:2]
    sh, sw = src.shape[:2]
    sy0 = max(0, -y)
    sx0 = max(0, -x)
    dy0 = max(0, y)
    dx0 = max(0, x)
    dy1 = min(dh, y + sh)
    dx1 = min(dw, x + sw)
    if dy1 <= dy0 or dx1 <= dx0:
        return
    ch = dy1 - dy0
    cw = dx1 - dx0
    patch = src[sy0 : sy0 + ch, sx0 : sx0 + cw]
    alpha = patch[:, :, 3:4].astype(np.float32) / 255.0
    region = dst[dy0:dy1, dx0:dx1]
    blended = (
        patch[:, :, :3].astype(np.float32) * alpha
        + region.astype(np.float32) * (1.0 - alpha)
    ).astype(np.uint8)
    dst[dy0:dy1, dx0:dx1] = blended


# Human-readable display names derived from the ItemType enum so every
# item is covered automatically, even newly added ones.
_ITEM_DISPLAY_NAMES: dict[int, str] = {
    int(it): it.name.replace("_", " ").title() for it in ItemType
}

_PALETTE_ROW_H = 22
_PALETTE_ICON_SIZE = 18
_PALETTE_BG = (35, 35, 40)


def get_palette_items_for_player() -> list[tuple[int, str]]:
    """Return all non-empty items for the player inventory palette.

    Returns:
        List of ``(item_type_int, display_name)`` pairs, one per
        ``ItemType`` member excluding ``EMPTY``.
    """
    return [
        (int(it), _ITEM_DISPLAY_NAMES[int(it)])
        for it in ItemType
        if it != ItemType.EMPTY
    ]


def get_palette_items_for_machine_slot(
    machine_type: int,
    slot_idx: int,
) -> list[tuple[int, str]]:
    """Return items valid for a specific machine slot role.

    If the slot has role ``NONE`` the list is empty, meaning no items
    can be placed there.  All other roles (INPUT, OUTPUT, STORAGE)
    allow every non-empty item type so the editor can pre-fill any
    value.

    Args:
        machine_type: ``MachineType`` integer value.
        slot_idx: Zero-based slot index within the machine.

    Returns:
        List of ``(item_type_int, display_name)`` pairs.  Empty list
        when the slot role is ``NONE``.
    """
    role = int(MACHINE_SLOT_ROLES[machine_type, slot_idx])
    if role == int(SlotRole.NONE):
        return []
    return [
        (int(it), _ITEM_DISPLAY_NAMES[int(it)])
        for it in ItemType
        if it != ItemType.EMPTY
    ]


def render_item_palette(
    items: list[tuple[int, str]],
    height: int,
    scroll_offset: int = 0,
) -> tuple[np.ndarray, list[ClickRegion]]:
    """Render an item palette sidebar for the inventory view mode.

    Each item is drawn as a 22px-tall row with an 18x18 icon on the
    left and a text label beside it.  A small "Items" header in the
    accent colour appears above the list.  Rows that fall outside the
    visible area (after applying *scroll_offset*) are clipped.

    Args:
        items: ``(ItemType_int, display_name)`` pairs to show.
        height: Available pixel height for the palette (same as the
            toolbar content area).
        scroll_offset: Pixel offset for vertical scrolling when the
            item list is taller than *height*.

    Returns:
        ``(image, regions)`` where *image* is an RGB array of shape
        ``(height, TOOLBAR_WIDTH, 3)`` and *regions* lists one
        :class:`ClickRegion` per visible item row with
        ``action="inv_item"`` and ``param=item_type_int``.
    """
    content_h = max(height, _estimate_content_height(len(items)))
    bar = np.full((content_h, TOOLBAR_WIDTH, 3), _BG, dtype=np.uint8)
    regions: list[ClickRegion] = []
    font = get_pixel_font(12)

    y = 4
    header = _render_text("Items", font, _ACCENT_COLOR)
    _blit_rgb(bar, header, y, 4)
    y += header.shape[0] + 4

    for item_type, name in items:
        bar[y : y + _PALETTE_ROW_H, 0:TOOLBAR_WIDTH] = _PALETTE_BG

        icon = render_item_icon(item_type, _PALETTE_ICON_SIZE)
        _blit_rgba(bar, icon, y + (_PALETTE_ROW_H - _PALETTE_ICON_SIZE) // 2, 4)

        txt = _render_text(name, font, _TEXT_COLOR)
        _blit_rgb(
            bar,
            txt,
            y + (_PALETTE_ROW_H - txt.shape[0]) // 2,
            26,
        )

        regions.append(
            ClickRegion(
                x=0,
                y=y,
                w=TOOLBAR_WIDTH,
                h=_PALETTE_ROW_H,
                action="inv_item",
                param=item_type,
            )
        )
        y += _PALETTE_ROW_H

    # Apply scroll and crop to the requested height.
    scroll_offset = max(0, min(scroll_offset, content_h - height))
    visible = bar[scroll_offset : scroll_offset + height]

    # Shift region y-coordinates by the scroll offset so hit-testing
    # matches the visible output.
    shifted: list[ClickRegion] = [
        ClickRegion(
            x=r.x,
            y=r.y - scroll_offset,
            w=r.w,
            h=r.h,
            action=r.action,
            param=r.param,
        )
        for r in regions
        if r.y + r.h > scroll_offset and r.y < scroll_offset + height
    ]
    return visible, shifted


def _estimate_content_height(num_items: int) -> int:
    """Estimate the total content height for the item palette.

    Args:
        num_items: Number of items in the list.

    Returns:
        Pixel height needed to render the header plus all rows.
    """
    # header (~12px text + 4+4 padding) + rows
    return 20 + num_items * _PALETTE_ROW_H
