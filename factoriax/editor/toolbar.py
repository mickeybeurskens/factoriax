"""Sidebar toolbar, menu bar, and status bar for the level editor.

Renders all chrome around the canvas: tool palette, block selector,
resource brush panel, machine palette, top menu bar, and bottom status
bar.  Each render function returns an RGB or RGBA image and a list of
:class:`~factoriax.play.ui.ClickRegion` objects for hit-testing.
"""

from __future__ import annotations

import numpy as np
import pygame

from factoriax.constants import (
    BlockType,
    ItemType,
    MachineType,
)
from factoriax.play.ui import ClickRegion, get_pixel_font
from factoriax.renderer import get_textures, render_item_icon

TOOLBAR_WIDTH = 120
MENU_BAR_HEIGHT = 32
STATUS_BAR_HEIGHT = 24

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

BLOCK_ITEMS: list[tuple[int, str]] = [
    (int(BlockType.DIRT), "Dirt"),
    (int(BlockType.WATER), "Water"),
    (int(BlockType.IRON), "Iron"),
    (int(BlockType.COPPER), "Copper"),
    (int(BlockType.COAL), "Coal"),
]

MACHINE_ITEMS: list[tuple[int, str]] = [
    (int(MachineType.MINER), "Miner"),
    (int(MachineType.CHEST), "Chest"),
    (int(MachineType.CONVEYOR_BELT), "Belt"),
    (int(MachineType.ARM), "Arm"),
]

MACHINE_TO_ITEM_MAP: dict[int, int] = {
    int(MachineType.MINER): int(ItemType.MINER),
    int(MachineType.CHEST): int(ItemType.CHEST),
    int(MachineType.CONVEYOR_BELT): int(ItemType.CONVEYOR_BELT),
    int(MachineType.ARM): int(ItemType.ARM),
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
) -> tuple[np.ndarray, list[ClickRegion]]:
    """Render the sidebar toolbar with all palette sections.

    Args:
        selected_tool: Active tool name (``"paint"``, ``"fill"``, ``"erase"``).
        selected_block: Active ``BlockType`` value.
        selected_machine: Active ``MachineType`` value, or 0 for none.
        direction: Current machine placement direction.
        resource_brush: :class:`~factoriax.editor.state.ResourceBrush`.
        height: Available height for the toolbar in pixels.

    Returns:
        ``(image, regions)`` where *image* is RGB shape
        ``(height, TOOLBAR_WIDTH, 3)``.
    """
    from factoriax.editor.state import ResourceBrush

    brush: ResourceBrush = resource_brush  # type: ignore[assignment]
    bar = np.full((height, TOOLBAR_WIDTH, 3), _BG, dtype=np.uint8)
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
            x=4, y=y, w=TOOLBAR_WIDTH - 8, h=16, action="toggle_res_mode", param=0
        )
    )
    y += 18

    if brush.mode == "exact":
        val_str = str(brush.exact_value)
    else:
        val_str = f"{brush.range_min}-{brush.range_max}"
    val_txt = _render_text(val_str, font, _TEXT_COLOR)
    _blit_rgb(bar, val_txt, y, 8)
    y += 18

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

    return bar, regions


def render_status_bar(
    tool: str,
    brush_name: str,
    cursor_tile: tuple[int, int] | None,
    level_name: str,
    dirty: bool,
    resource_info: str,
    width: int,
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

    Returns:
        RGB uint8 array of shape ``(STATUS_BAR_HEIGHT, width, 3)``.
    """
    bar = np.full((STATUS_BAR_HEIGHT, width, 3), _STATUS_BG, dtype=np.uint8)
    font = get_pixel_font(12)

    parts = [f"{tool.capitalize()}: {brush_name}"]
    parts.append(resource_info)
    if cursor_tile is not None:
        parts.append(f"({cursor_tile[0]}, {cursor_tile[1]})")
    name_display = level_name + (" *" if dirty else "")
    parts.append(name_display)

    text = "  ".join(parts)
    txt = _render_text(text, font, _DIM_TEXT)
    _blit_rgb(bar, txt, (STATUS_BAR_HEIGHT - txt.shape[0]) // 2, 8)
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
