"""Right-side inventory panel for the editor's inventory view mode.

Renders a grid of inventory slots for the selected player or machine,
with item icons, stack counts, role badges (for machines), and click
regions for slot interaction.
"""

from __future__ import annotations

import numpy as np
import pygame

from factoriax.constants import (
    MACHINE_SLOT_ROLES,
    MACHINE_TYPE_NAMES,
    MAX_MACHINE_INVENTORY_SLOTS,
    NUM_INVENTORY_SLOTS,
    SLOT_ROLE_COLORS,
    SLOT_ROLE_LABELS,
    ItemType,
)
from factoriax.editor.state import (
    EditorState,
    InvTarget,
    get_inventory_slots,
    get_num_slots,
)
from factoriax.ui.fonts import get_pixel_font, render_text_rgba
from factoriax.ui.icons import PLAYER_COLORS, render_item_icon
from factoriax.ui.primitives import ClickRegion

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

_PANEL_BG: tuple[int, int, int, int] = (30, 30, 35, 255)
_SLOT_BG: tuple[int, int, int, int] = (45, 45, 45, 255)
_FOCUSED_SLOT_BG: tuple[int, int, int, int] = (70, 70, 70, 255)
_BORDER: tuple[int, int, int, int] = (80, 80, 80, 255)
_TEXT_COLOR: tuple[int, int, int] = (200, 200, 200)
_EMPTY_TEXT_COLOR: tuple[int, int, int] = (100, 100, 100)
_PAD: int = 8
_SLOT_GAP: int = 4
_HEADER_H: int = 30

# Human-readable names derived from the ItemType enum. Underscores become
# spaces and each word is title-cased, giving e.g. "Conveyor Belt".
_ITEM_NAMES: dict[int, str] = {
    int(it): it.name.replace("_", " ").title() for it in ItemType
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _blit_alpha(dst: np.ndarray, src: np.ndarray, y: int, x: int) -> None:
    """Alpha-composite *src* onto *dst* with boundary clipping.

    Args:
        dst: Destination RGBA array (mutated in place).
        src: Source RGBA array.
        y: Top row in destination.
        x: Left column in destination.
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
    crop = src[sy0 : sy0 + ch, sx0 : sx0 + cw]
    region = dst[dy0:dy1, dx0:dx1]
    alpha = crop[:, :, 3:4].astype(np.float32) / 255.0
    region[:, :, :3] = (
        crop[:, :, :3].astype(np.float32) * alpha
        + region[:, :, :3].astype(np.float32) * (1.0 - alpha)
    ).astype(np.uint8)
    region[:, :, 3] = np.maximum(region[:, :, 3], crop[:, :, 3])


def _draw_player_header(
    panel: np.ndarray,
    player_idx: int,
    font: pygame.font.Font,
) -> None:
    """Draw a player inventory header with colored dot and title text.

    Args:
        panel: Destination RGBA panel (mutated in place).
        player_idx: Player index (determines color).
        font: Font used for the header text.
    """
    color_idx = player_idx % len(PLAYER_COLORS)
    body_rgb = PLAYER_COLORS[color_idx][0]

    # Draw color dot (circle via distance mask).
    dot_r = 5
    dot_cx = _PAD + dot_r
    dot_cy = _HEADER_H // 2
    ys, xs = np.ogrid[
        dot_cy - dot_r : dot_cy + dot_r + 1,
        dot_cx - dot_r : dot_cx + dot_r + 1,
    ]
    dist = (xs - dot_cx) ** 2 + (ys - dot_cy) ** 2
    mask = dist <= dot_r * dot_r
    panel[
        dot_cy - dot_r : dot_cy + dot_r + 1,
        dot_cx - dot_r : dot_cx + dot_r + 1,
    ][mask] = [*body_rgb, 255]

    label = render_text_rgba(f"Player {player_idx} Inventory", font, _TEXT_COLOR)
    text_x = dot_cx + dot_r + 6
    text_y = (_HEADER_H - label.shape[0]) // 2
    _blit_alpha(panel, label, text_y, text_x)


def _draw_machine_header(
    panel: np.ndarray,
    state: EditorState,
    tx: int,
    ty: int,
    font: pygame.font.Font,
) -> None:
    """Draw a machine inventory header with type name and coordinates.

    Args:
        panel: Destination RGBA panel (mutated in place).
        state: Editor state (for machine type lookup).
        tx: Tile x coordinate.
        ty: Tile y coordinate.
        font: Font used for the header text.
    """
    mt = int(state.machine_types[ty, tx])
    name = MACHINE_TYPE_NAMES.get(mt, "Unknown")
    label = render_text_rgba(f"{name} @ ({tx}, {ty})", font, _TEXT_COLOR)
    text_x = _PAD
    text_y = (_HEADER_H - label.shape[0]) // 2
    _blit_alpha(panel, label, text_y, text_x)


def _draw_slot(
    panel: np.ndarray,
    sx: int,
    sy: int,
    sw: int,
    sh: int,
    item_type: int,
    count: int,
    focused: bool,
    role: int | None,
    small_font: pygame.font.Font,
) -> None:
    """Render a single inventory slot cell onto the panel.

    Draws the slot background, optional role badge, item icon, count
    label, and item name (or "(empty)" placeholder).

    Args:
        panel: Destination RGBA panel (mutated in place).
        sx: Left edge of the slot cell.
        sy: Top edge of the slot cell.
        sw: Width of the slot cell.
        sh: Height of the slot cell.
        item_type: ItemType integer for this slot.
        count: Stack count.
        focused: Whether this slot is currently focused.
        role: SlotRole integer for machine slots, or None for players.
        small_font: Font for count and name text.
    """
    bg = _FOCUSED_SLOT_BG if focused else _SLOT_BG
    panel[sy : sy + sh, sx : sx + sw] = bg

    # 1-pixel border
    panel[sy, sx : sx + sw] = _BORDER
    panel[sy + sh - 1, sx : sx + sw] = _BORDER
    panel[sy : sy + sh, sx] = _BORDER
    panel[sy : sy + sh, sx + sw - 1] = _BORDER

    content_x = sx + 2
    content_w = sw - 4

    # Role badge for machine slots.
    if role is not None:
        role_label = SLOT_ROLE_LABELS.get(role, "")
        if role_label:
            badge_rgb = SLOT_ROLE_COLORS.get(role, (40, 40, 40))
            badge_w = 36
            badge_h = 14
            bx = content_x
            by = sy + 2
            panel[by : by + badge_h, bx : bx + badge_w] = [*badge_rgb, 255]
            badge_text = render_text_rgba(role_label, small_font, (230, 230, 230))
            btx = bx + (badge_w - badge_text.shape[1]) // 2
            bty = by + (badge_h - badge_text.shape[0]) // 2
            _blit_alpha(panel, badge_text, bty, btx)
            content_x = bx + badge_w + 4
            content_w = sx + sw - 4 - content_x

    # Icon and text area.
    icon_size = min(24, sh - 8, content_w - 4)
    if icon_size < 4:
        return

    is_empty = item_type == int(ItemType.EMPTY)

    if not is_empty:
        icon = render_item_icon(item_type, icon_size)
        icon_x = content_x + (content_w - icon_size) // 2
        icon_y = sy + 4
        _blit_alpha(panel, icon, icon_y, icon_x)

        # Count text to the right of the icon (vertically centered).
        if count > 0:
            count_txt = render_text_rgba(f"x{count}", small_font, _TEXT_COLOR)
            ctx = icon_x + icon_size + 2
            cty = icon_y + (icon_size - count_txt.shape[0]) // 2
            _blit_alpha(panel, count_txt, cty, ctx)

        # Item name below the icon.
        name = _ITEM_NAMES.get(item_type, "???")
        name_txt = render_text_rgba(name, small_font, _TEXT_COLOR)
        nx = content_x + (content_w - name_txt.shape[1]) // 2
        ny = sy + 4 + icon_size + 2
        _blit_alpha(panel, name_txt, ny, nx)
    else:
        empty_txt = render_text_rgba("(empty)", small_font, _EMPTY_TEXT_COLOR)
        ex = content_x + (content_w - empty_txt.shape[1]) // 2
        ey = sy + (sh - empty_txt.shape[0]) // 2
        _blit_alpha(panel, empty_txt, ey, ex)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_inventory_panel(
    state: EditorState,
    target: InvTarget,
    focused_slot: int,
    panel_w: int,
    panel_h: int,
) -> tuple[np.ndarray, list[ClickRegion]]:
    """Render the right-side inventory panel for the active target.

    Draws a header identifying the target (player or machine), followed
    by a grid of inventory slots. Each slot shows its item icon, stack
    count, name, and (for machines) a role badge.

    Args:
        state: Current editor state.
        target: ``("player", player_idx, 0)`` or
            ``("machine", tile_x, tile_y)``.
        focused_slot: Slot index that should be highlighted, or -1.
        panel_w: Desired panel width in pixels.
        panel_h: Desired panel height in pixels.

    Returns:
        Tuple of (RGBA image of shape ``(panel_h, panel_w, 4)``,
        list of :class:`ClickRegion` for slot hit-testing).
    """
    panel = np.zeros((panel_h, panel_w, 4), dtype=np.uint8)
    panel[:] = _PANEL_BG
    regions: list[ClickRegion] = []

    header_font = get_pixel_font(14)
    small_font = get_pixel_font(10)

    is_player = target[0] == "player"

    # -- Header --
    if is_player:
        _draw_player_header(panel, target[1], header_font)
    else:
        _draw_machine_header(panel, state, target[1], target[2], header_font)

    # -- Slot grid --
    slots = get_inventory_slots(state, target)
    num_slots = get_num_slots(state, target)

    if is_player:
        cols = 2
        rows = (NUM_INVENTORY_SLOTS + cols - 1) // cols
    else:
        cols = 1
        rows = min(num_slots, MAX_MACHINE_INVENTORY_SLOTS)

    usable_w = panel_w - 2 * _PAD - (cols - 1) * _SLOT_GAP
    usable_h = panel_h - _HEADER_H - 2 * _PAD - (rows - 1) * _SLOT_GAP
    slot_w = usable_w // cols if cols > 0 else 0
    slot_h = usable_h // rows if rows > 0 else 0

    # Clamp slot height so cells don't get absurdly tall on large panels.
    slot_h = min(slot_h, 52)

    # Machine type for role lookup (only used for machine targets).
    mt = 0
    if not is_player:
        mt = int(state.machine_types[target[2], target[1]])

    for idx in range(num_slots):
        col = idx % cols
        row = idx // cols
        sx = _PAD + col * (slot_w + _SLOT_GAP)
        sy = _HEADER_H + _PAD + row * (slot_h + _SLOT_GAP)

        item_type, count = slots[idx]
        role: int | None = None
        if not is_player:
            role = int(MACHINE_SLOT_ROLES[mt, idx])

        _draw_slot(
            panel,
            sx,
            sy,
            slot_w,
            slot_h,
            item_type,
            count,
            focused=(idx == focused_slot),
            role=role,
            small_font=small_font,
        )

        regions.append(
            ClickRegion(
                x=sx,
                y=sy,
                w=slot_w,
                h=slot_h,
                action="inv_slot",
                param=idx,
            )
        )

    return panel, regions
