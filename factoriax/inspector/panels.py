"""UI panels: info panel, timeline strip, and menu bar.

Each panel is a pure function returning an RGB or RGBA numpy array
plus optional click regions.
"""

from __future__ import annotations

import numpy as np

from factoriax.analysis.actions import DEFAULT_ACTION_LABELS
from factoriax.analysis.trajectory import Trajectory
from factoriax.constants import NUM_INVENTORY_SLOTS, Action, ItemType
from factoriax.renderer import render_item_icon
from factoriax.ui.fonts import get_pixel_font, render_text_rgba
from factoriax.ui.primitives import ClickRegion

_BG = (30, 30, 30)
_LABEL_COLOR = (180, 175, 140)
_VALUE_COLOR = (220, 215, 180)
_ACCENT_COLOR = (215, 195, 65)

_ACTION_LABELS = list(DEFAULT_ACTION_LABELS)
while len(_ACTION_LABELS) < len(Action):
    _ACTION_LABELS.append(f"ACT_{len(_ACTION_LABELS)}")


def render_info_panel(
    traj: Trajectory,
    episode: int,
    step: int,
    player: int,
    width: int,
    height: int,
) -> np.ndarray:
    """Render the player info panel as an RGB image.

    Shows position, current action, and inventory at the given step.

    Args:
        traj: Loaded trajectory.
        episode: Episode index.
        step: Current timestep.
        player: Player index to display.
        width: Panel width in pixels.
        height: Panel height in pixels.

    Returns:
        RGB uint8 array of shape ``(height, width, 3)``.
    """
    panel = np.full((height, width, 3), _BG, dtype=np.uint8)
    font = get_pixel_font(12)
    small_font = get_pixel_font(10)
    y = 4

    # Header.
    _draw_text(panel, f"Player {player}", font, _ACCENT_COLOR, 4, y)
    y += 16

    # Step counter.
    total_steps = traj.episode_length
    _draw_text(panel, f"Step {step}/{total_steps - 1}", small_font, _LABEL_COLOR, 4, y)
    y += 14

    # Current action.
    ep = traj.episode(episode)
    if ep.is_multi_player:
        action_val = int(ep.actions[0, step, player])
    else:
        action_val = int(ep.actions[0, step])
    action_name = (
        _ACTION_LABELS[action_val]
        if 0 <= action_val < len(_ACTION_LABELS)
        else f"?{action_val}"
    )
    _draw_text(panel, f"Action: {action_name}", small_font, _VALUE_COLOR, 4, y)
    y += 16

    # Position.
    if traj.positions is not None:
        pos = traj.positions[episode, step]
        # Positions may be (P, 2) or (2,) depending on multi-player.
        if pos.ndim > 1:
            pos = pos[player]
        px, py_val = int(pos[0]), int(pos[1])
        _draw_text(panel, f"Pos: ({px}, {py_val})", small_font, _VALUE_COLOR, 4, y)
    y += 16

    # Inventory.
    if traj.inventory_items is not None and traj.inventory_counts is not None:
        _draw_text(panel, "Inventory:", small_font, _LABEL_COLOR, 4, y)
        y += 14
        items = traj.inventory_items[episode, step]
        counts = traj.inventory_counts[episode, step]
        # May be (P, slots) or (slots,) depending on multi-player.
        if items.ndim > 1:
            items = items[player]
            counts = counts[player]

        icon_size = 14
        col = 0
        for slot in range(min(NUM_INVENTORY_SLOTS, len(items))):
            item = int(items[slot])
            count = int(counts[slot])
            if item == ItemType.EMPTY or count == 0:
                continue
            sx = 4 + col * (icon_size + 28)
            if sx + icon_size + 24 > width:
                y += icon_size + 4
                col = 0
                sx = 4
            icon = render_item_icon(item, icon_size)
            iy = min(y, height - icon_size)
            ix = min(sx, width - icon_size)
            if iy >= 0 and ix >= 0 and iy + icon_size <= height:
                panel[iy : iy + icon_size, ix : ix + icon_size] = icon[:, :, :3]
            cnt_txt = render_text_rgba(str(count), small_font, _VALUE_COLOR)
            tx = ix + icon_size + 2
            if tx + cnt_txt.shape[1] <= width and iy + cnt_txt.shape[0] <= height:
                _blit_rgb_from_rgba(panel, cnt_txt, iy, tx)
            col += 1

    return panel


def render_timeline(
    step: int,
    total_steps: int,
    width: int,
    height: int,
) -> tuple[np.ndarray, list[ClickRegion]]:
    """Render the timeline bar with transport controls.

    Args:
        step: Current step index.
        total_steps: Total number of steps.
        width: Bar width in pixels.
        height: Bar height in pixels.

    Returns:
        ``(image, regions)`` — RGB array and click regions for scrubbing
        and transport buttons.
    """
    bar = np.full((height, width, 3), (40, 40, 40), dtype=np.uint8)
    regions: list[ClickRegion] = []
    font = get_pixel_font(10)

    # Track area (where scrubbing happens).
    track_x = 60
    track_w = width - 120
    track_y = 4
    track_h = height - 8
    bar[track_y : track_y + track_h, track_x : track_x + track_w] = (55, 55, 55)

    # Playhead position.
    if total_steps > 1:
        head_x = track_x + int(step * (track_w - 2) / (total_steps - 1))
    else:
        head_x = track_x
    bar[track_y : track_y + track_h, head_x : head_x + 2] = (255, 255, 255)

    # Scrub region (the whole track).
    regions.append(ClickRegion(track_x, track_y, track_w, track_h, "scrub", 0))

    # Step counter text.
    txt = render_text_rgba(f"{step}/{total_steps - 1}", font, _VALUE_COLOR)
    _blit_rgb_from_rgba(bar, txt, (height - txt.shape[0]) // 2, 4)

    # Transport buttons on the right.
    btn_labels = ["|<", "<", ">", ">|"]
    btn_actions = ["first", "prev", "play_pause", "last"]
    btn_x = track_x + track_w + 4
    for i, (label, action) in enumerate(zip(btn_labels, btn_actions)):
        bx = btn_x + i * 15
        btn_txt = render_text_rgba(label, font, _ACCENT_COLOR)
        if bx + btn_txt.shape[1] <= width:
            _blit_rgb_from_rgba(bar, btn_txt, (height - btn_txt.shape[0]) // 2, bx)
            regions.append(ClickRegion(bx, 0, 14, height, action, 0))

    return bar, regions


def render_menu_bar(width: int) -> tuple[np.ndarray, list[ClickRegion]]:
    """Render a simple menu bar with episode info.

    Args:
        width: Bar width in pixels.

    Returns:
        ``(image, regions)`` — RGB array and click regions.
    """
    height = 24
    bar = np.full((height, width, 3), (35, 35, 35), dtype=np.uint8)
    bar[height - 1, :] = (60, 60, 60)
    font = get_pixel_font(11)
    txt = render_text_rgba("FactoriaX Inspector", font, _ACCENT_COLOR)
    _blit_rgb_from_rgba(bar, txt, (height - txt.shape[0]) // 2, 6)
    return bar, []


def _draw_text(
    img: np.ndarray,
    text: str,
    font: object,
    color: tuple[int, int, int],
    x: int,
    y: int,
) -> None:
    """Draw text onto an RGB image using the shared font renderer."""
    rgba = render_text_rgba(text, font, color)
    _blit_rgb_from_rgba(img, rgba, y, x)


def _blit_rgb_from_rgba(dst: np.ndarray, src_rgba: np.ndarray, y: int, x: int) -> None:
    """Blit an RGBA source onto an RGB destination, clipping to bounds."""
    dh, dw = dst.shape[:2]
    sh, sw = src_rgba.shape[:2]

    sy0 = max(0, -y)
    sx0 = max(0, -x)
    dy0, dx0 = max(0, y), max(0, x)
    dy1 = min(dh, y + sh)
    dx1 = min(dw, x + sw)

    if dy1 <= dy0 or dx1 <= dx0:
        return

    ch = dy1 - dy0
    cw = dx1 - dx0
    crop = src_rgba[sy0 : sy0 + ch, sx0 : sx0 + cw]
    region = dst[dy0:dy1, dx0:dx1]

    alpha = crop[:, :, 3:4].astype(np.float32) / 255.0
    region[:] = (
        crop[:, :, :3].astype(np.float32) * alpha
        + region.astype(np.float32) * (1.0 - alpha)
    ).astype(np.uint8)
