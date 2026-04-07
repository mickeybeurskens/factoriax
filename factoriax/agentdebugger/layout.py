"""Layout constants and frame composition for the debugger.

The debugger uses a 2x2 quadrant layout with a status bar at the
bottom:

.. code-block:: text

    +-------------------+-------------------+
    |                   |                   |
    |  Q1: Game View    |  Q2: Charts       |
    |                   |  (reward + cost)  |
    |                   |                   |
    +-------------------+-------------------+
    |                   |                   |
    |  Q3: (empty)      |  Q4: (empty)      |
    |                   |                   |
    +-------------------+-------------------+
    [status bar: mode | step N/M | player P]
"""

from __future__ import annotations

import numpy as np

from factoriax.agentdebugger.charts import (
    build_partial_trajectory,
    draw_cursor,
    render_cost_chart,
    render_reward_chart,
)
from factoriax.agentdebugger.state import DebuggerState
from factoriax.renderer import render_pixels
from factoriax.state import EnvState
from factoriax.ui.compositing import composite_rgba_over_rgb
from factoriax.ui.fonts import get_pixel_font, render_text_rgba

STATUS_BAR_HEIGHT = 28
DIVIDER_COLOR = (60, 60, 60)
BG_COLOR = (20, 20, 25)

# Minimum quadrant dimensions.
MIN_QUADRANT_W = 200
MIN_QUADRANT_H = 150


def compute_debugger_dimensions(
    quadrant_w: int = MIN_QUADRANT_W,
    quadrant_h: int = MIN_QUADRANT_H,
) -> tuple[int, int]:
    """Compute the base frame dimensions from quadrant size.

    Args:
        quadrant_w: Width of a single quadrant.
        quadrant_h: Height of a single quadrant.

    Returns:
        ``(base_w, base_h)`` tuple.
    """
    return quadrant_w * 2, quadrant_h * 2 + STATUS_BAR_HEIGHT


def render_debugger_frame(
    dbg: DebuggerState,
    states: list[EnvState],
    rewards: list[float],
    actions: list[int],
    costs: list[np.ndarray],
    constraint_names: list[str],
    base_w: int,
    base_h: int,
    quadrant_w: int,
    quadrant_h: int,
    *,
    has_reward: bool = False,
    has_cost: bool = False,
    player_idx: int = 0,
) -> np.ndarray:
    """Compose the full debugger frame from all quadrants.

    Args:
        dbg: Current debugger state.
        states: History of environment states (index 0 is initial).
        rewards: Per-step reward values.
        actions: Per-step action integers.
        costs: Per-step cost vectors.
        constraint_names: Labels for each constraint dimension.
        base_w: Total frame width.
        base_h: Total frame height.
        quadrant_w: Width of each quadrant.
        quadrant_h: Height of each quadrant.
        has_reward: Whether a reward function is configured.
        has_cost: Whether a constraint function is configured.
        player_idx: Player index being controlled.

    Returns:
        RGB uint8 array of shape ``(base_h, base_w, 3)``.
    """
    frame = np.full((base_h, base_w, 3), BG_COLOR, dtype=np.uint8)

    current_state = states[dbg.current_step]
    total_steps = len(states)

    # ------------------------------------------------------------------
    # Q1: Game view (top-left)
    # ------------------------------------------------------------------
    game_img = render_pixels(
        current_state, block_pixel_size=_tile_px(current_state),
        frame_tick=dbg.frame_tick,
    )
    _blit_game_frame(frame, game_img, 0, 0, quadrant_w, quadrant_h)

    # ------------------------------------------------------------------
    # Q2: Charts (top-right)
    # ------------------------------------------------------------------
    chart_x = quadrant_w
    if has_reward or has_cost:
        # Split Q2 vertically: reward top, cost bottom.
        if has_reward and has_cost:
            reward_h = quadrant_h // 2
            cost_h = quadrant_h - reward_h
        elif has_reward:
            reward_h = quadrant_h
            cost_h = 0
        else:
            reward_h = 0
            cost_h = quadrant_h

        if has_reward and reward_h > 0:
            if dbg.reward_chart_cache is None and rewards:
                traj = build_partial_trajectory(rewards, actions)
                dbg.reward_chart_cache = render_reward_chart(
                    traj, 0, quadrant_w, reward_h,
                )
            if dbg.reward_chart_cache is not None:
                chart = draw_cursor(
                    dbg.reward_chart_cache,
                    dbg.current_step,
                    total_steps,
                )
                _blit_exact(frame, chart, chart_x, 0)

        if has_cost and cost_h > 0:
            if dbg.cost_chart_cache is None and costs:
                dbg.cost_chart_cache = render_cost_chart(
                    costs, constraint_names, quadrant_w, cost_h,
                )
            if dbg.cost_chart_cache is not None:
                chart = draw_cursor(
                    dbg.cost_chart_cache,
                    dbg.current_step,
                    total_steps,
                )
                _blit_exact(frame, chart, chart_x, reward_h)
    else:
        _draw_quadrant_label(
            frame, chart_x, 0, quadrant_w, quadrant_h, "Q2",
        )

    # ------------------------------------------------------------------
    # Q3: empty (bottom-left)
    # ------------------------------------------------------------------
    _draw_quadrant_label(
        frame, 0, quadrant_h, quadrant_w, quadrant_h, "Q3",
    )

    # ------------------------------------------------------------------
    # Q4: empty (bottom-right)
    # ------------------------------------------------------------------
    _draw_quadrant_label(
        frame, quadrant_w, quadrant_h, quadrant_w, quadrant_h, "Q4",
    )

    # ------------------------------------------------------------------
    # Dividers
    # ------------------------------------------------------------------
    # Vertical center.
    frame[0 : quadrant_h * 2, quadrant_w] = DIVIDER_COLOR
    # Horizontal center.
    frame[quadrant_h, 0 : quadrant_w * 2] = DIVIDER_COLOR

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------
    status_y = quadrant_h * 2
    _render_status_bar(
        frame, status_y, base_w,
        mode=dbg.mode,
        step=dbg.current_step,
        total=total_steps,
        player=player_idx,
        done=dbg.done,
        awaiting="human" if dbg.mode == "human" else "",
    )

    return frame


def _tile_px(state: EnvState) -> int:
    """Choose a tile pixel size that keeps the map reasonably sized."""
    map_h, map_w = state.map.shape
    target = 300
    return max(4, min(target // max(map_w, 1), target // max(map_h, 1)))


def _blit_game_frame(
    frame: np.ndarray,
    game_img: np.ndarray,
    wx: int,
    wy: int,
    cw: int,
    ch: int,
) -> None:
    """Scale and center a game image into a canvas region."""
    gh, gw = game_img.shape[:2]
    scale = min(cw / max(gw, 1), ch / max(gh, 1))
    new_w = max(1, int(gw * scale))
    new_h = max(1, int(gh * scale))

    ys = np.linspace(0, gh - 1, new_h).astype(int)
    xs = np.linspace(0, gw - 1, new_w).astype(int)
    scaled = game_img[np.ix_(ys, xs)]

    ox = wx + (cw - new_w) // 2
    oy = wy + (ch - new_h) // 2
    sy = max(0, -oy)
    sx = max(0, -ox)
    dy = max(0, oy)
    dx = max(0, ox)
    bh = min(new_h - sy, frame.shape[0] - dy)
    bw = min(new_w - sx, frame.shape[1] - dx)
    if bh > 0 and bw > 0:
        frame[dy : dy + bh, dx : dx + bw] = scaled[
            sy : sy + bh, sx : sx + bw
        ]


def _blit_exact(
    frame: np.ndarray,
    src: np.ndarray,
    x: int,
    y: int,
) -> None:
    """Copy *src* into *frame* at (x, y), clipping to bounds."""
    sh, sw = src.shape[:2]
    fh, fw = frame.shape[:2]
    sy = max(0, -y)
    sx = max(0, -x)
    dy = max(0, y)
    dx = max(0, x)
    bh = min(sh - sy, fh - dy)
    bw = min(sw - sx, fw - dx)
    if bh > 0 and bw > 0:
        frame[dy : dy + bh, dx : dx + bw] = src[
            sy : sy + bh, sx : sx + bw, :3
        ]


def _render_status_bar(
    frame: np.ndarray,
    y: int,
    width: int,
    *,
    mode: str,
    step: int,
    total: int,
    player: int,
    done: bool,
    awaiting: str,
) -> None:
    """Draw the status bar at the bottom of the frame."""
    bar_h = STATUS_BAR_HEIGHT
    frame[y : y + bar_h, :width] = (30, 30, 35)
    # Top edge highlight.
    frame[y, :width] = DIVIDER_COLOR

    parts = [
        f"MODE: {mode.upper()}",
        f"STEP: {step}/{total - 1}",
        f"PLAYER: {player}",
    ]
    if done:
        parts.append("DONE")
    if awaiting:
        parts.append("AWAITING INPUT")

    text = "  |  ".join(parts)
    font = get_pixel_font(12)
    text_rgba = render_text_rgba(text, font, (180, 180, 170))
    th, tw = text_rgba.shape[:2]
    tx = 6
    ty = y + (bar_h - th) // 2
    if ty >= 0 and ty + th <= frame.shape[0] and tx + tw <= frame.shape[1]:
        composite_rgba_over_rgb(
            frame[ty : ty + th, tx : tx + tw],
            text_rgba,
        )


def _draw_quadrant_label(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    label: str,
) -> None:
    """Draw a dim label in the center of an empty quadrant."""
    color = (50, 50, 55)
    glyphs: dict[str, list[int]] = {
        "Q": [0b111, 0b101, 0b101, 0b111, 0b010],
        "2": [0b111, 0b001, 0b111, 0b100, 0b111],
        "3": [0b111, 0b001, 0b111, 0b001, 0b111],
        "4": [0b101, 0b101, 0b111, 0b001, 0b001],
    }
    cx = x + w // 2 - len(label) * 2
    cy = y + h // 2 - 2
    for ci, ch_char in enumerate(label):
        rows = glyphs.get(ch_char, [0] * 5)
        for row_idx, row_bits in enumerate(rows):
            for col_idx in range(3):
                if row_bits & (1 << (2 - col_idx)):
                    px = cx + ci * 4 + col_idx
                    py = cy + row_idx
                    if (
                        0 <= py < frame.shape[0]
                        and 0 <= px < frame.shape[1]
                    ):
                        frame[py, px] = color
