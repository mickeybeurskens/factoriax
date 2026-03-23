"""Layout constants and frame composition for the inspector."""

from __future__ import annotations

import numpy as np

from factoriax.analysis.trajectory import Trajectory
from factoriax.inspector.charts import (
    draw_cursor,
    render_action_strip,
    render_reward_chart,
)
from factoriax.inspector.panels import (
    render_info_panel,
    render_menu_bar,
    render_timeline,
)
from factoriax.inspector.state import InspectorState

# Layout geometry (pixels).
MENU_BAR_HEIGHT = 24
INFO_PANEL_WIDTH = 180
TIMELINE_HEIGHT = 32
CHART_HEIGHT = 100
ACTION_STRIP_HEIGHT = 24

# Minimum canvas area for the game world.
MIN_CANVAS_W = 300
MIN_CANVAS_H = 200


def compute_base_dimensions(
    canvas_w: int = MIN_CANVAS_W,
    canvas_h: int = MIN_CANVAS_H,
) -> tuple[int, int]:
    """Compute the base frame dimensions from canvas size.

    Args:
        canvas_w: Width of the game world canvas.
        canvas_h: Height of the game world canvas.

    Returns:
        ``(base_w, base_h)`` tuple.
    """
    bw = INFO_PANEL_WIDTH + canvas_w
    bh = (
        MENU_BAR_HEIGHT
        + canvas_h
        + TIMELINE_HEIGHT
        + CHART_HEIGHT
        + ACTION_STRIP_HEIGHT
    )
    return bw, bh


def rebuild_caches(
    traj: Trajectory,
    state: InspectorState,
    chart_width: int,
) -> None:
    """Pre-render cached chart images after episode/player change.

    Args:
        traj: Loaded trajectory.
        state: Inspector state (caches are set in place).
        chart_width: Pixel width for the chart images.
    """
    state.reward_chart_cache = render_reward_chart(
        traj, state.selected_episode, chart_width, CHART_HEIGHT
    )
    state.action_strip_cache = render_action_strip(
        traj,
        state.selected_episode,
        state.selected_player,
        chart_width,
        ACTION_STRIP_HEIGHT,
    )


def render_frame(
    traj: Trajectory | None,
    state: InspectorState,
    base_w: int,
    base_h: int,
    canvas_w: int,
    canvas_h: int,
) -> np.ndarray:
    """Compose the full inspector frame from all panels.

    Args:
        traj: Loaded trajectory, or None if nothing loaded.
        state: Current inspector state.
        base_w: Total frame width in pixels.
        base_h: Total frame height in pixels.
        canvas_w: Game world canvas width.
        canvas_h: Game world canvas height.

    Returns:
        RGB uint8 array of shape ``(base_h, base_w, 3)``.
    """
    frame = np.full((base_h, base_w, 3), (30, 30, 30), dtype=np.uint8)
    chart_width = base_w

    # Menu bar.
    menu, _ = render_menu_bar(base_w)
    menu_h = min(menu.shape[0], MENU_BAR_HEIGHT)
    frame[:menu_h, :] = menu[:menu_h]

    if traj is None:
        return frame

    total_steps = traj.episode_length

    # Info panel.
    info = render_info_panel(
        traj,
        state.selected_episode,
        state.current_step,
        state.selected_player,
        INFO_PANEL_WIDTH,
        canvas_h,
    )
    y_off = MENU_BAR_HEIGHT
    ih = min(info.shape[0], canvas_h)
    frame[y_off : y_off + ih, :INFO_PANEL_WIDTH] = info[:ih]

    # Game world placeholder (dark area for now — phase 2 adds rendering).
    world_x = INFO_PANEL_WIDTH
    world_y = MENU_BAR_HEIGHT
    frame[world_y : world_y + canvas_h, world_x : world_x + canvas_w] = (20, 20, 25)
    # Draw a simple position dot if positions are available.
    if traj.positions is not None:
        _draw_position_dot(frame, traj, state, world_x, world_y, canvas_w, canvas_h)

    # Timeline.
    tl_y = MENU_BAR_HEIGHT + canvas_h
    timeline, _ = render_timeline(
        state.current_step, total_steps, base_w, TIMELINE_HEIGHT
    )
    tl_h = min(timeline.shape[0], TIMELINE_HEIGHT)
    frame[tl_y : tl_y + tl_h, :base_w] = timeline[:tl_h, :base_w]

    # Reward chart with cursor.
    chart_y = tl_y + TIMELINE_HEIGHT
    if state.reward_chart_cache is not None:
        chart = draw_cursor(state.reward_chart_cache, state.current_step, total_steps)
        ch = min(chart.shape[0], CHART_HEIGHT)
        cw = min(chart.shape[1], chart_width)
        frame[chart_y : chart_y + ch, :cw] = chart[:ch, :cw]

    # Action strip with cursor.
    strip_y = chart_y + CHART_HEIGHT
    if state.action_strip_cache is not None:
        strip = draw_cursor(
            state.action_strip_cache,
            state.current_step,
            total_steps,
            color=(255, 255, 0),
        )
        sh = min(strip.shape[0], ACTION_STRIP_HEIGHT)
        sw = min(strip.shape[1], chart_width)
        frame[strip_y : strip_y + sh, :sw] = strip[:sh, :sw]

    return frame


def _draw_position_dot(
    frame: np.ndarray,
    traj: Trajectory,
    state: InspectorState,
    wx: int,
    wy: int,
    cw: int,
    ch: int,
) -> None:
    """Draw a colored dot for each player's position on the world area."""
    from factoriax.renderer import PLAYER_COLORS

    ep = state.selected_episode
    step = state.current_step
    num_p = traj.num_players

    # Determine map bounds from positions to scale dot placement.
    all_pos = traj.positions[ep]  # (T, P, 2) or (T, 2)
    if all_pos.ndim == 2:
        all_pos = all_pos[:, np.newaxis, :]
    max_x = max(int(all_pos[:, :, 0].max()) + 1, 1)
    max_y = max(int(all_pos[:, :, 1].max()) + 1, 1)

    for p in range(num_p):
        pos = all_pos[step, p]
        px = int(pos[0])
        py = int(pos[1])

        # Scale to canvas pixels.
        sx = wx + int(px * cw / max_x)
        sy = wy + int(py * ch / max_y)

        body_color, _ = PLAYER_COLORS[p % len(PLAYER_COLORS)]
        r = 3
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy <= r * r:
                    fy = sy + dy
                    fx = sx + dx
                    if 0 <= fy < frame.shape[0] and 0 <= fx < frame.shape[1]:
                        frame[fy, fx] = body_color
