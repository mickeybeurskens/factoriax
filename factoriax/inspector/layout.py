"""Layout constants and frame composition for the inspector."""

from __future__ import annotations

import numpy as np

from factoriax.analysis.trajectory import Trajectory
from factoriax.inspector.charts import (
    draw_cursor,
    render_action_legend,
    render_action_sankey,
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
LEGEND_HEIGHT = 20
SANKEY_HEIGHT = 120

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
        + LEGEND_HEIGHT
        + SANKEY_HEIGHT
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
    state.action_legend_cache = render_action_legend(
        traj,
        state.selected_episode,
        state.selected_player,
        chart_width,
        LEGEND_HEIGHT,
    )
    state.sankey_cache = render_action_sankey(
        traj,
        state.selected_episode,
        state.selected_player,
        chart_width,
        SANKEY_HEIGHT,
    )


def render_frame(
    traj: Trajectory | None,
    state: InspectorState,
    base_w: int,
    base_h: int,
    canvas_w: int,
    canvas_h: int,
    frames: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Compose the full inspector frame from all panels.

    Args:
        traj: Loaded trajectory, or None if nothing loaded.
        state: Current inspector state.
        base_w: Total frame width in pixels.
        base_h: Total frame height in pixels.
        canvas_w: Game world canvas width.
        canvas_h: Game world canvas height.
        frames: Pre-rendered game world frames from replay, or None.

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

    # Game world area.
    world_x = INFO_PANEL_WIDTH
    world_y = MENU_BAR_HEIGHT
    frame[world_y : world_y + canvas_h, world_x : world_x + canvas_w] = (20, 20, 25)
    if frames is not None and state.current_step < len(frames):
        game_img = frames[state.current_step]
        # Apply fog-of-war for LOCAL observations.
        if state.show_obs_overlay:
            game_img = _maybe_apply_fog(game_img, traj, state)
        _blit_game_frame(frame, game_img, world_x, world_y, canvas_w, canvas_h)
    elif traj.positions is not None:
        _draw_world(frame, traj, state, world_x, world_y, canvas_w, canvas_h)

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

    # Action legend.
    legend_y = strip_y + ACTION_STRIP_HEIGHT
    if state.action_legend_cache is not None:
        lg = state.action_legend_cache
        lh = min(lg.shape[0], LEGEND_HEIGHT)
        lw = min(lg.shape[1], chart_width)
        frame[legend_y : legend_y + lh, :lw] = lg[:lh, :lw]

    # Action transition heatmap (Sankey).
    sankey_y = legend_y + LEGEND_HEIGHT
    if state.sankey_cache is not None:
        sk = state.sankey_cache
        skh = min(sk.shape[0], SANKEY_HEIGHT)
        skw = min(sk.shape[1], chart_width)
        frame[sankey_y : sankey_y + skh, :skw] = sk[:skh, :skw]

    return frame


def _maybe_apply_fog(
    game_img: np.ndarray,
    traj: Trajectory,
    state: InspectorState,
) -> np.ndarray:
    """Apply fog-of-war if the trajectory uses LOCAL observations.

    Returns the image unmodified for GLOBAL/UNDETERMINED/PLAYER types.
    """
    from factoriax.inspector.obs_types import ObservationType, apply_fog_of_war

    scheme = traj.observation_scheme
    if scheme is None:
        return game_img
    obs_type = scheme.get("type", ObservationType.UNDETERMINED)
    if obs_type != ObservationType.LOCAL:
        return game_img

    radius = scheme.get("radius", 7)

    # Get player position at current step.
    if traj.positions is None:
        return game_img
    pos = traj.positions[state.selected_episode, state.current_step]
    if pos.ndim > 1:
        pos = pos[state.selected_player]
    px, py = int(pos[0]), int(pos[1])

    # Infer tile size and map dimensions from the rendered frame and map data.
    if traj.block_map is not None:
        map_data = traj.block_map[state.selected_episode, state.current_step]
        map_h, map_w = map_data.shape
    else:
        map_h = max(1, int(traj.positions[state.selected_episode, :, ..., 1].max()) + 1)
        map_w = max(1, int(traj.positions[state.selected_episode, :, ..., 0].max()) + 1)

    fh, fw = game_img.shape[:2]
    tile_size = max(1, min(fw // map_w, fh // map_h))

    return apply_fog_of_war(game_img, px, py, radius, tile_size, map_w, map_h)


def _blit_game_frame(
    frame: np.ndarray,
    game_img: np.ndarray,
    wx: int,
    wy: int,
    cw: int,
    ch: int,
) -> None:
    """Scale and center a pre-rendered game image into the world canvas."""
    gh, gw = game_img.shape[:2]
    # Fit to canvas, preserving aspect ratio.
    scale = min(cw / gw, ch / gh)
    new_w = max(1, int(gw * scale))
    new_h = max(1, int(gh * scale))

    # Nearest-neighbor resize via index mapping.
    ys = np.linspace(0, gh - 1, new_h).astype(int)
    xs = np.linspace(0, gw - 1, new_w).astype(int)
    scaled = game_img[np.ix_(ys, xs)]

    # Center in canvas.
    ox = wx + (cw - new_w) // 2
    oy = wy + (ch - new_h) // 2
    # Clip to frame bounds.
    sy = max(0, -oy)
    sx = max(0, -ox)
    dy = max(0, oy)
    dx = max(0, ox)
    bh = min(new_h - sy, frame.shape[0] - dy)
    bw = min(new_w - sx, frame.shape[1] - dx)
    if bh > 0 and bw > 0:
        frame[dy : dy + bh, dx : dx + bw] = scaled[sy : sy + bh, sx : sx + bw]


def _draw_world(
    frame: np.ndarray,
    traj: Trajectory,
    state: InspectorState,
    wx: int,
    wy: int,
    cw: int,
    ch: int,
) -> None:
    """Draw a grid-based world view with player positions."""
    from factoriax.renderer import PLAYER_COLORS

    ep = state.selected_episode
    step = state.current_step
    num_p = traj.num_players

    all_pos = traj.positions[ep]
    if all_pos.ndim == 2:
        all_pos = all_pos[:, np.newaxis, :]
    grid_w = max(int(all_pos[:, :, 0].max()) + 1, 1)
    grid_h = max(int(all_pos[:, :, 1].max()) + 1, 1)

    # Compute tile size so the grid fits the canvas.
    tile_w = max(1, cw // grid_w)
    tile_h = max(1, ch // grid_h)
    tile = min(tile_w, tile_h)

    # Center the grid in the canvas.
    ox = wx + (cw - grid_w * tile) // 2
    oy = wy + (ch - grid_h * tile) // 2

    # Draw tile backgrounds (alternating dark shades).
    for gy in range(grid_h):
        for gx in range(grid_w):
            x0 = ox + gx * tile
            y0 = oy + gy * tile
            shade = 32 if (gx + gy) % 2 == 0 else 38
            x1 = min(x0 + tile, frame.shape[1])
            y1 = min(y0 + tile, frame.shape[0])
            if x0 < frame.shape[1] and y0 < frame.shape[0]:
                frame[y0:y1, x0:x1] = (shade, shade, shade + 5)

    # Draw grid lines.
    grid_color = (55, 55, 60)
    for gx in range(grid_w + 1):
        lx = ox + gx * tile
        if 0 <= lx < frame.shape[1]:
            y0 = max(0, oy)
            y1 = min(frame.shape[0], oy + grid_h * tile)
            frame[y0:y1, lx] = grid_color
    for gy in range(grid_h + 1):
        ly = oy + gy * tile
        if 0 <= ly < frame.shape[0]:
            x0 = max(0, ox)
            x1 = min(frame.shape[1], ox + grid_w * tile)
            frame[ly, x0:x1] = grid_color

    # Draw player dots.
    for p in range(num_p):
        pos = all_pos[step, p]
        px, py = int(pos[0]), int(pos[1])
        cx = ox + px * tile + tile // 2
        cy = oy + py * tile + tile // 2

        body_color, _ = PLAYER_COLORS[p % len(PLAYER_COLORS)]
        r = max(2, tile // 3)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy <= r * r:
                    fy, fx = cy + dy, cx + dx
                    if 0 <= fy < frame.shape[0] and 0 <= fx < frame.shape[1]:
                        frame[fy, fx] = body_color
