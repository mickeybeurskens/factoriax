"""Layout constants and frame composition for the debugger.

The debugger uses a 2x2 quadrant layout with a status bar at the
bottom. Q1 shows the game view and Q2 the player's inventory in
both modes. Q3/Q4 switch on mode: live mode leaves Q3 as a cost
chart placeholder and Q4 holds the reward chart; replay mode shows
the action strip + legend in Q3 and the reward chart in Q4.

.. code-block:: text

    +-------------------+-------------------+
    |                   |                   |
    |  Q1: Game View    |  Q2: Inventory    |
    |                   |                   |
    |                   |                   |
    +-------------------+-------------------+
    |                   |                   |
    |  Q3: (cost /      |  Q4: Reward       |
    |       action      |       chart       |
    |       strip)      |                   |
    +-------------------+-------------------+
    [status bar: mode | step N/M | player P]
"""

from __future__ import annotations

import numpy as np

from factoriax.agentdebugger.charts import (
    build_partial_trajectory,
    draw_cursor,
    render_action_legend,
    render_action_strip,
    render_cost_chart,
    render_inventory_panel,
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
    """Compose the full debugger frame from all quadrants (live mode).

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
        current_state,
        block_pixel_size=_tile_px(current_state),
        frame_tick=dbg.frame_tick,
    )
    _blit_game_frame(frame, game_img, 0, 0, quadrant_w, quadrant_h)

    # ------------------------------------------------------------------
    # Q2: Inventory (top-right) — re-rendered every frame since counts
    # change on almost every step.
    # ------------------------------------------------------------------
    inv_panel = render_inventory_panel(
        np.asarray(current_state.player_inventory[player_idx]),
        quadrant_w,
        quadrant_h,
    )
    _blit_exact(frame, inv_panel, quadrant_w, 0)

    # ------------------------------------------------------------------
    # Q3: cost chart (bottom-left) when configured, else empty.
    # ------------------------------------------------------------------
    if has_cost:
        if dbg.cost_chart_cache is None and costs:
            dbg.cost_chart_cache = render_cost_chart(
                costs,
                constraint_names,
                quadrant_w,
                quadrant_h,
            )
        if dbg.cost_chart_cache is not None:
            chart = draw_cursor(
                dbg.cost_chart_cache,
                dbg.current_step,
                total_steps,
            )
            _blit_exact(frame, chart, 0, quadrant_h)
    else:
        _draw_quadrant_label(
            frame,
            0,
            quadrant_h,
            quadrant_w,
            quadrant_h,
            "Q3",
        )

    # ------------------------------------------------------------------
    # Q4: reward chart (bottom-right) when configured, else empty.
    # ------------------------------------------------------------------
    if has_reward:
        if dbg.reward_chart_cache is None and rewards:
            traj = build_partial_trajectory(rewards, actions)
            dbg.reward_chart_cache = render_reward_chart(
                traj,
                0,
                quadrant_w,
                quadrant_h,
            )
        if dbg.reward_chart_cache is not None:
            chart = draw_cursor(
                dbg.reward_chart_cache,
                dbg.current_step,
                total_steps,
            )
            _blit_exact(frame, chart, quadrant_w, quadrant_h)
    else:
        _draw_quadrant_label(
            frame,
            quadrant_w,
            quadrant_h,
            quadrant_w,
            quadrant_h,
            "Q4",
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
        frame,
        status_y,
        base_w,
        mode=dbg.mode,
        step=dbg.current_step,
        total=total_steps,
        player=player_idx,
        done=dbg.done,
    )

    return frame


# ------------------------------------------------------------------
# Replay-mode frame composition
# ------------------------------------------------------------------


def render_replay_frame(
    dbg: DebuggerState,
    base_w: int,
    base_h: int,
    quadrant_w: int,
    quadrant_h: int,
) -> np.ndarray:
    """Compose the full debugger frame for replay mode.

    Q1 shows the game world (from pre-rendered frames or grid
    fallback), Q2 shows the player inventory, Q3 shows the action
    strip + legend, and Q4 shows the reward chart.

    Args:
        dbg: Current debugger state (must have ``replay_mode=True``).
        base_w: Total frame width.
        base_h: Total frame height.
        quadrant_w: Width of each quadrant.
        quadrant_h: Height of each quadrant.

    Returns:
        RGB uint8 array of shape ``(base_h, base_w, 3)``.
    """
    traj = dbg.trajectory
    frame = np.full((base_h, base_w, 3), BG_COLOR, dtype=np.uint8)
    total_steps = traj.episode_length if traj is not None else 1

    # ------------------------------------------------------------------
    # Q1: Game view (top-left)
    # ------------------------------------------------------------------
    if dbg.rendered_frames is not None and dbg.current_step < len(dbg.rendered_frames):
        game_img = dbg.rendered_frames[dbg.current_step]
        if dbg.show_obs_overlay and traj is not None:
            game_img = _maybe_apply_fog(game_img, traj, dbg)
        _blit_game_frame(frame, game_img, 0, 0, quadrant_w, quadrant_h)
    elif traj is not None and traj.positions is not None:
        _draw_world(
            frame,
            traj,
            dbg,
            0,
            0,
            quadrant_w,
            quadrant_h,
        )
    else:
        _draw_centered_text(
            frame,
            0,
            0,
            quadrant_w,
            quadrant_h,
            "No state data -- provide --level",
        )

    # ------------------------------------------------------------------
    # Q2: Inventory (top-right) — pulled per-step from the trajectory.
    # ------------------------------------------------------------------
    chart_x = quadrant_w
    inv_vec = _inventory_at_step(traj, dbg)
    if inv_vec is not None:
        inv_panel = render_inventory_panel(inv_vec, quadrant_w, quadrant_h)
        _blit_exact(frame, inv_panel, chart_x, 0)
    else:
        _draw_quadrant_label(
            frame,
            chart_x,
            0,
            quadrant_w,
            quadrant_h,
            "Q2",
        )

    # ------------------------------------------------------------------
    # Q3: Action strip + legend (bottom-left)
    # ------------------------------------------------------------------
    q3_y = quadrant_h
    strip_h = quadrant_h * 3 // 5

    if dbg.action_strip_cache is not None:
        strip = draw_cursor(
            dbg.action_strip_cache,
            dbg.current_step,
            total_steps,
            color=(255, 255, 0),
        )
        _blit_exact(frame, strip, 0, q3_y)

    if dbg.action_legend_cache is not None:
        _blit_exact(frame, dbg.action_legend_cache, 0, q3_y + strip_h)

    if dbg.action_strip_cache is None and dbg.action_legend_cache is None:
        _draw_quadrant_label(
            frame,
            0,
            q3_y,
            quadrant_w,
            quadrant_h,
            "Q3",
        )

    # ------------------------------------------------------------------
    # Q4: Reward chart (bottom-right) — moved down from Q2 so the
    # inventory panel can own the top-right slot.
    # ------------------------------------------------------------------
    if dbg.reward_chart_cache is not None:
        chart = draw_cursor(
            dbg.reward_chart_cache,
            dbg.current_step,
            total_steps,
        )
        _blit_exact(frame, chart, quadrant_w, q3_y)
    else:
        _draw_quadrant_label(
            frame,
            quadrant_w,
            q3_y,
            quadrant_w,
            quadrant_h,
            "Q4",
        )

    # ------------------------------------------------------------------
    # Dividers
    # ------------------------------------------------------------------
    frame[0 : quadrant_h * 2, quadrant_w] = DIVIDER_COLOR
    frame[quadrant_h, 0 : quadrant_w * 2] = DIVIDER_COLOR

    # ------------------------------------------------------------------
    # Status bar (replay-specific)
    # ------------------------------------------------------------------
    status_y = quadrant_h * 2
    _render_replay_status_bar(
        frame,
        status_y,
        base_w,
        dbg,
        total_steps,
    )

    return frame


def rebuild_replay_caches(
    dbg: DebuggerState,
    quadrant_w: int,
    quadrant_h: int,
) -> None:
    """Pre-render cached chart images for replay mode.

    Called after loading a trajectory or switching episode/player.

    Args:
        dbg: Debugger state with trajectory set.
        quadrant_w: Width of each quadrant (used for chart width).
        quadrant_h: Height of each quadrant.
    """
    traj = dbg.trajectory
    if traj is None:
        return

    ep = dbg.selected_episode
    player = dbg.selected_player

    dbg.reward_chart_cache = render_reward_chart(
        traj,
        ep,
        quadrant_w,
        quadrant_h,
    )

    strip_h = quadrant_h * 3 // 5
    legend_h = quadrant_h - strip_h

    dbg.action_strip_cache = render_action_strip(
        traj,
        ep,
        player,
        quadrant_w,
        strip_h,
    )
    dbg.action_legend_cache = render_action_legend(
        traj,
        ep,
        player,
        quadrant_w,
        legend_h,
    )
    # Sankey dropped — Q4 now holds the reward chart.
    dbg.sankey_cache = None


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------


def _inventory_at_step(traj: object, dbg: DebuggerState) -> np.ndarray | None:
    """Extract the player inventory at the current replay step.

    Trajectories that lack ``player_inventory`` (e.g. action-only
    captures) return ``None`` and the caller falls back to the empty
    quadrant label.
    """
    from factoriax.analysis.trajectory import Trajectory

    if not isinstance(traj, Trajectory) or traj.player_inventory is None:
        return None
    ep = dbg.selected_episode
    step = dbg.current_step
    inv = traj.player_inventory[ep, step]
    if inv.ndim > 1:
        inv = inv[dbg.selected_player]
    return np.asarray(inv)


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
        frame[dy : dy + bh, dx : dx + bw] = scaled[sy : sy + bh, sx : sx + bw]


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
        frame[dy : dy + bh, dx : dx + bw] = src[sy : sy + bh, sx : sx + bw, :3]


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
) -> None:
    """Draw the status bar at the bottom of the frame (live mode)."""
    bar_h = STATUS_BAR_HEIGHT
    frame[y : y + bar_h, :width] = (30, 30, 35)
    frame[y, :width] = DIVIDER_COLOR

    parts = [
        f"MODE: {mode.upper()}",
        f"STEP: {step}/{total - 1}",
        f"PLAYER: {player}",
    ]
    if done:
        parts.append("DONE")

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


def _render_replay_status_bar(
    frame: np.ndarray,
    y: int,
    width: int,
    dbg: DebuggerState,
    total_steps: int,
) -> None:
    """Draw the status bar for replay mode."""
    bar_h = STATUS_BAR_HEIGHT
    frame[y : y + bar_h, :width] = (30, 30, 35)
    frame[y, :width] = DIVIDER_COLOR

    traj = dbg.trajectory
    parts = ["REPLAY"]
    parts.append(f"STEP: {dbg.current_step}/{total_steps - 1}")
    if traj is not None and traj.num_episodes > 1:
        parts.append(
            f"EP: {dbg.selected_episode + 1}/{traj.num_episodes}",
        )
    if traj is not None and traj.is_multi_player:
        parts.append(f"P{dbg.selected_player}")
    if dbg.playback_speed > 1:
        parts.append(f"x{dbg.playback_speed}")
    parts.append("PLAYING" if dbg.playing else "PAUSED")

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
                    if 0 <= py < frame.shape[0] and 0 <= px < frame.shape[1]:
                        frame[py, px] = color


def _draw_centered_text(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    text: str,
) -> None:
    """Draw dim centered text in a region."""
    font = get_pixel_font(10)
    text_rgba = render_text_rgba(text, font, (100, 100, 95))
    th, tw = text_rgba.shape[:2]
    tx = x + (w - tw) // 2
    ty = y + (h - th) // 2
    if 0 <= ty < frame.shape[0] - th and 0 <= tx < frame.shape[1] - tw:
        composite_rgba_over_rgb(
            frame[ty : ty + th, tx : tx + tw],
            text_rgba,
        )


# ------------------------------------------------------------------
# Fog-of-war and grid fallback (ported from inspector)
# ------------------------------------------------------------------


def _maybe_apply_fog(
    game_img: np.ndarray,
    traj: object,
    dbg: DebuggerState,
) -> np.ndarray:
    """Apply fog-of-war if the trajectory uses LOCAL observations.

    Returns the image unmodified for GLOBAL/UNDETERMINED/PLAYER types.
    """
    from factoriax.agentdebugger.obs_types import (
        ObservationType,
        apply_fog_of_war,
    )
    from factoriax.analysis.trajectory import Trajectory

    if not isinstance(traj, Trajectory):
        return game_img

    scheme = traj.observation_scheme
    if scheme is None:
        return game_img
    obs_type = scheme.get("type", ObservationType.UNDETERMINED)
    if obs_type != ObservationType.LOCAL:
        return game_img

    radius_val = scheme.get("radius", 7)
    radius = radius_val if isinstance(radius_val, int) else 7

    if traj.positions is None:
        return game_img
    pos = traj.positions[dbg.selected_episode, dbg.current_step]
    if pos.ndim > 1:
        pos = pos[dbg.selected_player]
    px, py = int(pos[0]), int(pos[1])

    if traj.block_map is not None:
        map_data = traj.block_map[dbg.selected_episode, dbg.current_step]
        map_h, map_w = map_data.shape
    else:
        map_h = max(
            1,
            int(
                traj.positions[dbg.selected_episode, :, ..., 1].max(),
            )
            + 1,
        )
        map_w = max(
            1,
            int(
                traj.positions[dbg.selected_episode, :, ..., 0].max(),
            )
            + 1,
        )

    fh, fw = game_img.shape[:2]
    tile_size = max(1, min(fw // map_w, fh // map_h))

    return apply_fog_of_war(game_img, px, py, radius, tile_size, map_w, map_h)


def _draw_world(
    frame: np.ndarray,
    traj: object,
    dbg: DebuggerState,
    wx: int,
    wy: int,
    cw: int,
    ch: int,
) -> None:
    """Draw a grid-based world view with player positions.

    Used as fallback when no rendered frames are available (trajectory
    has positions but no block_map).
    """
    from factoriax.analysis.trajectory import Trajectory
    from factoriax.renderer import PLAYER_COLORS

    if not isinstance(traj, Trajectory) or traj.positions is None:
        return

    ep = dbg.selected_episode
    step = dbg.current_step
    num_p = traj.num_players

    all_pos = traj.positions[ep]
    if all_pos.ndim == 2:
        all_pos = all_pos[:, np.newaxis, :]
    grid_w = max(int(all_pos[:, :, 0].max()) + 1, 1)
    grid_h = max(int(all_pos[:, :, 1].max()) + 1, 1)

    tile_w = max(1, cw // grid_w)
    tile_h = max(1, ch // grid_h)
    tile = min(tile_w, tile_h)

    ox = wx + (cw - grid_w * tile) // 2
    oy = wy + (ch - grid_h * tile) // 2

    # Tile backgrounds (alternating shades).
    for gy in range(grid_h):
        for gx in range(grid_w):
            x0 = ox + gx * tile
            y0 = oy + gy * tile
            shade = 32 if (gx + gy) % 2 == 0 else 38
            x1 = min(x0 + tile, frame.shape[1])
            y1 = min(y0 + tile, frame.shape[0])
            if x0 < frame.shape[1] and y0 < frame.shape[0]:
                frame[y0:y1, x0:x1] = (shade, shade, shade + 5)

    # Grid lines.
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

    # Player dots.
    for p in range(num_p):
        pos = all_pos[step, p]
        ppx, ppy = int(pos[0]), int(pos[1])
        cx = ox + ppx * tile + tile // 2
        cy = oy + ppy * tile + tile // 2

        body_color, _ = PLAYER_COLORS[p % len(PLAYER_COLORS)]
        r = max(2, tile // 3)
        for ddy in range(-r, r + 1):
            for ddx in range(-r, r + 1):
                if ddx * ddx + ddy * ddy <= r * r:
                    fy, fx = cy + ddy, cx + ddx
                    if 0 <= fy < frame.shape[0] and 0 <= fx < frame.shape[1]:
                        frame[fy, fx] = body_color
