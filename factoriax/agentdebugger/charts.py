"""Chart rendering for the debugger.

All matplotlib-based chart rendering lives here: reward line plots,
constraint cost charts, action color strips, action legends, and
action transition heatmaps (Sankey). Charts are rendered once when
data changes, then cached as numpy RGB arrays. Each frame only draws
a thin cursor line on top of the cached image.
"""

from __future__ import annotations

import matplotlib
import matplotlib.figure
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from factoriax.analysis.actions import (
    DEFAULT_ACTION_COLORS,
    DEFAULT_ACTION_LABELS,
)
from factoriax.analysis.trajectory import Trajectory
from factoriax.constants import ITEM_COLORS, Action, ItemType
from factoriax.ui.compositing import composite_rgba_over_rgb
from factoriax.ui.fonts import get_pixel_font, render_text_rgba

__all__ = [
    "build_partial_trajectory",
    "draw_cursor",
    "render_action_legend",
    "render_action_sankey",
    "render_action_strip",
    "render_cost_chart",
    "render_inventory_panel",
    "render_reward_chart",
]


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------


def _fig_to_rgb(
    fig: matplotlib.figure.Figure,
    width: int,
    height: int,
) -> np.ndarray:
    """Render a matplotlib figure to a fixed-size RGB numpy array.

    Args:
        fig: Matplotlib figure to render.
        width: Desired output width in pixels.
        height: Desired output height in pixels.

    Returns:
        RGB uint8 array of shape ``(height, width, 3)``.
    """
    fig.set_size_inches(width / fig.dpi, height / fig.dpi)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())  # type: ignore[attr-defined]
    img = rgba[:, :, :3].copy()
    plt.close(fig)
    # Resize to exact target if rounding caused a mismatch.
    if img.shape[0] != height or img.shape[1] != width:
        ys = np.linspace(0, img.shape[0] - 1, height).astype(int)
        xs = np.linspace(0, img.shape[1] - 1, width).astype(int)
        img = img[np.ix_(ys, xs)]
    return img


def _action_color_lut() -> tuple[np.ndarray, list[str], list[str]]:
    """Build the action color lookup table, labels, and hex codes.

    Returns:
        ``(color_lut, labels, hex_codes)`` where color_lut is shape
        ``(num_actions, 3)`` uint8.
    """
    num_actions = len(Action)
    colors_hex = list(DEFAULT_ACTION_COLORS)
    while len(colors_hex) < num_actions:
        colors_hex.append("#888888")
    labels = list(DEFAULT_ACTION_LABELS)
    while len(labels) < num_actions:
        labels.append(f"ACT_{len(labels)}")

    color_lut = np.zeros((num_actions, 3), dtype=np.uint8)
    for i, hc in enumerate(colors_hex[:num_actions]):
        hc = hc.lstrip("#")
        color_lut[i] = [int(hc[0:2], 16), int(hc[2:4], 16), int(hc[4:6], 16)]
    return color_lut, labels, colors_hex[:num_actions]


# ------------------------------------------------------------------
# Partial trajectory helper (for live-stepping mode)
# ------------------------------------------------------------------


def build_partial_trajectory(
    rewards: list[float],
    actions: list[int],
) -> Trajectory:
    """Build a single-episode Trajectory from accumulated step data.

    The returned trajectory has batch dimension 1 and time dimension
    equal to the number of steps recorded so far.

    Args:
        rewards: Per-step reward values.
        actions: Per-step action integers.

    Returns:
        A minimal :class:`Trajectory` suitable for
        :func:`render_reward_chart`.
    """
    t = max(len(actions), len(rewards), 1)
    act = np.array(actions, dtype=np.int32) if actions else np.zeros(1, dtype=np.int32)
    rew = (
        np.array(rewards, dtype=np.float32)
        if rewards
        else np.zeros(1, dtype=np.float32)
    )
    # Pad to equal length if they differ.
    if len(act) < t:
        act = np.pad(act, (0, t - len(act)))
    if len(rew) < t:
        rew = np.pad(rew, (0, t - len(rew)))
    return Trajectory(
        actions=act.reshape(1, t),
        rewards=rew.reshape(1, t),
    )


# ------------------------------------------------------------------
# Reward chart
# ------------------------------------------------------------------


def render_reward_chart(
    traj: Trajectory,
    episode: int,
    width: int,
    height: int,
) -> np.ndarray:
    """Render a cumulative reward line chart as an RGB image.

    Args:
        traj: Loaded trajectory (may have rewards=None).
        episode: Episode index to plot.
        width: Output width in pixels.
        height: Output height in pixels.

    Returns:
        RGB uint8 array of shape ``(height, width, 3)``.
    """
    fig, ax = plt.subplots(dpi=100)
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#1e1e1e")
    ax.tick_params(colors="#aaaaaa", labelsize=7)
    ax.spines["bottom"].set_color("#555555")
    ax.spines["left"].set_color("#555555")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if traj.rewards is not None:
        rewards = traj.rewards[episode]
        cumulative = np.cumsum(rewards)
        ax.plot(
            cumulative,
            color="#4fc3f7",
            linewidth=1.2,
            label="cumulative",
        )
        ax.plot(
            rewards,
            color="#81c784",
            linewidth=0.6,
            alpha=0.5,
            label="per-step",
        )
        ax.legend(
            fontsize=7,
            facecolor="#1e1e1e",
            edgecolor="#555555",
            labelcolor="#aaaaaa",
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No reward data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="#777777",
            fontsize=10,
        )

    ax.set_xlabel("Step", fontsize=8, color="#aaaaaa")
    ax.set_ylabel("Reward", fontsize=8, color="#aaaaaa")
    fig.tight_layout(pad=0.5)

    total_steps = len(traj.rewards[episode]) if traj.rewards is not None else 1
    ax.set_xlim(0, max(1, total_steps - 1))
    fig.canvas.draw()

    bbox = ax.get_position()
    frac_x0 = int(bbox.x0 * 255)
    frac_x1 = int(bbox.x1 * 255)

    img = _fig_to_rgb(fig, width, height)
    # Embed plot-area bounds for cursor positioning.
    img[0, 0, 0] = min(max(frac_x0, 1), 255)
    img[0, 1, 0] = min(max(frac_x1, 1), 255)
    return img


# ------------------------------------------------------------------
# Cost chart
# ------------------------------------------------------------------


def render_cost_chart(
    costs: list[np.ndarray],
    names: list[str],
    width: int,
    height: int,
) -> np.ndarray:
    """Render constraint costs over time as a multi-line chart.

    One line per constraint dimension, colored distinctly. Uses the
    same dark theme as the reward chart.

    Args:
        costs: Per-step cost vectors (each element shape ``(K,)``).
        names: Human-readable label for each constraint dimension.
        width: Output width in pixels.
        height: Output height in pixels.

    Returns:
        RGB uint8 array of shape ``(height, width, 3)``.
    """
    fig, ax = plt.subplots(dpi=100)
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#1e1e1e")
    ax.tick_params(colors="#aaaaaa", labelsize=7)
    ax.spines["bottom"].set_color("#555555")
    ax.spines["left"].set_color("#555555")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if costs:
        cost_arr = np.stack(costs)  # (T, K)
        colors = ["#ef5350", "#ffa726", "#ab47bc", "#26c6da"]
        for k in range(cost_arr.shape[1]):
            label = names[k] if k < len(names) else f"cost_{k}"
            color = colors[k % len(colors)]
            ax.plot(
                cost_arr[:, k],
                color=color,
                linewidth=1.0,
                label=label,
            )
        ax.legend(
            fontsize=6,
            facecolor="#1e1e1e",
            edgecolor="#555555",
            labelcolor="#aaaaaa",
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No cost data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="#777777",
            fontsize=10,
        )

    ax.set_xlabel("Step", fontsize=8, color="#aaaaaa")
    ax.set_ylabel("Cost", fontsize=8, color="#aaaaaa")
    fig.tight_layout(pad=0.5)

    total = len(costs) if costs else 1
    ax.set_xlim(0, max(1, total - 1))
    fig.canvas.draw()

    bbox = ax.get_position()
    frac_x0 = int(bbox.x0 * 255)
    frac_x1 = int(bbox.x1 * 255)

    img = _fig_to_rgb(fig, width, height)
    img[0, 0, 0] = min(max(frac_x0, 1), 255)
    img[0, 1, 0] = min(max(frac_x1, 1), 255)
    return img


# ------------------------------------------------------------------
# Cursor overlay
# ------------------------------------------------------------------


def draw_cursor(
    img: np.ndarray,
    step: int,
    total_steps: int,
    color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Draw a vertical cursor line on a chart image (returns a copy).

    If the image has plot area bounds embedded in pixels (0,0) and
    (0,1) by :func:`render_reward_chart`, the cursor is mapped to the
    plot area. Otherwise falls back to full-width mapping (correct for
    the action strip which has no margins).

    Args:
        img: Source RGB image.
        step: Current step index.
        total_steps: Total number of steps in the episode.
        color: RGB cursor color.

    Returns:
        New RGB array with cursor drawn.
    """
    result = img.copy()
    if total_steps <= 0:
        return result

    frac_x0 = int(img[0, 0, 0])
    frac_x1 = int(img[0, 1, 0])
    w = img.shape[1]
    if frac_x0 > 0 and frac_x1 > frac_x0:
        plot_x0 = int(frac_x0 / 255.0 * w)
        plot_x1 = int(frac_x1 / 255.0 * w)
        x = plot_x0 + int(
            step * (plot_x1 - plot_x0) / max(1, total_steps - 1),
        )
    else:
        x = int(step * (w - 1) / max(1, total_steps - 1))

    x = max(0, min(x, img.shape[1] - 1))
    result[:, x] = color
    if x + 1 < img.shape[1]:
        result[:, x + 1] = color
    return result


# ------------------------------------------------------------------
# Action strip
# ------------------------------------------------------------------


def render_action_strip(
    traj: Trajectory,
    episode: int,
    player: int,
    width: int,
    height: int,
) -> np.ndarray:
    """Render a color-coded action timeline as an RGB image.

    Each column of pixels corresponds to one timestep, colored by
    the action taken. Pure numpy, no matplotlib needed.

    Args:
        traj: Loaded trajectory.
        episode: Episode index.
        player: Player index (ignored for single-player trajectories).
        width: Output width in pixels.
        height: Output height in pixels.

    Returns:
        RGB uint8 array of shape ``(height, width, 3)``.
    """
    ep = traj.episode(episode)
    if ep.is_multi_player:
        actions = ep.actions[0, :, player]
    else:
        actions = ep.actions[0]

    total_steps = len(actions)
    num_actions = len(Action)
    color_lut, _, _ = _action_color_lut()

    # Build strip: one column per step, stretched to width.
    strip = np.zeros((1, total_steps, 3), dtype=np.uint8)
    for t in range(total_steps):
        a = int(actions[t])
        if 0 <= a < num_actions:
            strip[0, t] = color_lut[a]

    # Scale to output dimensions.
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for x in range(width):
        t = min(int(x * total_steps / width), total_steps - 1)
        img[:, x] = strip[0, t]

    return img


# ------------------------------------------------------------------
# Action legend
# ------------------------------------------------------------------


def render_action_legend(
    traj: Trajectory,
    episode: int,
    player: int,
    width: int,
    height: int,
) -> np.ndarray:
    """Render a compact legend showing which color maps to which action.

    Only actions that actually appear in the episode are shown to keep
    the legend small.

    Args:
        traj: Loaded trajectory.
        episode: Episode index.
        player: Player index.
        width: Output width in pixels.
        height: Output height in pixels.

    Returns:
        RGB uint8 array of shape ``(height, width, 3)``.
    """
    ep = traj.episode(episode)
    if ep.is_multi_player:
        actions = ep.actions[0, :, player]
    else:
        actions = ep.actions[0]

    color_lut, labels, _ = _action_color_lut()
    present = sorted(set(int(a) for a in actions))

    fig, ax = plt.subplots(dpi=100)
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#1e1e1e")
    ax.axis("off")

    x_pos = 0.02
    y_pos = 0.5
    for a in present:
        if a < 0 or a >= len(labels):
            continue
        c = tuple(color_lut[a] / 255.0)
        ax.plot(
            x_pos,
            y_pos,
            "s",
            color=c,
            markersize=6,
            transform=ax.transAxes,
        )
        ax.text(
            x_pos + 0.02,
            y_pos,
            labels[a],
            fontsize=6,
            color="#cccccc",
            va="center",
            transform=ax.transAxes,
        )
        x_pos += max(0.07, len(labels[a]) * 0.008 + 0.04)
        if x_pos > 0.95:
            break

    fig.tight_layout(pad=0.1)
    return _fig_to_rgb(fig, width, height)


# ------------------------------------------------------------------
# Action transition heatmap (Sankey)
# ------------------------------------------------------------------


def render_action_sankey(
    traj: Trajectory,
    episode: int,
    player: int,
    width: int,
    height: int,
) -> np.ndarray:
    """Render an action transition diagram as an RGB image.

    Shows the most common action-to-action transitions as a flow
    matrix heatmap. The x-axis is "next action" and y-axis is
    "current action", with cell intensity proportional to transition
    probability.

    Args:
        traj: Loaded trajectory.
        episode: Episode index.
        player: Player index.
        width: Output width in pixels.
        height: Output height in pixels.

    Returns:
        RGB uint8 array of shape ``(height, width, 3)``.
    """
    ep = traj.episode(episode)
    if ep.is_multi_player:
        actions = ep.actions[0, :, player]
    else:
        actions = ep.actions[0]

    color_lut, labels, hex_codes = _action_color_lut()
    num_actions = len(Action)

    # Build transition matrix.
    mat = np.zeros((num_actions, num_actions), dtype=np.float64)
    for t in range(len(actions) - 1):
        a_from = int(actions[t])
        a_to = int(actions[t + 1])
        if 0 <= a_from < num_actions and 0 <= a_to < num_actions:
            mat[a_from, a_to] += 1

    # Only show actions that appear.
    present = sorted(
        i for i in range(num_actions) if mat[i].sum() > 0 or mat[:, i].sum() > 0
    )
    if not present:
        present = list(range(min(5, num_actions)))

    sub_mat = mat[np.ix_(present, present)]
    sub_labels = [labels[i] for i in present]

    # Normalize rows to probabilities.
    row_sums = sub_mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    prob_mat = sub_mat / row_sums

    fig, ax = plt.subplots(dpi=100)
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#1e1e1e")

    im = ax.imshow(prob_mat, cmap="inferno", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(sub_labels)))
    ax.set_yticks(range(len(sub_labels)))
    ax.set_xticklabels(
        sub_labels,
        rotation=45,
        ha="right",
        fontsize=5,
        color="#aaaaaa",
    )
    ax.set_yticklabels(sub_labels, fontsize=5, color="#aaaaaa")
    ax.set_xlabel("Next", fontsize=6, color="#aaaaaa")
    ax.set_ylabel("Current", fontsize=6, color="#aaaaaa")
    ax.tick_params(colors="#555555", length=2)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=5, colors="#aaaaaa")
    cbar.outline.set_edgecolor("#555555")  # type: ignore[operator]

    fig.tight_layout(pad=0.3)
    return _fig_to_rgb(fig, width, height)


# ------------------------------------------------------------------
# Inventory panel
# ------------------------------------------------------------------


# Every non-EMPTY ItemType, ordered by enum value so the layout is stable.
_INVENTORY_ITEMS: tuple[ItemType, ...] = tuple(
    it for it in ItemType if it != ItemType.EMPTY
)


def _item_label(item: ItemType) -> str:
    """Human-readable label for an ``ItemType`` (title-cased, 12 char cap)."""
    name = item.name.replace("_", " ").title()
    return name if len(name) <= 14 else name[:13] + "."


def _inventory_slot_positions(
    width: int,
    height: int,
    num_items: int,
    *,
    row_top: int,
    min_row_h: int = 10,
    max_row_h: int = 16,
    pad_x: int = 6,
) -> list[tuple[int, int, int, int]]:
    """Compute ``(x, y, col_w, row_h)`` for each inventory item.

    Picks a single-column layout when the quadrant is tall enough; else
    splits into two columns so every item has a visible slot. The return
    list length always equals ``num_items`` — items in the right column
    are at indices ``ceil(num_items / 2)`` onward.

    Args:
        width: Panel width in pixels.
        height: Panel height in pixels.
        num_items: Number of inventory rows to place.
        row_top: Y offset after the title bar.
        min_row_h: Minimum per-row height that still fits the label font.
        max_row_h: Maximum per-row height (single-column keeps rows
            readable when the quadrant is oversized).
        pad_x: Left/right padding in pixels.

    Returns:
        List of ``(x, y, col_w, row_h)`` tuples in item order.
    """
    if num_items <= 0:
        return []
    avail = max(1, height - row_top - 2)
    one_col_row_h = min(max_row_h, max(min_row_h, avail // num_items))
    if one_col_row_h * num_items <= avail:
        # Single column fits.
        col_w = width - 2 * pad_x
        return [
            (pad_x, row_top + i * one_col_row_h, col_w, one_col_row_h)
            for i in range(num_items)
        ]
    # Two columns. Split items into left/right halves so column 0 holds
    # the first ceil(n/2) items and column 1 the rest.
    left_n = (num_items + 1) // 2
    right_n = num_items - left_n
    rows_per_col = max(left_n, right_n)
    two_col_row_h = min(max_row_h, max(min_row_h, avail // rows_per_col))
    col_w = (width - 3 * pad_x) // 2
    left_x = pad_x
    right_x = pad_x + col_w + pad_x
    positions: list[tuple[int, int, int, int]] = []
    for i in range(num_items):
        if i < left_n:
            col_x = left_x
            row_i = i
        else:
            col_x = right_x
            row_i = i - left_n
        positions.append((col_x, row_top + row_i * two_col_row_h, col_w, two_col_row_h))
    return positions


def render_inventory_panel(
    inventory: np.ndarray,
    width: int,
    height: int,
    *,
    title: str = "Inventory",
) -> np.ndarray:
    """Render the player inventory as an RGB panel.

    Each non-EMPTY ``ItemType`` is shown on its own row with a color
    swatch, label, and count. Rows with count 0 are dimmed so the eye
    can sweep to what the agent is actually holding.

    Args:
        inventory: 1-D array of per-item counts indexed by ``ItemType``.
        width: Output width in pixels.
        height: Output height in pixels.
        title: Panel heading.

    Returns:
        RGB uint8 array of shape ``(height, width, 3)``.
    """
    img = np.full((height, width, 3), (30, 30, 35), dtype=np.uint8)
    font = get_pixel_font(11)
    title_font = get_pixel_font(12)

    # Title bar.
    title_rgba = render_text_rgba(title, title_font, (210, 210, 200))
    th = title_rgba.shape[0]
    pad_x = 6
    top_pad = 4
    if top_pad + th <= height and pad_x + title_rgba.shape[1] <= width:
        composite_rgba_over_rgb(
            img[top_pad : top_pad + th, pad_x : pad_x + title_rgba.shape[1]],
            title_rgba,
        )

    row_top = top_pad + th + 4
    positions = _inventory_slot_positions(
        width,
        height,
        len(_INVENTORY_ITEMS),
        row_top=row_top,
        pad_x=pad_x,
    )

    for item, (slot_x, y, col_w, row_h) in zip(
        _INVENTORY_ITEMS, positions, strict=True
    ):
        if y + row_h > height:
            continue
        swatch_sz = max(6, row_h - 4)
        count = int(inventory[int(item)]) if int(item) < inventory.shape[0] else 0
        active = count > 0

        # Color swatch.
        sx = slot_x
        sy = y + (row_h - swatch_sz) // 2
        rgb = ITEM_COLORS.get(int(item), (120, 120, 120))
        if not active:
            rgb = tuple(max(0, c // 3) for c in rgb)
        img[sy : sy + swatch_sz, sx : sx + swatch_sz] = rgb

        # Label.
        label = _item_label(item)
        label_color = (220, 220, 210) if active else (110, 110, 110)
        label_rgba = render_text_rgba(label, font, label_color)
        lh, lw = label_rgba.shape[:2]
        lx = sx + swatch_sz + 5
        ly = y + (row_h - lh) // 2
        # Count area is ~20 px from the right edge of this slot's column.
        col_right = slot_x + col_w
        if lx + lw < col_right - 20 and 0 <= ly and ly + lh <= height:
            composite_rgba_over_rgb(img[ly : ly + lh, lx : lx + lw], label_rgba)

        # Count, right-aligned inside this slot's column.
        if active:
            count_str = str(count)
            count_color = (130, 220, 150)
        else:
            count_str = "-"
            count_color = (90, 90, 95)
        count_rgba = render_text_rgba(count_str, font, count_color)
        ch, cw = count_rgba.shape[:2]
        cx = col_right - cw
        cy = y + (row_h - ch) // 2
        if cx >= 0 and 0 <= cy and cy + ch <= height:
            composite_rgba_over_rgb(img[cy : cy + ch, cx : cx + cw], count_rgba)

    return img
