"""Chart rendering: reward line plot and action color strip.

Charts are rendered once when an episode is loaded, then cached as
numpy RGB arrays. Each frame only needs to draw a thin cursor line
on top of the cached image, which is extremely cheap.
"""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from factoriax.analysis.actions import DEFAULT_ACTION_COLORS, DEFAULT_ACTION_LABELS
from factoriax.analysis.trajectory import Trajectory
from factoriax.constants import Action


def _fig_to_rgb(fig: plt.Figure, width: int, height: int) -> np.ndarray:
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
    rgba = np.asarray(fig.canvas.buffer_rgba())
    img = rgba[:, :, :3].copy()
    plt.close(fig)
    # Resize to exact target if rounding caused a mismatch.
    if img.shape[0] != height or img.shape[1] != width:
        # Simple nearest-neighbor resize via numpy (no PIL dependency).
        ys = np.linspace(0, img.shape[0] - 1, height).astype(int)
        xs = np.linspace(0, img.shape[1] - 1, width).astype(int)
        img = img[np.ix_(ys, xs)]
    return img


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
        ax.plot(cumulative, color="#4fc3f7", linewidth=1.2, label="cumulative")
        ax.plot(rewards, color="#81c784", linewidth=0.6, alpha=0.5, label="per-step")
        ax.legend(
            fontsize=7, facecolor="#1e1e1e", edgecolor="#555555", labelcolor="#aaaaaa"
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
    return _fig_to_rgb(fig, width, height)


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
    count.

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
    ax.set_xticklabels(sub_labels, rotation=45, ha="right", fontsize=5, color="#aaaaaa")
    ax.set_yticklabels(sub_labels, fontsize=5, color="#aaaaaa")
    ax.set_xlabel("Next", fontsize=6, color="#aaaaaa")
    ax.set_ylabel("Current", fontsize=6, color="#aaaaaa")
    ax.tick_params(colors="#555555", length=2)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=5, colors="#aaaaaa")
    cbar.outline.set_edgecolor("#555555")

    fig.tight_layout(pad=0.3)
    return _fig_to_rgb(fig, width, height)


def draw_cursor(
    img: np.ndarray,
    step: int,
    total_steps: int,
    color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Draw a vertical cursor line on a chart image (returns a copy).

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
    x = int(step * (img.shape[1] - 1) / max(1, total_steps - 1))
    x = max(0, min(x, img.shape[1] - 1))
    result[:, x] = color
    if x + 1 < img.shape[1]:
        result[:, x + 1] = color
    return result
