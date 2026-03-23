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
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = buf.reshape(
        int(fig.get_figheight() * fig.dpi), int(fig.get_figwidth() * fig.dpi), 3
    )
    plt.close(fig)
    # Resize to exact target if rounding caused a mismatch.
    if img.shape[0] != height or img.shape[1] != width:
        from PIL import Image

        pil = Image.fromarray(img).resize((width, height), Image.NEAREST)
        img = np.array(pil)
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

    # Pad the color list if new actions were added.
    colors_hex = list(DEFAULT_ACTION_COLORS)
    while len(colors_hex) < num_actions:
        colors_hex.append("#888888")
    labels = list(DEFAULT_ACTION_LABELS)
    while len(labels) < num_actions:
        labels.append(f"ACT_{len(labels)}")

    # Convert hex to RGB tuples.
    color_lut = np.zeros((num_actions, 3), dtype=np.uint8)
    for i, hex_color in enumerate(colors_hex[:num_actions]):
        hex_color = hex_color.lstrip("#")
        color_lut[i] = [
            int(hex_color[0:2], 16),
            int(hex_color[2:4], 16),
            int(hex_color[4:6], 16),
        ]

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
