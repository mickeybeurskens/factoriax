"""Chart rendering for the debugger.

Builds a lightweight :class:`~factoriax.analysis.trajectory.Trajectory`
from accumulated step data and delegates to the inspector's chart
functions. Also provides a cost chart for constraint dimensions.
"""

from __future__ import annotations

import matplotlib
import matplotlib.figure
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from factoriax.analysis.trajectory import Trajectory
from factoriax.inspector.charts import (
    _fig_to_rgb,
    draw_cursor,
    render_reward_chart,
)

__all__ = [
    "build_partial_trajectory",
    "draw_cursor",
    "render_cost_chart",
    "render_reward_chart",
]


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


def render_cost_chart(
    costs: list[np.ndarray],
    names: list[str],
    width: int,
    height: int,
) -> np.ndarray:
    """Render constraint costs over time as a multi-line chart.

    One line per constraint dimension, colored distinctly. Uses the
    same dark theme as the inspector charts.

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
            0.5, 0.5, "No cost data",
            transform=ax.transAxes,
            ha="center", va="center",
            color="#777777", fontsize=10,
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
