"""State evolution analysis and visualization.

This module tracks how environment state changes over episodes:
inventory progression, resource depletion, player movement patterns,
and machine placement.

All functions require the corresponding optional fields on
:class:`~factoriax.analysis.trajectory.Trajectory` to be populated.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .trajectory import Trajectory
from .utils import resolve_ax

# Default item labels matching factoriax ItemType enum
DEFAULT_ITEM_LABELS = ["EMPTY", "COAL", "IRON", "COPPER", "MINER"]
DEFAULT_ITEM_COLORS = ["#bdbdbd", "#363636", "#c0c0c0", "#b87333", "#00c800"]


# ---------------------------------------------------------------------------
# Inventory evolution
# ---------------------------------------------------------------------------


def inventory_over_time(
    traj: Trajectory,
    player: int = 0,
    num_item_types: int = 5,
) -> np.ndarray:
    """Compute mean per-type item counts at each timestep.

    Parameters
    ----------
    traj : Trajectory
        Must have ``player_inventory`` populated. The current
        engine stores inventory as a dense ``(P, num_item_types)``
        count vector per state, so the slot-aggregation step the
        old slot-model required is gone.
    player : int

    num_item_types : int

    traj :
        Trajectory:
    player :
        int:  (Default value = 0)
    num_item_types :
        int:  (Default value = 5)
    traj: Trajectory :

    player: int :
         (Default value = 0)
    num_item_types: int :
         (Default value = 5)

    Returns
    -------

    """
    if traj.player_inventory is None:
        raise ValueError("inventory_over_time requires player_inventory")

    if traj.is_multi_player:
        inv = traj.player_inventory[:, :, player, :num_item_types]  # (B, T, K)
    else:
        inv = traj.player_inventory[..., :num_item_types]  # (B, T, K)

    return np.asarray(inv.mean(axis=0))  # (T, num_item_types)


def plot_inventory(
    traj: Trajectory,
    player: int = 0,
    num_item_types: int = 5,
    item_labels: list[str] | None = None,
    item_colors: Sequence[str] | None = None,
    exclude_empty: bool = True,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (12, 5),
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot inventory evolution over an episode.

    Parameters
    ----------
    traj : Trajectory

    player : int

    num_item_types : int

    item_labels, item_colors :

    exclude_empty : bool
        If *True*, skip the EMPTY item type.
    traj :
        Trajectory:
    player :
        int:  (Default value = 0)
    num_item_types :
        int:  (Default value = 5)
    item_labels :
        list[str] | None:  (Default value = None)
    item_colors :
        Sequence[str] | None:  (Default value = None)
    exclude_empty :
        bool:  (Default value = True)
    ax :
        Axes | None:  (Default value = None)
    figsize :
        tuple[float:
    float] :
        (Default value = (12)
    5) :

    title :
        str | None:  (Default value = None)
    traj: Trajectory :

    player: int :
         (Default value = 0)
    num_item_types: int :
         (Default value = 5)
    item_labels: list[str] | None :
         (Default value = None)
    item_colors: Sequence[str] | None :
         (Default value = None)
    exclude_empty: bool :
         (Default value = True)
    ax: Axes | None :
         (Default value = None)
    figsize: tuple[float :

    title: str | None :
         (Default value = None)

    Returns
    -------

    """
    inv = inventory_over_time(traj, player, num_item_types)  # (T, num_item_types)

    if item_labels is None:
        item_labels = DEFAULT_ITEM_LABELS[:num_item_types]
    if item_colors is None:
        item_colors = DEFAULT_ITEM_COLORS[:num_item_types]

    fig, ax = resolve_ax(ax, figsize)

    start = 1 if exclude_empty else 0
    for i in range(start, num_item_types):
        ax.plot(inv[:, i], label=item_labels[i], color=item_colors[i], linewidth=1.5)

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Mean item count")
    ax.set_title(title or f"Inventory evolution — Player {player}")
    ax.legend(fontsize="small")
    ax.set_xlim(0, inv.shape[0] - 1)

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Movement heatmap
# ---------------------------------------------------------------------------


def position_heatmap(
    traj: Trajectory,
    player: int = 0,
    map_width: int = 32,
    map_height: int = 32,
    time_range: tuple[int, int] | None = None,
) -> np.ndarray:
    """Compute a 2D visit-frequency heatmap for a player.

    Parameters
    ----------
    traj : Trajectory
        Must have ``positions``.
    player : int

    map_width, map_height :

    time_range : tuple[int

    traj :
        Trajectory:
    player :
        int:  (Default value = 0)
    map_width :
        int:  (Default value = 32)
    map_height :
        int:  (Default value = 32)
    time_range :
        tuple[int:
    int] | None :
        (Default value = None)
    traj: Trajectory :

    player: int :
         (Default value = 0)
    map_width: int :
         (Default value = 32)
    map_height: int :
         (Default value = 32)
    time_range: tuple[int :


    Returns
    -------

    """
    if traj.positions is None:
        raise ValueError("position_heatmap requires positions data")

    if traj.is_multi_player:
        pos = traj.positions[:, :, player, :]  # (B, T, 2)
    else:
        pos = traj.positions  # (B, T, 2)

    if time_range is not None:
        pos = pos[:, time_range[0] : time_range[1], :]

    heatmap = np.zeros((map_height, map_width), dtype=np.float64)
    xs = pos[:, :, 0].ravel().astype(int)
    ys = pos[:, :, 1].ravel().astype(int)

    # Clip to map bounds
    xs = np.clip(xs, 0, map_width - 1)
    ys = np.clip(ys, 0, map_height - 1)

    np.add.at(heatmap, (ys, xs), 1)
    total = heatmap.sum()
    if total > 0:
        heatmap /= total

    return heatmap


def plot_position_heatmap(
    traj: Trajectory,
    player: int = 0,
    map_width: int = 32,
    map_height: int = 32,
    time_range: tuple[int, int] | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (7, 7),
    cmap: str = "hot",
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot a spatial heatmap of player positions.

    Parameters
    ----------
    traj, player, map_width, map_height, time_range :
        See :func:`position_heatmap`.
    traj :
        Trajectory:
    player :
        int:  (Default value = 0)
    map_width :
        int:  (Default value = 32)
    map_height :
        int:  (Default value = 32)
    time_range :
        tuple[int:
    int] | None :
        (Default value = None)
    ax :
        Axes | None:  (Default value = None)
    figsize :
        tuple[float:
    float] :
        (Default value = (7)
    7) :

    cmap :
        str:  (Default value = "hot")
    title :
        str | None:  (Default value = None)
    traj: Trajectory :

    player: int :
         (Default value = 0)
    map_width: int :
         (Default value = 32)
    map_height: int :
         (Default value = 32)
    time_range: tuple[int :

    ax: Axes | None :
         (Default value = None)
    figsize: tuple[float :

    cmap: str :
         (Default value = "hot")
    title: str | None :
         (Default value = None)

    Returns
    -------

    """
    hm = position_heatmap(traj, player, map_width, map_height, time_range)

    fig, ax = resolve_ax(ax, figsize)

    im = ax.imshow(hm, cmap=cmap, interpolation="nearest", origin="upper")
    fig.colorbar(im, ax=ax, label="Visit frequency")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title or f"Position heatmap — Player {player}")

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Movement trajectory trace
# ---------------------------------------------------------------------------


def plot_trajectory_trace(
    traj: Trajectory,
    episode: int = 0,
    player: int = 0,
    map_width: int = 32,
    map_height: int = 32,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (7, 7),
    title: str | None = None,
    cmap: str = "viridis",
) -> tuple[Figure, Axes]:
    """Plot the movement path of a player for a single episode.

    Color intensity encodes time progression (dark = early, bright = late).

    Parameters
    ----------
    traj : Trajectory
        Must have ``positions``.
    episode, player :

    map_width, map_height :

    traj :
        Trajectory:
    episode :
        int:  (Default value = 0)
    player :
        int:  (Default value = 0)
    map_width :
        int:  (Default value = 32)
    map_height :
        int:  (Default value = 32)
    ax :
        Axes | None:  (Default value = None)
    figsize :
        tuple[float:
    float] :
        (Default value = (7)
    7) :

    title :
        str | None:  (Default value = None)
    cmap :
        str:  (Default value = "viridis")
    traj: Trajectory :

    episode: int :
         (Default value = 0)
    player: int :
         (Default value = 0)
    map_width: int :
         (Default value = 32)
    map_height: int :
         (Default value = 32)
    ax: Axes | None :
         (Default value = None)
    figsize: tuple[float :

    title: str | None :
         (Default value = None)
    cmap: str :
         (Default value = "viridis")

    Returns
    -------

    """
    if traj.positions is None:
        raise ValueError("plot_trajectory_trace requires positions data")

    if traj.is_multi_player:
        pos = traj.positions[episode, :, player, :]  # (T, 2)
    else:
        pos = traj.positions[episode, :, :]  # (T, 2)

    fig, ax = resolve_ax(ax, figsize)

    T = pos.shape[0]
    colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, T))

    ax.scatter(pos[:, 0], pos[:, 1], c=colors, s=3, zorder=2)
    ax.plot(pos[:, 0], pos[:, 1], color="gray", alpha=0.2, linewidth=0.5, zorder=1)

    # Mark start and end
    ax.scatter(
        pos[0, 0],
        pos[0, 1],
        marker="o",
        s=80,
        c="green",
        edgecolors="black",
        zorder=3,
        label="Start",
    )
    ax.scatter(
        pos[-1, 0],
        pos[-1, 1],
        marker="s",
        s=80,
        c="red",
        edgecolors="black",
        zorder=3,
        label="End",
    )

    ax.set_xlim(-0.5, map_width - 0.5)
    ax.set_ylim(map_height - 0.5, -0.5)  # Invert y to match grid convention
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title or f"Trajectory — Episode {episode}, Player {player}")
    ax.legend(fontsize="small")
    ax.set_aspect("equal")

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Resource depletion
# ---------------------------------------------------------------------------


def plot_resource_depletion(
    traj: Trajectory,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (12, 5),
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot total remaining block resources over time.

    Requires the ``block_resources`` field on the trajectory,
    shaped ``(B, T, H, W)``.  This is not stored by default; researchers
    need to snapshot ``env_state.block_resources`` during rollouts.

    Parameters
    ----------
    traj :
        Trajectory:
    ax :
        Axes | None:  (Default value = None)
    figsize :
        tuple[float:
    float] :
        (Default value = (12)
    5) :

    title :
        str | None:  (Default value = None)
    traj: Trajectory :

    ax: Axes | None :
         (Default value = None)
    figsize: tuple[float :

    title: str | None :
         (Default value = None)

    Returns
    -------

    """
    if traj.block_resources is None:
        raise ValueError(
            "plot_resource_depletion requires block_resources data. "
            "Snapshot env_state.block_resources during rollouts."
        )

    resources = traj.block_resources  # (B, T, H, W)
    total = resources.sum(axis=(-1, -2))  # (B, T)
    mean = total.mean(axis=0)
    std = total.std(axis=0)

    fig, ax = resolve_ax(ax, figsize)

    T = mean.shape[0]
    ax.plot(mean, color="#d62728", linewidth=1.5)
    ax.fill_between(range(T), mean - std, mean + std, alpha=0.2, color="#d62728")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Total remaining resources")
    ax.set_title(title or "Resource depletion over episode")
    ax.set_xlim(0, T - 1)

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Episode reward curves
# ---------------------------------------------------------------------------


def plot_episode_rewards(
    traj: Trajectory,
    episode: int = 0,
    figsize: tuple[float, float] = (10, 6),
    title: str | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot per-step and cumulative rewards for a single episode.

    The top panel shows a bar chart of per-step rewards and the bottom
    panel shows the cumulative reward curve with shaded area.

    Parameters
    ----------
    traj :
        Trajectory with the ``rewards`` field populated.
    episode :
        Episode index to plot.
    figsize :
        Figure size.
    title :
        Title for the top panel.
    traj :
        Trajectory:
    episode :
        int:  (Default value = 0)
    figsize :
        tuple[float:
    float] :
        (Default value = (10)
    6) :

    title :
        str | None:  (Default value = None)
    traj: Trajectory :

    episode: int :
         (Default value = 0)
    figsize: tuple[float :

    title: str | None :
         (Default value = None)

    Returns
    -------

        Tuple of ``(fig, axes)`` where *axes* is a length-2 array

        Tuple of ``(fig, axes)`` where *axes* is a length-2 array
        of the step-reward and cumulative-reward axes.

    Raises
    ------
    ValueError
        If the trajectory has no ``rewards`` field.

    """
    if traj.rewards is None:
        raise ValueError("plot_episode_rewards requires the rewards field.")

    r = traj.rewards[episode].astype(np.float64)
    cumulative = np.cumsum(r)
    steps = np.arange(len(r))

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    ax_step, ax_cum = axes

    ax_step.bar(steps, r, width=1.0, color="steelblue", alpha=0.7)
    ax_step.set_ylabel("Step reward")
    ax_step.set_title(title or "Evaluation episode rewards")
    ax_step.grid(axis="y", linestyle="--", alpha=0.4)

    ax_cum.plot(steps, cumulative, color="darkorange", linewidth=1.5)
    ax_cum.fill_between(
        steps,
        0,
        cumulative,
        color="darkorange",
        alpha=0.2,
    )
    ax_cum.set_xlabel("Timestep")
    ax_cum.set_ylabel("Cumulative reward")
    ax_cum.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    return fig, axes
