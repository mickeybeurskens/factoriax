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

# Default item labels matching factoriax ItemType enum
DEFAULT_ITEM_LABELS = ["EMPTY", "COAL", "IRON", "COPPER", "MINER"]
DEFAULT_ITEM_COLORS = ["#bdbdbd", "#363636", "#c0c0c0", "#b87333", "#00c800"]


# ---------------------------------------------------------------------------
# Inventory evolution
# ---------------------------------------------------------------------------


def inventory_over_time(
    traj,
    player: int = 0,
    num_item_types: int = 5,
) -> np.ndarray:
    """Compute total item counts per type at each timestep.

    Parameters
    ----------
    traj : Trajectory
        Must have ``inventory_items`` and ``inventory_counts``.
    player : int
    num_item_types : int

    Returns
    -------
    counts : np.ndarray
        Shape ``(T, num_item_types)`` — mean across episodes of total
        inventory count per item type.
    """
    if traj.inventory_items is None or traj.inventory_counts is None:
        raise ValueError(
            "inventory_over_time requires inventory_items and inventory_counts"
        )

    # Get single-player view
    if traj.is_multi_player:
        items = traj.inventory_items[:, :, player, :]  # (B, T, slots)
        counts = traj.inventory_counts[:, :, player, :]  # (B, T, slots)
    else:
        items = traj.inventory_items  # (B, T, slots)
        counts = traj.inventory_counts  # (B, T, slots)

    B, T, S = items.shape
    result = np.zeros((B, T, num_item_types))

    for item_type in range(num_item_types):
        mask = items == item_type
        result[:, :, item_type] = (counts * mask).sum(axis=-1)

    return result.mean(axis=0)  # (T, num_item_types)


def plot_inventory(
    traj,
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
    item_labels, item_colors : optional
    exclude_empty : bool
        If *True*, skip the EMPTY item type.

    Returns
    -------
    fig, ax : Figure, Axes
    """
    inv = inventory_over_time(traj, player, num_item_types)  # (T, num_item_types)

    if item_labels is None:
        item_labels = DEFAULT_ITEM_LABELS[:num_item_types]
    if item_colors is None:
        item_colors = DEFAULT_ITEM_COLORS[:num_item_types]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

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
    traj,
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
    map_width, map_height : int
    time_range : tuple[int, int], optional

    Returns
    -------
    heatmap : np.ndarray
        Shape ``(map_height, map_width)``, normalized to sum to 1.
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
    traj,
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
    traj, player, map_width, map_height, time_range
        See :func:`position_heatmap`.

    Returns
    -------
    fig, ax : Figure, Axes
    """
    hm = position_heatmap(traj, player, map_width, map_height, time_range)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

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
    traj,
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
    episode, player : int
    map_width, map_height : int

    Returns
    -------
    fig, ax : Figure, Axes
    """
    if traj.positions is None:
        raise ValueError("plot_trajectory_trace requires positions data")

    if traj.is_multi_player:
        pos = traj.positions[episode, :, player, :]  # (T, 2)
    else:
        pos = traj.positions[episode, :, :]  # (T, 2)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    T = pos.shape[0]
    colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, T))

    ax.scatter(pos[:, 0], pos[:, 1], c=colors, s=3, zorder=2)
    ax.plot(pos[:, 0], pos[:, 1], color="gray", alpha=0.2, linewidth=0.5, zorder=1)

    # Mark start and end
    ax.scatter(
        *pos[0],
        marker="o",
        s=80,
        c="green",
        edgecolors="black",
        zorder=3,
        label="Start",
    )
    ax.scatter(
        *pos[-1], marker="s", s=80, c="red", edgecolors="black", zorder=3, label="End"
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
    traj,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (12, 5),
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot total remaining block resources over time.

    Requires the ``block_resources`` field on the trajectory,
    shaped ``(B, T, H, W)``.  This is not stored by default; researchers
    need to snapshot ``env_state.block_resources`` during rollouts.

    Returns
    -------
    fig, ax : Figure, Axes
    """
    if not hasattr(traj, "metadata") or "block_resources" not in (traj.metadata or {}):
        # Check if it's stored as a direct field via metadata
        raise ValueError(
            "plot_resource_depletion requires block_resources data. "
            "Store it via trajectory metadata or extend the Trajectory class."
        )

    resources = traj.metadata["block_resources"]  # (B, T, H, W)
    total = resources.sum(axis=(-1, -2))  # (B, T)
    mean = total.mean(axis=0)
    std = total.std(axis=0)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    T = mean.shape[0]
    ax.plot(mean, color="#d62728", linewidth=1.5)
    ax.fill_between(range(T), mean - std, mean + std, alpha=0.2, color="#d62728")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Total remaining resources")
    ax.set_title(title or "Resource depletion over episode")
    ax.set_xlim(0, T - 1)

    fig.tight_layout()
    return fig, ax
