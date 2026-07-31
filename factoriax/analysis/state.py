"""Read and draw how the world changed over an episode.

The module covers four subjects. Each has a plot, and two also have a
finder that returns the numbers. The subjects are the inventory of a
player, where a player walked, the resource left on the map, and the
reward at each step.

Every function reads an optional field of
:class:`~factoriax.analysis.trajectory.Trajectory` and raises
``ValueError`` when that field was not recorded.

Two coordinate orders meet in this module. A position is ordered
``(x, y)``, following the engine. A map array is indexed ``[y, x]``,
following row-major order. The plots invert the y axis so that a
picture matches the map, with y growing downward.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .trajectory import Trajectory
from .utils import resolve_ax

#: Fallback item names for :func:`plot_inventory`, used when the caller
#: passes none.
#:
#: These do not match the engine. ``ItemType`` gives ``IRON_ORE``,
#: ``COPPER_ORE``, and ``TIN_ORE`` at ids 2, 3, and 4, so a default plot
#: labels the tin ore line "MINER". Pass real ``ItemType`` names.
DEFAULT_ITEM_LABELS = ["EMPTY", "COAL", "IRON", "COPPER", "MINER"]

#: Fallback line colors, one for each entry of
#: :data:`DEFAULT_ITEM_LABELS` and in the same order.
DEFAULT_ITEM_COLORS = ["#bdbdbd", "#363636", "#c0c0c0", "#b87333", "#00c800"]


# ---------------------------------------------------------------------------
# Inventory evolution
# ---------------------------------------------------------------------------


def inventory_over_time(
    traj: Trajectory,
    player: int = 0,
    num_item_types: int = 5,
) -> np.ndarray:
    """Return the mean item count of each type at each timestep.

    Parameters
    ----------
    traj :
        A trajectory whose ``player_inventory`` field is set. That
        field already holds a count for each item, so nothing has to
        map slots to items.
    player :
        Which player to read. It is ignored when the trajectory is not
        multi-player.
    num_item_types :
        How many leading item ids to keep. The default of 5 is far
        below the 34 the engine defines, so a default call drops most
        items.

    Returns
    -------
    numpy.ndarray
        Shape ``(T, num_item_types)``, the mean over episodes at each
        timestep. Column ``i`` is the item with ``ItemType`` value
        ``i``, and column 0 is ``EMPTY``, which stays at zero.

    Raises
    ------
    ValueError
        When the trajectory carries no ``player_inventory``.
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
    """Draw one line for each item type over the episode.

    Parameters
    ----------
    traj :
        A trajectory whose ``player_inventory`` field is set.
    player :
        Which player to read.
    num_item_types :
        How many leading item ids to draw. It also indexes
        ``item_labels`` and ``item_colors``, so all three must agree.
    item_labels :
        The name of each item, or ``None`` for
        :data:`DEFAULT_ITEM_LABELS`. Pass real names from ``ItemType``
        here. The default list is wrong past index 1, so a default
        call mislabels the legend.
    item_colors :
        The line color of each item, or ``None`` for
        :data:`DEFAULT_ITEM_COLORS`.
    exclude_empty :
        True to skip item 0, ``EMPTY``, which is always zero.
    ax :
        The axes to draw on, or ``None`` for a new figure.
    figsize :
        The size of that new figure in inches. It is ignored when
        ``ax`` is given.
    title :
        The figure title, or ``None`` for a default.

    Returns
    -------
    tuple
        ``(figure, axes)``. The figure is not written and not closed,
        so the caller owns it.

    Raises
    ------
    IndexError
        When ``num_item_types`` is greater than the length of
        ``item_labels`` or ``item_colors``. The two defaults hold five
        entries each.
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
    """Count how often a player stood on each map tile.

    Parameters
    ----------
    traj :
        A trajectory whose ``positions`` field is set. Positions are
        ordered ``(x, y)``, which is the opposite of the ``[y, x]``
        order used to index the returned grid.
    player :
        Which player to read. It is ignored when the trajectory is not
        multi-player.
    map_width, map_height :
        The map size in tiles. These must match the map the episode
        ran on. A value too small folds the outer tiles onto the
        border, because positions are clipped and not dropped.
    time_range :
        ``(start, end)`` to count only that window of timesteps, or
        ``None`` for the whole episode.

    Returns
    -------
    numpy.ndarray
        Shape ``(map_height, map_width)``, indexed ``[y, x]``. The
        values sum to 1.0, so each is the share of visits to that
        tile. A trajectory with no steps gives all zeros instead,
        because the division is skipped.

    Raises
    ------
    ValueError
        When the trajectory carries no ``positions``.
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
    """Draw the visit-frequency heatmap of one player.

    Parameters
    ----------
    traj, player, map_width, map_height, time_range :
        As in :func:`position_heatmap`.
    ax :
        The axes to draw on, or ``None`` for a new figure.
    figsize :
        The size of that new figure in inches. It is ignored when
        ``ax`` is given.
    cmap :
        Any matplotlib colormap name.
    title :
        The figure title, or ``None`` for a default.

    Returns
    -------
    tuple
        ``(figure, axes)``. The image uses ``origin="upper"``, so y
        grows downward and the picture matches the map.
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
    """Draw the path one player walked in one episode.

    The color of the path runs with time, dark at the start and bright
    at the end. A green circle marks the first position and a red
    square the last.

    Parameters
    ----------
    traj :
        A trajectory whose ``positions`` field is set.
    episode :
        Which episode to draw.
    player :
        Which player to draw.
    map_width, map_height :
        The map size in tiles, which sets the axis limits.
    ax :
        The axes to draw on, or ``None`` for a new figure.
    figsize :
        The size of that new figure in inches. It is ignored when
        ``ax`` is given.
    title :
        The figure title, or ``None`` for a default.
    cmap :
        The colormap used for the time gradient.

    Returns
    -------
    tuple
        ``(figure, axes)``. The y axis is inverted, so the picture
        matches the map rather than standard plot orientation.

    Raises
    ------
    ValueError
        When the trajectory carries no ``positions``.
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
    """Draw the total resource left on the map over time.

    The line is the mean over episodes, and the band around it is one
    standard deviation. The total sums every map tile, so it mixes
    every ore into one number and does not separate them.

    Parameters
    ----------
    traj :
        A trajectory whose ``block_resources`` field is set, shaped
        ``(B, T, H, W)``. Recordings do not carry this field by
        default. A rollout has to snapshot
        ``env_state.block_resources`` at each step.
    ax :
        The axes to draw on, or ``None`` for a new figure.
    figsize :
        The size of that new figure in inches. It is ignored when
        ``ax`` is given.
    title :
        The figure title, or ``None`` for a default.

    Returns
    -------
    tuple
        ``(figure, axes)``. The figure is not written and not closed,
        so the caller owns it.

    Raises
    ------
    ValueError
        When the trajectory carries no ``block_resources``.
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
    """Draw the step reward and the running total of one episode.

    The figure holds two stacked panels that share an x axis. The top
    panel is a bar for each step. The bottom is the cumulative sum.

    Parameters
    ----------
    traj :
        A trajectory whose ``rewards`` field is set.
    episode :
        Which episode to draw.
    figsize :
        The size of the figure in inches. This function always makes
        its own figure, because it needs two panels.
    title :
        The title over the top panel, or ``None`` for a default.

    Returns
    -------
    tuple
        ``(figure, axes)``, where ``axes`` holds the step-reward panel
        and the cumulative panel, in that order.

    Raises
    ------
    ValueError
        When the trajectory carries no ``rewards``.

    Notes
    -----
    A trajectory padded to a common episode length carries zeros in
    the pad steps. The cumulative line therefore runs flat at the end
    rather than stopping.
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
