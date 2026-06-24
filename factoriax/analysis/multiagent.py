"""Multi-agent coordination analysis and visualization.

This module provides tools for understanding how multiple agents interact,
specialize, and coordinate within episodes.  It requires multi-player
trajectory data (actions with shape ``(B, T, P)``).

Key analyses
------------
* **Action comparison raster** — side-by-side raster plots per player
* **Role divergence** — how different are each player's action distributions?
* **Spatial overlap** — how often do players occupy the same or nearby tiles?
* **Joint action analysis** — what action pairs do players take simultaneously?
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from factoriax.engine.constants import NUM_ACTIONS

from .actions import (
    DEFAULT_ACTION_LABELS,
    action_raster,
)
from .trajectory import Trajectory
from .utils import resolve_ax


def _require_multi_player(traj: Trajectory) -> None:
    """

    Parameters
    ----------
    traj :
        Trajectory:
    traj: Trajectory :
        

    Returns
    -------

    """
    if not traj.is_multi_player:
        raise ValueError(
            "Multi-agent analysis requires multi-player trajectories "
            f"(actions.ndim == 3), got shape {traj.actions.shape}"
        )


# ---------------------------------------------------------------------------
# Comparative raster
# ---------------------------------------------------------------------------


def plot_comparative_raster(
    traj: Trajectory,
    num_actions: int = NUM_ACTIONS,
    action_labels: list[str] | None = None,
    colors: Sequence[str] | None = None,
    figsize_per_player: tuple[float, float] = (14, 4),
    player_labels: list[str] | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot action rasters for all players stacked vertically.
    
    This is the most direct way to see whether players are doing the same
    thing at the same time, taking turns, or operating independently.

    Parameters
    ----------
    traj : Trajectory
        Must be multi-player.
    player_labels : list[str]
        Labels for each player subplot.
    traj :
        Trajectory:
    num_actions :
        int:  (Default value = NUM_ACTIONS)
    action_labels :
        list[str] | None:  (Default value = None)
    colors :
        Sequence[str] | None:  (Default value = None)
    figsize_per_player :
        tuple[float:
    float] :
        (Default value = (14)
    4) :
        
    player_labels :
        list[str] | None:  (Default value = None)
    traj: Trajectory :
        
    num_actions: int :
         (Default value = NUM_ACTIONS)
    action_labels: list[str] | None :
         (Default value = None)
    colors: Sequence[str] | None :
         (Default value = None)
    figsize_per_player: tuple[float :
        
    player_labels: list[str] | None :
         (Default value = None)

    Returns
    -------

    """
    _require_multi_player(traj)
    P = traj.num_players

    fig, axes = plt.subplots(
        P,
        1,
        figsize=(figsize_per_player[0], figsize_per_player[1] * P),
        sharex=True,
    )
    if P == 1:
        axes = np.array([axes])

    for p, ax in enumerate(axes):
        label = player_labels[p] if player_labels else f"Player {p}"
        action_raster(
            traj,
            player=p,
            num_actions=num_actions,
            action_labels=action_labels,
            colors=colors,
            ax=ax,
            title=label,
        )

    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Role divergence
# ---------------------------------------------------------------------------


def role_divergence(
    traj: Trajectory,
    num_actions: int = NUM_ACTIONS,
    time_range: tuple[int, int] | None = None,
) -> np.ndarray:
    """Compute pairwise Jensen-Shannon divergence between players' action distributions.
    
    A high JSD means the two players are taking very different actions
    (role specialization).  A low JSD means they behave similarly.

    Parameters
    ----------
    traj : Trajectory
        Must be multi-player.
    num_actions : int
        
    time_range : tuple[int
        Restrict to a specific timestep window.
    traj :
        Trajectory:
    num_actions :
        int:  (Default value = NUM_ACTIONS)
    time_range :
        tuple[int:
    int] | None :
        (Default value = None)
    traj: Trajectory :
        
    num_actions: int :
         (Default value = NUM_ACTIONS)
    time_range: tuple[int :
        

    Returns
    -------

    """
    _require_multi_player(traj)
    P = traj.num_players

    # Compute per-player action distributions
    dists = np.zeros((P, num_actions))
    for p in range(P):
        acts = traj.player(p).actions  # (B, T)
        if time_range is not None:
            acts = acts[:, time_range[0] : time_range[1]]
        dists[p] = np.bincount(acts.ravel(), minlength=num_actions).astype(np.float64)
        dists[p] /= dists[p].sum()

    # Pairwise JSD
    def _kl(p: np.ndarray, q: np.ndarray) -> float:
        """

        Parameters
        ----------
        p :
            np.ndarray:
        q :
            np.ndarray:
        p: np.ndarray :
            
        q: np.ndarray :
            

        Returns
        -------

        """
        mask = (p > 0) & (q > 0)
        return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))

    jsd = np.zeros((P, P))
    for i in range(P):
        for j in range(i + 1, P):
            m = 0.5 * (dists[i] + dists[j])
            val = 0.5 * _kl(dists[i], m) + 0.5 * _kl(dists[j], m)
            jsd[i, j] = val
            jsd[j, i] = val

    return jsd


def plot_role_divergence(
    traj: Trajectory,
    num_actions: int = NUM_ACTIONS,
    phases: list[tuple[int, int]] | None = None,
    player_labels: list[str] | None = None,
    figsize: tuple[float, float] = (6, 5),
) -> tuple[Figure, Axes | np.ndarray]:
    """Plot role divergence heatmap, optionally across episode phases.

    Parameters
    ----------
    traj : Trajectory
        
    phases : list of (start
        If provided, creates subplots for each phase.
    player_labels : list[str]
        
    traj :
        Trajectory:
    num_actions :
        int:  (Default value = NUM_ACTIONS)
    phases :
        list[tuple[int:
    int]] | None :
        (Default value = None)
    player_labels :
        list[str] | None:  (Default value = None)
    figsize :
        tuple[float:
    float] :
        (Default value = (6)
    5) :
        
    traj: Trajectory :
        
    num_actions: int :
         (Default value = NUM_ACTIONS)
    phases: list[tuple[int :
        
    player_labels: list[str] | None :
         (Default value = None)
    figsize: tuple[float :
        

    Returns
    -------

    """
    _require_multi_player(traj)
    P = traj.num_players
    if player_labels is None:
        player_labels = [f"P{i}" for i in range(P)]

    if phases is None:
        fig, ax = plt.subplots(figsize=figsize)
        jsd = role_divergence(traj, num_actions)
        im = ax.imshow(jsd, cmap="YlOrRd", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, label="Jensen-Shannon divergence")
        ax.set_xticks(range(P))
        ax.set_yticks(range(P))
        ax.set_xticklabels(player_labels)
        ax.set_yticklabels(player_labels)
        ax.set_title("Role divergence (action distribution)")
        for i in range(P):
            for j in range(P):
                ax.text(j, i, f"{jsd[i, j]:.3f}", ha="center", va="center", fontsize=9)
        fig.tight_layout()
        return fig, ax
    else:
        n = len(phases)
        fig, axes = plt.subplots(1, n, figsize=(figsize[0] * n, figsize[1]))
        if n == 1:
            axes = np.array([axes])
        for idx, ((s, e), ax) in enumerate(zip(phases, axes)):
            jsd = role_divergence(traj, num_actions, time_range=(s, e))
            im = ax.imshow(jsd, cmap="YlOrRd", vmin=0, vmax=1)
            ax.set_xticks(range(P))
            ax.set_yticks(range(P))
            ax.set_xticklabels(player_labels)
            ax.set_yticklabels(player_labels)
            ax.set_title(f"Steps {s}–{e}")
            for i in range(P):
                for j in range(P):
                    ax.text(
                        j,
                        i,
                        f"{jsd[i, j]:.3f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                    )
        fig.colorbar(im, ax=axes[-1], label="JSD")
        fig.tight_layout()
        return fig, axes


# ---------------------------------------------------------------------------
# Spatial overlap
# ---------------------------------------------------------------------------


def spatial_overlap(
    traj: Trajectory,
    distance_threshold: int = 0,
) -> np.ndarray:
    """Compute per-timestep spatial overlap between players.

    Parameters
    ----------
    traj : Trajectory
        Must have ``positions`` field set.  Shape ``(B, T, P, 2)``.
    distance_threshold : int
        Maximum L1 (Manhattan) distance to count as "overlapping".
        0 means exact same tile.
    traj :
        Trajectory:
    distance_threshold :
        int:  (Default value = 0)
    traj: Trajectory :
        
    distance_threshold: int :
         (Default value = 0)

    Returns
    -------

    """
    _require_multi_player(traj)
    if traj.positions is None:
        raise ValueError("spatial_overlap requires positions data")

    positions = traj.positions  # (B, T, P, 2)
    B, T, P, _ = positions.shape

    overlap = np.zeros(T)
    for t in range(T):
        close_count = 0
        for b in range(B):
            # Check all pairs
            all_close = True
            for i in range(P):
                for j in range(i + 1, P):
                    dist = np.abs(positions[b, t, i] - positions[b, t, j]).sum()
                    if dist > distance_threshold:
                        all_close = False
                        break
                if not all_close:
                    break
            if all_close:
                close_count += 1
        overlap[t] = close_count / B

    return overlap


def plot_spatial_overlap(
    traj: Trajectory,
    distance_thresholds: list[int] = [0, 1, 3],
    ax: Axes | None = None,
    figsize: tuple[float, float] = (12, 4),
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot spatial overlap curves for multiple distance thresholds.

    Parameters
    ----------
    traj : Trajectory
        
    distance_thresholds : list[int]
        Multiple thresholds to plot.
    traj :
        Trajectory:
    distance_thresholds :
        list[int]:  (Default value = [0)
    1 :
        
    3] :
        
    ax :
        Axes | None:  (Default value = None)
    figsize :
        tuple[float:
    float] :
        (Default value = (12)
    4) :
        
    title :
        str | None:  (Default value = None)
    traj: Trajectory :
        
    distance_thresholds: list[int] :
         (Default value = [0)
    ax: Axes | None :
         (Default value = None)
    figsize: tuple[float :
        
    title: str | None :
         (Default value = None)

    Returns
    -------

    """
    fig, ax = resolve_ax(ax, figsize)

    for d in distance_thresholds:
        overlap = spatial_overlap(traj, d)
        label = "Same tile" if d == 0 else f"Within {d} tiles"
        ax.plot(overlap, label=label, linewidth=1.2)

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Fraction of episodes")
    ax.set_title(title or "Spatial overlap between players")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize="small")
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Joint action analysis
# ---------------------------------------------------------------------------


def joint_action_matrix(
    traj: Trajectory,
    player_a: int = 0,
    player_b: int = 1,
    num_actions: int = NUM_ACTIONS,
    normalize: bool = True,
    time_range: tuple[int, int] | None = None,
) -> np.ndarray:
    """Compute the joint action frequency matrix for two players.

    Parameters
    ----------
    traj : Trajectory
        
    player_a, player_b :
        Player indices.
    num_actions : int
        
    normalize : bool
        
    time_range : tuple[int
        
    traj :
        Trajectory:
    player_a :
        int:  (Default value = 0)
    player_b :
        int:  (Default value = 1)
    num_actions :
        int:  (Default value = NUM_ACTIONS)
    normalize :
        bool:  (Default value = True)
    time_range :
        tuple[int:
    int] | None :
        (Default value = None)
    traj: Trajectory :
        
    player_a: int :
         (Default value = 0)
    player_b: int :
         (Default value = 1)
    num_actions: int :
         (Default value = NUM_ACTIONS)
    normalize: bool :
         (Default value = True)
    time_range: tuple[int :
        

    Returns
    -------

    """
    _require_multi_player(traj)
    acts_a = traj.player(player_a).actions  # (B, T)
    acts_b = traj.player(player_b).actions

    if time_range is not None:
        acts_a = acts_a[:, time_range[0] : time_range[1]]
        acts_b = acts_b[:, time_range[0] : time_range[1]]

    mat = np.zeros((num_actions, num_actions), dtype=np.float64)
    for a, b in zip(acts_a.ravel(), acts_b.ravel()):
        mat[a, b] += 1

    if normalize:
        total = mat.sum()
        if total > 0:
            mat /= total

    return mat


def plot_joint_actions(
    traj: Trajectory,
    player_a: int = 0,
    player_b: int = 1,
    num_actions: int = NUM_ACTIONS,
    action_labels: list[str] | None = None,
    time_range: tuple[int, int] | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (8, 7),
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot the joint action frequency matrix as a heatmap.

    Parameters
    ----------
    traj, player_a, player_b, num_actions, action_labels, time_range :
        See :func:`joint_action_matrix`.
    traj :
        Trajectory:
    player_a :
        int:  (Default value = 0)
    player_b :
        int:  (Default value = 1)
    num_actions :
        int:  (Default value = NUM_ACTIONS)
    action_labels :
        list[str] | None:  (Default value = None)
    time_range :
        tuple[int:
    int] | None :
        (Default value = None)
    ax :
        Axes | None:  (Default value = None)
    figsize :
        tuple[float:
    float] :
        (Default value = (8)
    7) :
        
    title :
        str | None:  (Default value = None)
    traj: Trajectory :
        
    player_a: int :
         (Default value = 0)
    player_b: int :
         (Default value = 1)
    num_actions: int :
         (Default value = NUM_ACTIONS)
    action_labels: list[str] | None :
         (Default value = None)
    time_range: tuple[int :
        
    ax: Axes | None :
         (Default value = None)
    figsize: tuple[float :
        
    title: str | None :
         (Default value = None)

    Returns
    -------

    """
    mat = joint_action_matrix(traj, player_a, player_b, num_actions, True, time_range)

    if action_labels is None:
        action_labels = DEFAULT_ACTION_LABELS[:num_actions]

    fig, ax = resolve_ax(ax, figsize)

    im = ax.imshow(mat, cmap="Blues")
    fig.colorbar(im, ax=ax, label="Joint probability")

    ax.set_xticks(range(num_actions))
    ax.set_yticks(range(num_actions))
    ax.set_xticklabels(action_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(action_labels, fontsize=8)
    ax.set_xlabel(f"Player {player_b}")
    ax.set_ylabel(f"Player {player_a}")
    ax.set_title(title or f"Joint action distribution (P{player_a} × P{player_b})")

    fig.tight_layout()
    return fig, ax
