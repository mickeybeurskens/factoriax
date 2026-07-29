"""Achievement and milestone analysis.

Tracks *when* key events happen during episodes — first craft, first
machine placement, resource milestones, etc.  Useful for understanding
learning progress and comparing policies.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .trajectory import Trajectory
from .utils import resolve_ax, resolve_player_actions


def achievement_timing(
    traj: Trajectory,
) -> np.ndarray:
    """Find the first timestep each achievement is unlocked.

    Parameters
    ----------
    traj : Trajectory
        Must have ``achievements`` field, shape ``(B, T, num_achievements)``.
    traj :
        Trajectory:
    traj: Trajectory :


    Returns
    -------

    """
    if traj.achievements is None:
        raise ValueError("achievement_timing requires achievements data")

    achievements = traj.achievements  # (B, T, A)
    B, T, A = achievements.shape

    timing = np.full((B, A), -1, dtype=np.int32)
    for b in range(B):
        for a in range(A):
            unlocked = np.where(achievements[b, :, a])[0]
            if len(unlocked) > 0:
                timing[b, a] = unlocked[0]

    return timing


def plot_achievement_timing(
    traj: Trajectory,
    achievement_labels: list[str] | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (10, 5),
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Box plot showing the distribution of achievement unlock times.

    Parameters
    ----------
    traj : Trajectory

    achievement_labels : list[str]

    ax, figsize, title :

    traj :
        Trajectory:
    achievement_labels :
        list[str] | None:  (Default value = None)
    ax :
        Axes | None:  (Default value = None)
    figsize :
        tuple[float:
    float] :
        (Default value = (10)
    5) :

    title :
        str | None:  (Default value = None)
    traj: Trajectory :

    achievement_labels: list[str] | None :
         (Default value = None)
    ax: Axes | None :
         (Default value = None)
    figsize: tuple[float :

    title: str | None :
         (Default value = None)

    Returns
    -------

    """
    timing = achievement_timing(traj)  # (B, A)
    B, A = timing.shape

    if achievement_labels is None:
        achievement_labels = [f"Achievement {i}" for i in range(A)]

    fig, ax = resolve_ax(ax, figsize)

    # Filter to only include episodes where achievement was unlocked
    data = []
    labels_used = []
    for a in range(A):
        times = timing[:, a]
        unlocked = times[times >= 0]
        if len(unlocked) > 0:
            data.append(unlocked)
            pct = len(unlocked) / B * 100
            labels_used.append(f"{achievement_labels[a]} ({pct:.0f}%)")

    if data:
        bp = ax.boxplot(
            data,
            orientation="horizontal",
            tick_labels=labels_used,
            patch_artist=True,
            showfliers=True,
            flierprops=dict(markersize=3, alpha=0.5),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("#4c72b0")
            patch.set_alpha(0.6)

    ax.set_xlabel("Timestep")
    ax.set_title(title or "Achievement unlock timing")
    fig.tight_layout()
    return fig, ax


def plot_achievement_progress(
    traj: Trajectory,
    achievement_labels: list[str] | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (12, 5),
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot cumulative achievement unlock rate over time.

    Shows what fraction of episodes have unlocked each achievement by
    each timestep — essentially a CDF of unlock times.

    Parameters
    ----------
    traj : Trajectory

    achievement_labels : list[str]

    traj :
        Trajectory:
    achievement_labels :
        list[str] | None:  (Default value = None)
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

    achievement_labels: list[str] | None :
         (Default value = None)
    ax: Axes | None :
         (Default value = None)
    figsize: tuple[float :

    title: str | None :
         (Default value = None)

    Returns
    -------

    """
    timing = achievement_timing(traj)  # (B, A)
    B, A = timing.shape
    T = traj.episode_length

    if achievement_labels is None:
        achievement_labels = [f"Achievement {i}" for i in range(A)]

    fig, ax = resolve_ax(ax, figsize)

    for a in range(A):
        times = timing[:, a]
        unlocked = times[times >= 0]
        if len(unlocked) == 0:
            continue
        sorted_times = np.sort(unlocked)
        cdf_y = np.arange(1, len(sorted_times) + 1) / B
        # Extend to full episode
        ax.step(
            np.concatenate([[0], sorted_times, [T]]),
            np.concatenate([[0], cdf_y, [cdf_y[-1]]]),
            where="post",
            label=achievement_labels[a],
            linewidth=1.5,
        )

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Fraction of episodes unlocked")
    ax.set_title(title or "Achievement progress")
    ax.set_xlim(0, T)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize="small")

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Event detection helpers
# ---------------------------------------------------------------------------


def first_action_timestep(
    traj: Trajectory,
    action_id: int,
    player: int | None = None,
) -> np.ndarray:
    """Find the first timestep a specific action is taken per episode.

    Parameters
    ----------
    traj : Trajectory

    action_id : int

    player : int

    traj :
        Trajectory:
    action_id :
        int:
    player :
        int | None:  (Default value = None)
    traj: Trajectory :

    action_id: int :

    player: int | None :
         (Default value = None)

    Returns
    -------

    """
    actions = resolve_player_actions(traj, player)
    B, T = actions.shape
    result = np.full(B, -1, dtype=np.int32)
    for b in range(B):
        matches = np.where(actions[b] == action_id)[0]
        if len(matches) > 0:
            result[b] = matches[0]
    return result


def plot_first_action_timing(
    traj: Trajectory,
    action_ids: list[int],
    action_labels: list[str] | None = None,
    player: int | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (10, 4),
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot the distribution of when each action is first taken.

    Useful for seeing when agents first mine, first craft, first place
    a machine, etc.

    Parameters
    ----------
    traj : Trajectory

    action_ids : list[int]
        Action IDs to analyze.
    action_labels : list[str]

    traj :
        Trajectory:
    action_ids :
        list[int]:
    action_labels :
        list[str] | None:  (Default value = None)
    player :
        int | None:  (Default value = None)
    ax :
        Axes | None:  (Default value = None)
    figsize :
        tuple[float:
    float] :
        (Default value = (10)
    4) :

    title :
        str | None:  (Default value = None)
    traj: Trajectory :

    action_ids: list[int] :

    action_labels: list[str] | None :
         (Default value = None)
    player: int | None :
         (Default value = None)
    ax: Axes | None :
         (Default value = None)
    figsize: tuple[float :

    title: str | None :
         (Default value = None)

    Returns
    -------

    """
    if action_labels is None:
        from .actions import DEFAULT_ACTION_LABELS

        action_labels = [DEFAULT_ACTION_LABELS[a] for a in action_ids]

    fig, ax = resolve_ax(ax, figsize)

    data = []
    labels_used = []
    for aid, label in zip(action_ids, action_labels):
        times = first_action_timestep(traj, aid, player)
        found = times[times >= 0]
        if len(found) > 0:
            data.append(found)
            pct = len(found) / len(times) * 100
            labels_used.append(f"First {label} ({pct:.0f}%)")

    if data:
        bp = ax.boxplot(
            data,
            orientation="horizontal",
            tick_labels=labels_used,
            patch_artist=True,
            showfliers=True,
            flierprops=dict(markersize=3, alpha=0.5),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("#ff7f0e")
            patch.set_alpha(0.6)

    ax.set_xlabel("Timestep")
    ax.set_title(title or "First action timing")
    fig.tight_layout()
    return fig, ax
