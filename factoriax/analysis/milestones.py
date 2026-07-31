"""Find when key events first happen in an episode, and plot them.

Two questions, each with a finder and its plots. When did each
achievement unlock, and when did the agent first take a given action?
The answers show how far a policy gets and how quickly, which is what
makes two policies comparable.

Every timestep here is a 0-based index along the time axis of the
trajectory. An event that never happened reports ``-1``, so a caller
masks with ``>= 0`` before it takes a mean.

The two box plots drop the episodes where the event never happened,
and put the share that did happen in the tick label. A box therefore
describes the episodes that reached the event, and not the whole
batch. A policy that reaches an achievement once, quickly, can look
better than one that reaches it every time, slowly.
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .trajectory import Trajectory
from .utils import resolve_ax, resolve_player_actions


def achievement_timing(
    traj: Trajectory,
) -> np.ndarray:
    """Find the first timestep at which each achievement unlocked.

    Parameters
    ----------
    traj :
        A trajectory whose ``achievements`` field holds a mask of
        shape ``(B, T, A)``. The mask is read as a boolean, so any
        non-zero value counts as unlocked.

    Returns
    -------
    numpy.ndarray
        int32 of shape ``(B, A)``. Entry ``(b, a)`` is the first
        timestep at which achievement ``a`` was set in episode ``b``,
        or ``-1`` when it never was.

    Raises
    ------
    ValueError
        When the trajectory carries no ``achievements`` field.

    Notes
    -----
    The scan is a Python loop over every episode and achievement, so
    the cost grows with ``B`` times ``A``.
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
    """Draw one box for each achievement, over its unlock timesteps.

    An achievement that no episode unlocked gets no box. For the rest,
    the box covers the episodes that did unlock it, and the tick label
    gives that share as a percentage. Read the box and the percentage
    together: a low percentage means the box describes few episodes.

    Parameters
    ----------
    traj :
        A trajectory with an ``achievements`` field.
    achievement_labels :
        The display name of each achievement, or ``None`` for
        ``"Achievement 0"`` and so on. The list must be at least as
        long as the achievement axis.
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
    """Draw the share of episodes that unlocked each achievement.

    Each line is a cumulative distribution over unlock time. At
    timestep ``t`` it gives the share of all episodes that had
    unlocked that achievement by ``t``.

    The share is over every episode, and not only over the episodes
    that unlocked the achievement. A line therefore flattens below 1.0
    when some episodes never reached it, and that height is the
    success rate. This differs from
    :func:`plot_achievement_timing`, which drops the episodes that
    never reached the achievement.

    Parameters
    ----------
    traj :
        A trajectory with an ``achievements`` field.
    achievement_labels :
        The display name of each achievement, or ``None`` for
        ``"Achievement 0"`` and so on.
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
    """Find the first timestep at which one action was taken.

    Parameters
    ----------
    traj :
        The recording to read. Actions of shape ``(B, T)``,
        ``(B, T, P)``, and ``(B, T, 1)`` all work.
    action_id :
        The action to look for.
    player :
        Which player to read, or ``None`` for player 0.

    Returns
    -------
    numpy.ndarray
        int32 of shape ``(B,)``. Entry ``b`` is the first timestep at
        which episode ``b`` took the action, or ``-1`` when it never
        did.

    Raises
    ------
    IndexError
        When ``player`` names a slot the player axis does not have.
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
    """Draw one box for each action, over its first-use timesteps.

    This answers when an agent first mines, first crafts, or first
    places a machine.

    An action that no episode took gets no box. For the rest, the box
    covers the episodes that did take it, and the tick label gives
    that share as a percentage. Read the box and the percentage
    together.

    Parameters
    ----------
    traj :
        The recording to read.
    action_ids :
        The actions to draw, one box each.
    action_labels :
        The display name of each action, or ``None`` to look the names
        up in ``DEFAULT_ACTION_LABELS``. The list must be as long as
        ``action_ids``.
    player :
        Which player to read, or ``None`` for player 0.
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
