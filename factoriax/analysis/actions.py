"""Find temporal patterns in what an agent did, and draw them.

Every function takes a
:class:`~factoriax.analysis.trajectory.Trajectory`. Each plot returns
its ``(figure, axes)`` without writing or closing anything, so the
caller can change the result before saving it.

Five views, each with a plot and most with a finder that returns the
numbers behind it:

* A raster draws one colored cell for each episode and timestep. Phase
  changes and loops show up as bands.
* A transition matrix counts how often each action follows each other
  action.
* An n-gram count finds the short sequences an agent repeats.
* An entropy curve measures how much the episodes of a batch agree at
  each timestep.
* A run length measures how long an action was repeated without a
  break.

:data:`DEFAULT_ACTION_LABELS` and :data:`DEFAULT_ACTION_COLORS` are
both built from the ``Action`` enum, so neither can fall behind it and
both cover every action. The colors carry no meaning, though. Pass
``colors`` where the scheme has to say something, such as cool tones
for movement.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from factoriax.engine.constants import NUM_ACTIONS, Action

from .trajectory import Trajectory
from .utils import distinct_colors, resolve_ax, resolve_player_actions

#: The name of each action, in ``Action`` value order. Built from the
#: enum, so it cannot drift from it and always covers every action.
DEFAULT_ACTION_LABELS: list[str] = [a.name for a in Action]

#: Fallback line and cell colors, in ``Action`` value order, one for
#: each action. Generated, so the list cannot fall behind the enum the
#: way a hand-written one does.
#:
#: The colors carry no meaning: two neighboring actions are not
#: related, and an action has no fixed color. Pass ``colors`` to any
#: plot that needs a chosen scheme, such as cool tones for movement.
DEFAULT_ACTION_COLORS: list[str] = distinct_colors(NUM_ACTIONS)


def _get_action_cmap(
    num_actions: int,
    colors: Sequence[str] | None = None,
) -> mcolors.ListedColormap:
    """Build a discrete colormap with one color for each action.

    Parameters
    ----------
    num_actions :
        How many colors the map needs.
    colors :
        One color for each action, or ``None`` for
        :data:`DEFAULT_ACTION_COLORS`.

    Returns
    -------
    matplotlib.colors.ListedColormap
        A map with ``min(num_actions, len(colors))`` entries. A
        ``colors`` shorter than ``num_actions`` gives a short map, and
        matplotlib then hands every index past its end the same last
        color. The default list covers every action, so this happens
        only with a short list from the caller.

    Notes
    -----
    The default colors are generated and carry no meaning. Pass
    ``colors`` where the scheme has to say something, such as cool
    tones for movement.
    """
    if colors is None:
        colors = DEFAULT_ACTION_COLORS[:num_actions]
    return mcolors.ListedColormap(colors[:num_actions])


# ---------------------------------------------------------------------------
# Raster plot
# ---------------------------------------------------------------------------


def action_raster(
    traj: Trajectory,
    player: int | None = None,
    num_actions: int = NUM_ACTIONS,
    action_labels: list[str] | None = None,
    colors: Sequence[str] | None = None,
    episode_labels: list[str] | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (14, 6),
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Draw the action of every episode and timestep as a colored cell.

    One row is an episode, one column a timestep, and the color is the
    action. This is the first view to reach for when looking for
    temporal structure: phase changes, loops, and repeated
    sub-behaviors all show up as bands.

    Parameters
    ----------
    traj :
        The recording to read. Actions of shape ``(B, T)``,
        ``(B, T, P)``, and ``(B, T, 1)`` all work.
    player :
        Which player to read, or ``None`` for player 0.
    num_actions :
        How many actions the color scale covers.
    action_labels :
        The name of each action, or ``None`` for
        :data:`DEFAULT_ACTION_LABELS`, which is built from the
        ``Action`` enum and cannot drift from it.
    colors :
        One color for each action, or ``None`` for the defaults.
    episode_labels :
        The name of each row, or ``None`` to number them. The list
        must hold one entry for each episode.
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

    Notes
    -----
    The default colors are generated and carry no meaning. Pass
    ``colors`` where the scheme has to say something, such as cool
    tones for movement.
    """
    actions = resolve_player_actions(traj, player)  # (B, T)
    B, T = actions.shape

    if action_labels is None:
        action_labels = DEFAULT_ACTION_LABELS[:num_actions]

    cmap = _get_action_cmap(num_actions, colors)
    norm = mcolors.BoundaryNorm(np.arange(-0.5, num_actions), num_actions)

    fig, ax = resolve_ax(ax, figsize)

    ax.imshow(
        actions,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        origin="upper",
    )

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Episode")
    if title:
        ax.set_title(title)
    elif traj.is_multi_player and player is not None:
        ax.set_title(f"Action raster — Player {player}")

    if episode_labels is not None:
        ax.set_yticks(range(B))
        ax.set_yticklabels(episode_labels)

    # Legend
    patches = [
        Patch(facecolor=cmap(i), label=action_labels[i]) for i in range(num_actions)
    ]
    ax.legend(
        handles=patches,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        fontsize="small",
        ncol=1,
        frameon=True,
    )

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Transition matrix
# ---------------------------------------------------------------------------


def transition_matrix(
    traj: Trajectory,
    player: int | None = None,
    num_actions: int = NUM_ACTIONS,
    normalize: bool = True,
    time_range: tuple[int, int] | None = None,
) -> np.ndarray:
    """Count how often each action follows each other action.

    Parameters
    ----------
    traj :
        The recording to read. Actions of shape ``(B, T)``,
        ``(B, T, P)``, and ``(B, T, 1)`` all work.
    player :
        Which player to read, or ``None`` for player 0.
    num_actions :
        The side length of the matrix. An action id at or above this
        value raises ``IndexError``.
    normalize :
        True to divide each row by its total, giving the probability
        of the next action. False to return raw counts.
    time_range :
        ``(start, end)`` to count only that window of timesteps, or
        ``None`` for the whole episode.

    Returns
    -------
    numpy.ndarray
        Shape ``(num_actions, num_actions)``, float64. Entry
        ``(i, j)`` counts the steps where action ``i`` was followed by
        action ``j``. Pairs are counted inside an episode only, so the
        last step of one episode is not paired with the first of the
        next.

    Notes
    -----
    With ``normalize=True``, a row for an action that never occurred
    is divided by 1 rather than by 0. Such a row is therefore all
    zeros and not ``NaN``, so it plots as an empty row and does not
    sum to 1.
    """
    actions = resolve_player_actions(traj, player)  # (B, T)
    if time_range is not None:
        actions = actions[:, time_range[0] : time_range[1]]

    mat = np.zeros((num_actions, num_actions), dtype=np.float64)
    for ep in actions:
        for t in range(len(ep) - 1):
            mat[ep[t], ep[t + 1]] += 1

    if normalize:
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        mat = mat / row_sums

    return mat


def plot_transition_matrix(
    traj: Trajectory,
    player: int | None = None,
    num_actions: int = NUM_ACTIONS,
    action_labels: list[str] | None = None,
    time_range: tuple[int, int] | None = None,
    normalize: bool = True,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (8, 7),
    cmap: str = "Blues",
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Draw the transition matrix as a heatmap.

    Parameters
    ----------
    traj :
        The recording to read. Actions of shape ``(B, T)``,
        ``(B, T, P)``, and ``(B, T, 1)`` all work.
    player :
        Which player to read, or ``None`` for player 0.
    num_actions :
        The side length of the matrix.
    action_labels :
        The name of each action, or ``None`` for the defaults. The
        names label both axes.
    time_range :
        ``(start, end)`` to count only that window, or ``None``.
    normalize :
        True for probabilities, False for raw counts. This also sets
        the colorbar label.
    ax :
        The axes to draw on, or ``None`` for a new figure.
    figsize :
        The size of that new figure in inches. It is ignored when
        ``ax`` is given.
    title :
        The figure title, or ``None`` for a default.
    cmap :
        Any matplotlib colormap name.

    Returns
    -------
    tuple
        ``(figure, axes)``. The figure is not written and not closed,
        so the caller owns it.
    """
    mat = transition_matrix(traj, player, num_actions, normalize, time_range)

    if action_labels is None:
        action_labels = DEFAULT_ACTION_LABELS[:num_actions]

    fig, ax = resolve_ax(ax, figsize)

    im = ax.imshow(mat, cmap=cmap, vmin=0)
    fig.colorbar(im, ax=ax, label="Probability" if normalize else "Count")

    ax.set_xticks(range(num_actions))
    ax.set_yticks(range(num_actions))
    ax.set_xticklabels(action_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(action_labels, fontsize=8)
    ax.set_xlabel("Next action")
    ax.set_ylabel("Current action")

    # Annotate cells
    fmt = ".2f" if normalize else "d"
    for i in range(num_actions):
        for j in range(num_actions):
            val = mat[i, j]
            if val > 0.005 or (not normalize and val > 0):
                color = "white" if val > (mat.max() * 0.6) else "black"
                ax.text(
                    j,
                    i,
                    f"{val:{fmt}}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=color,
                )

    if title:
        ax.set_title(title)
    elif time_range:
        ax.set_title(f"Transitions (steps {time_range[0]}–{time_range[1]})")

    fig.tight_layout()
    return fig, ax


def plot_phase_transitions(
    traj: Trajectory,
    phases: list[tuple[int, int]],
    player: int | None = None,
    num_actions: int = NUM_ACTIONS,
    action_labels: list[str] | None = None,
    figsize_per_phase: tuple[float, float] = (5, 4.5),
    cmap: str = "Blues",
    phase_labels: list[str] | None = None,
) -> tuple[Figure, np.ndarray]:
    """Draw one transition matrix for each phase, side by side.

    Use this to see how the behavior of a policy changes over an
    episode. An early phase and a late phase each get their own
    matrix, on a shared color scale.

    Parameters
    ----------
    traj :
        The recording to read. Actions of shape ``(B, T)``,
        ``(B, T, P)``, and ``(B, T, 1)`` all work.
    player :
        Which player to read, or ``None`` for player 0.
    phases :
        One ``(start, end)`` timestep window for each panel. The
        windows may overlap and need not cover the whole episode.
    num_actions :
        The side length of each matrix.
    action_labels :
        The name of each action, or ``None`` for the defaults.
    figsize_per_phase :
        The size of one panel in inches. The figure width grows with
        the number of phases.
    cmap :
        Any matplotlib colormap name.
    phase_labels :
        The title of each panel, or ``None`` to build one from the
        window. The list must be as long as ``phases``.

    Returns
    -------
    tuple
        ``(figure, axes)``, where ``axes`` holds one entry for each
        phase. The figure is not written and not closed.
    """
    n = len(phases)
    fig, axes = plt.subplots(
        1, n, figsize=(figsize_per_phase[0] * n, figsize_per_phase[1])
    )
    if n == 1:
        axes = np.array([axes])

    if action_labels is None:
        action_labels = DEFAULT_ACTION_LABELS[:num_actions]

    for i, (phase, ax) in enumerate(zip(phases, axes)):
        label = phase_labels[i] if phase_labels else f"Steps {phase[0]}–{phase[1]}"
        plot_transition_matrix(
            traj,
            player,
            num_actions,
            action_labels,
            time_range=phase,
            ax=ax,
            cmap=cmap,
            title=label,
        )

    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# N-gram analysis
# ---------------------------------------------------------------------------


def action_ngrams(
    traj: Trajectory,
    n: int = 2,
    player: int | None = None,
    top_k: int = 20,
    action_labels: list[str] | None = None,
) -> list[tuple[Any, int]]:
    """Count the most common runs of ``n`` actions in a row.

    Parameters
    ----------
    traj :
        The recording to read. Actions of shape ``(B, T)``,
        ``(B, T, P)``, and ``(B, T, 1)`` all work.
    player :
        Which player to read, or ``None`` for player 0.
    n :
        How many actions make one gram. An ``n`` above the episode
        length gives an empty result.
    top_k :
        How many of the most common grams to return.
    action_labels :
        The name of each action, or ``None`` to key the result by
        action id instead of by name.

    Returns
    -------
    list
        ``(gram, count)`` pairs, most common first. A gram is a tuple
        of ``n`` action ids, or of ``n`` names when ``action_labels``
        is given. Grams are counted inside an episode only, so none
        spans two episodes.
    """
    actions = resolve_player_actions(traj, player)  # (B, T)
    counter: Counter[tuple[int, ...]] = Counter()

    for ep in actions:
        for t in range(len(ep) - n + 1):
            gram = tuple(ep[t : t + n].tolist())
            counter[gram] += 1

    if action_labels is not None:
        labeled: Counter[tuple[str, ...]] = Counter()
        for gram, count in counter.items():
            labeled[tuple(action_labels[a] for a in gram)] = count
        return labeled.most_common(top_k)

    return counter.most_common(top_k)


def _blend_ngram_color(
    action_indices: tuple[int, ...],
    colors: Sequence[str],
) -> str:
    """Average the colors of the actions in one gram.

    A gram has no color of its own, so its bar takes the mean of the
    colors of its actions. Two grams that share most of their actions
    therefore look alike.

    Parameters
    ----------
    action_indices :
        The action ids in the gram.
    colors :
        One color for each action. An index past its end wraps around
        with a modulo, so a short list gives a color rather than
        raising.

    Returns
    -------
    str
        The mean color as a ``#rrggbb`` string.
    """
    rgbs = [mcolors.to_rgb(colors[min(i, len(colors) - 1)]) for i in action_indices]
    avg = tuple(sum(c) / len(c) for c in zip(*rgbs))
    return mcolors.to_hex(avg)


def plot_ngram_sweep(
    traj: Trajectory,
    n_range: tuple[int, int] = (2, 10),
    player: int | None = None,
    top_k: int = 3,
    action_labels: list[str] | None = None,
    colors: Sequence[str] | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> tuple[Figure, np.ndarray]:
    """Draw the top grams for each length in a range, one row each.

    Parameters
    ----------
    traj :
        The recording to read. Actions of shape ``(B, T)``,
        ``(B, T, P)``, and ``(B, T, 1)`` all work.
    player :
        Which player to read, or ``None`` for player 0.
    n_range :
        ``(lowest, highest)`` gram length, both included. One row of
        panels is drawn for each length.
    top_k :
        How many grams to show for each length.
    action_labels :
        The name of each action, or ``None`` for the defaults.
    colors :
        One color for each action, or ``None`` for the defaults. Each
        bar takes the mean color of its gram.
    figsize :
        The size of the figure in inches, or ``None`` to size it from
        the row count.
    title :
        The figure title, or ``None`` for a default.

    Returns
    -------
    tuple
        ``(figure, axes)``, where ``axes`` holds one entry for each
        gram length. The figure is not written and not closed.

    Notes
    -----
    The default colors are generated and carry no meaning. Pass
    ``colors`` where the scheme has to say something, such as cool
    tones for movement.
    """
    if action_labels is None:
        action_labels = DEFAULT_ACTION_LABELS
    if colors is None:
        colors = DEFAULT_ACTION_COLORS
    n_min, n_max = n_range
    n_values = list(range(n_min, n_max + 1))
    num_rows = len(n_values)

    from matplotlib.patches import Patch, Rectangle

    # Compute layout: each row is a text table, no axes needed.
    # n=N | [colored boxes] | ACTION ACTION ACTION (xN count)
    sq_size = 0.015  # box size in figure fraction
    sq_gap = 0.002
    row_h = max(0.06, 0.8 / (num_rows * top_k))
    legend_h = 0.08
    header_h = 0.04

    total_h = legend_h + header_h + num_rows * top_k * row_h + 0.04
    if figsize is None:
        figsize = (12, max(3.0, total_h))

    fig = plt.figure(figsize=figsize)

    # Action color legend at the top.
    num_actions = len(action_labels)
    legend_patches = [
        Patch(facecolor=colors[i], label=action_labels[i])
        for i in range(min(num_actions, len(colors)))
    ]
    fig.legend(
        handles=legend_patches,
        loc="upper center",
        ncol=min(num_actions, 8),
        fontsize=7,
        frameon=True,
        bbox_to_anchor=(0.5, 0.99),
    )

    if title:
        fig.text(0.5, 0.995, title, ha="center", fontsize=11)

    # Layout columns (in figure fraction).
    col_n = 0.04  # "n=N"
    col_boxes = 0.08  # start of colored boxes
    col_text = col_boxes + (n_max + 1) * (sq_size + sq_gap) + 0.01
    col_count = 0.92  # right-aligned count

    y_cursor = 1.0 - legend_h - header_h

    for row, n in enumerate(n_values):
        grams_int = action_ngrams(traj, n, player, top_k)
        if not grams_int:
            y_cursor -= top_k * row_h
            continue

        for idx, (gram, count) in enumerate(grams_int):
            y = y_cursor - idx * row_h

            # n=N label (only on first entry per group).
            if idx == 0:
                fig.text(
                    col_n,
                    y,
                    f"n={n}",
                    fontsize=9,
                    fontweight="bold",
                    va="center",
                    ha="left",
                )

            # Colored boxes.
            for j, a in enumerate(gram):
                a_idx = int(a)
                bx = col_boxes + j * (sq_size + sq_gap)
                fig.add_artist(
                    Rectangle(
                        (bx, y - sq_size / 2),
                        sq_size,
                        sq_size,
                        facecolor=colors[min(a_idx, len(colors) - 1)],
                        edgecolor="white",
                        linewidth=0.3,
                        transform=fig.transFigure,
                        clip_on=False,
                    )
                )

            # Action name sequence.
            label = " ".join(action_labels[int(a)] for a in gram)
            fig.text(
                col_text,
                y,
                label,
                fontsize=7,
                family="monospace",
                va="center",
                ha="left",
            )

            # Count on the right.
            fig.text(
                col_count,
                y,
                f"x{count}",
                fontsize=7,
                va="center",
                ha="right",
                color="gray",
            )

        y_cursor -= top_k * row_h + 0.01

    # Return empty axes array for API consistency.
    return fig, np.array([])


def plot_ngrams(
    traj: Trajectory,
    n: int = 2,
    player: int | None = None,
    top_k: int = 15,
    action_labels: list[str] | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (10, 5),
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Draw the most common grams of one length as a bar chart.

    Parameters
    ----------
    traj :
        The recording to read. Actions of shape ``(B, T)``,
        ``(B, T, P)``, and ``(B, T, 1)`` all work.
    player :
        Which player to read, or ``None`` for player 0.
    n :
        How many actions make one gram.
    top_k :
        How many of the most common grams to draw.
    action_labels :
        The name of each action, or ``None`` for the defaults.
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
        action_labels = DEFAULT_ACTION_LABELS
    grams = action_ngrams(traj, n, player, top_k, action_labels)

    labels = [" → ".join(g) for g, _ in grams]
    counts = [c for _, c in grams]

    fig, ax = resolve_ax(ax, figsize)

    ax.barh(range(len(labels)), counts, color="#4c72b0")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title(title or f"Top {top_k} action {n}-grams")

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Entropy curve
# ---------------------------------------------------------------------------


def action_entropy(
    traj: Trajectory,
    player: int | None = None,
    num_actions: int = NUM_ACTIONS,
    window: int = 1,
) -> np.ndarray:
    """Measure how much the chosen action varies across episodes.

    At each timestep the function builds the distribution of actions
    over the episodes of the batch, then takes its Shannon entropy in
    bits. A value of 0 means every episode took the same action at
    that step. The largest possible value is ``log2(num_actions)``.

    This measures agreement between episodes at one timestep. It does
    not measure how varied one episode is over time.

    Parameters
    ----------
    traj :
        The recording to read. Actions of shape ``(B, T)``,
        ``(B, T, P)``, and ``(B, T, 1)`` all work.
    player :
        Which player to read, or ``None`` for player 0.
    num_actions :
        How many action ids the distribution covers.
    window :
        The width of a rolling mean over time. A ``window`` of 1
        applies no smoothing.

    Returns
    -------
    numpy.ndarray
        Shape ``(T,)``, in bits.

    Notes
    -----
    The smoothing uses ``mode="same"``, so the first and last
    ``window // 2`` values average over fewer real timesteps and pull
    toward zero. Read the two ends with that in mind.
    """
    actions = resolve_player_actions(traj, player)  # (B, T)
    B, T = actions.shape

    entropy = np.zeros(T)
    for t in range(T):
        counts = np.bincount(actions[:, t], minlength=num_actions).astype(np.float64)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy[t] = -np.sum(probs * np.log2(probs))

    if window > 1:
        kernel = np.ones(window) / window
        entropy = np.convolve(entropy, kernel, mode="same")

    return entropy


def plot_entropy(
    traj: Trajectory,
    player: int | None = None,
    num_actions: int = NUM_ACTIONS,
    window: int = 5,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (12, 4),
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Draw the action entropy over the course of an episode.

    Parameters
    ----------
    traj :
        The recording to read. Actions of shape ``(B, T)``,
        ``(B, T, P)``, and ``(B, T, 1)`` all work.
    player :
        Which player to read, or ``None`` for player 0.
    num_actions :
        How many action ids the distribution covers.
    window :
        The width of the rolling mean, as in :func:`action_entropy`.
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
    ent = action_entropy(traj, player, num_actions, window)

    fig, ax = resolve_ax(ax, figsize)

    ax.plot(ent, color="#4c72b0", linewidth=1.2)
    ax.fill_between(range(len(ent)), ent, alpha=0.15, color="#4c72b0")

    max_ent = np.log2(num_actions)
    ax.axhline(max_ent, ls="--", color="gray", alpha=0.5, label=f"Max ({max_ent:.2f})")

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Entropy (bits)")
    ax.set_title(title or "Action entropy over episode")
    ax.set_xlim(0, len(ent) - 1)
    ax.set_ylim(0, max_ent * 1.05)
    ax.legend(fontsize="small")

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Run-length analysis
# ---------------------------------------------------------------------------


def run_lengths(
    traj: Trajectory,
    player: int | None = None,
) -> dict[int, list[int]]:
    """Measure how long each action was repeated without a break.

    Parameters
    ----------
    traj :
        The recording to read. Actions of shape ``(B, T)``,
        ``(B, T, P)``, and ``(B, T, 1)`` all work.
    player :
        Which player to read, or ``None`` for player 0.

    Returns
    -------
    dict
        ``{action_id: [length, ...]}``. A length counts the steps in
        one unbroken run of that action. Runs are collected inside
        an episode only. The run at the end of an episode is closed at
        the last step, and not joined to the next episode. An action
        that never occurred is absent from the dict.
    """
    actions = resolve_player_actions(traj, player)
    result: dict[int, list[int]] = {}

    for ep in actions:
        if len(ep) == 0:
            continue
        current = ep[0]
        length = 1
        for t in range(1, len(ep)):
            if ep[t] == current:
                length += 1
            else:
                result.setdefault(int(current), []).append(length)
                current = ep[t]
                length = 1
        result.setdefault(int(current), []).append(length)

    return result


def plot_run_lengths(
    traj: Trajectory,
    player: int | None = None,
    action_labels: list[str] | None = None,
    num_actions: int = NUM_ACTIONS,
    colors: Sequence[str] | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (10, 5),
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Draw one box for each action, over its run lengths.

    An action that never occurred gets no box, so the chart shows only
    the actions the policy used.

    Parameters
    ----------
    traj :
        The recording to read. Actions of shape ``(B, T)``,
        ``(B, T, P)``, and ``(B, T, 1)`` all work.
    player :
        Which player to read, or ``None`` for player 0.
    action_labels :
        The name of each action, or ``None`` for the defaults.
    num_actions :
        How many action ids to consider.
    colors :
        One color for each action, or ``None`` for the defaults.
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

    Notes
    -----
    The default colors are generated and carry no meaning. Pass
    ``colors`` where the scheme has to say something, such as cool
    tones for movement.
    """
    if action_labels is None:
        action_labels = DEFAULT_ACTION_LABELS[:num_actions]
    if colors is None:
        colors = DEFAULT_ACTION_COLORS[:num_actions]

    runs = run_lengths(traj, player)

    fig, ax = resolve_ax(ax, figsize)

    data = [runs.get(i, []) for i in range(num_actions)]
    present = [(i, d) for i, d in enumerate(data) if d]

    if present:
        bp = ax.boxplot(
            [d for _, d in present],
            tick_labels=[action_labels[i] for i, _ in present],
            orientation="vertical",
            patch_artist=True,
            showfliers=True,
            flierprops=dict(markersize=2, alpha=0.4),
        )
        for patch, (i, _) in zip(bp["boxes"], present):
            patch.set_facecolor(colors[i])
            patch.set_alpha(0.7)

    ax.set_ylabel("Run length")
    ax.set_title(title or "Run-length distributions by action")
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Action distribution over time (stacked area)
# ---------------------------------------------------------------------------


def plot_action_distribution(
    traj: Trajectory,
    player: int | None = None,
    num_actions: int = NUM_ACTIONS,
    action_labels: list[str] | None = None,
    colors: Sequence[str] | None = None,
    window: int = 10,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (14, 5),
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Draw how the mix of actions changes over an episode.

    The chart is a stacked area, so each band is the share of one
    action at that timestep and the bands sum to 1.

    Parameters
    ----------
    traj :
        The recording to read. Actions of shape ``(B, T)``,
        ``(B, T, P)``, and ``(B, T, 1)`` all work.
    player :
        Which player to read, or ``None`` for player 0.
    num_actions :
        How many action ids to draw as bands.
    action_labels :
        The name of each action, or ``None`` for the defaults.
    colors :
        One color for each action, or ``None`` for the defaults.
    window :
        The width of a rolling mean over time, which smooths the
        bands.
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

    Notes
    -----
    The default colors are generated and carry no meaning. Pass
    ``colors`` where the scheme has to say something, such as cool
    tones for movement.
    """
    actions = resolve_player_actions(traj, player)  # (B, T)
    B, T = actions.shape

    if action_labels is None:
        action_labels = DEFAULT_ACTION_LABELS[:num_actions]
    if colors is None:
        colors = DEFAULT_ACTION_COLORS[:num_actions]

    # Compute per-timestep distribution
    dist = np.zeros((T, num_actions))
    for t in range(T):
        dist[t] = np.bincount(actions[:, t], minlength=num_actions)
    dist = dist / dist.sum(axis=1, keepdims=True)

    # Smooth (only when the episode is longer than the kernel)
    effective_window = min(window, T)
    if effective_window > 1:
        kernel = np.ones(effective_window) / effective_window
        for a in range(num_actions):
            dist[:, a] = np.convolve(dist[:, a], kernel, mode="same")
        # Renormalize after smoothing
        dist = dist / dist.sum(axis=1, keepdims=True)

    fig, ax = resolve_ax(ax, figsize)

    ax.stackplot(
        range(T),
        *[dist[:, a] for a in range(num_actions)],
        labels=action_labels,
        colors=colors,
        alpha=0.8,
    )
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Action proportion")
    ax.set_title(title or "Action distribution over episode")
    ax.set_xlim(0, T - 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize="small")

    fig.tight_layout()
    return fig, ax
