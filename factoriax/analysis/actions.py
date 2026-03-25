"""Action sequence analysis and visualization.

This module provides tools for understanding temporal patterns in agent
action sequences.  All plotting functions return ``(fig, ax)`` tuples so
researchers can further customize the output.

Key analyses
------------
* **Raster plots** — color-coded timelines showing raw action sequences
* **Transition matrices** — action-to-action transition probabilities
* **N-gram analysis** — frequency of short action subsequences
* **Entropy curves** — action diversity over time
* **Run-length analysis** — distribution of consecutive repeated actions

All functions accept a :class:`~factoriax.analysis.trajectory.Trajectory`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from .trajectory import Trajectory

# Default action labels matching factoriax Action enum
DEFAULT_ACTION_LABELS = [
    "NOOP",
    "LEFT",
    "RIGHT",
    "UP",
    "DOWN",
    "MINE",
    "CRAFT",
    "PLACE",
    "NEXT_SLOT",
    "PREV_SLOT",
    "NEXT_RECIPE",
    "PREV_RECIPE",
    "PICKUP",
    "DEPOSIT",
    "WITHDRAW",
]

# Qualitative colormap designed for 15 distinguishable actions.
# Movement = cool tones, interaction = warm tones, UI = grays/purples.
DEFAULT_ACTION_COLORS = [
    "#bdbdbd",  # NOOP       - gray
    "#1f77b4",  # LEFT       - blue
    "#aec7e8",  # RIGHT      - light blue
    "#2ca02c",  # UP         - green
    "#98df8a",  # DOWN       - light green
    "#d62728",  # MINE       - red
    "#ff7f0e",  # CRAFT      - orange
    "#ffbb78",  # PLACE      - light orange
    "#9467bd",  # NEXT_SLOT  - purple
    "#c5b0d5",  # PREV_SLOT  - light purple
    "#8c564b",  # NEXT_REC   - brown
    "#c49c94",  # PREV_REC   - light brown
    "#e377c2",  # PICKUP     - pink
    "#17becf",  # DEPOSIT    - cyan
    "#bcbd22",  # WITHDRAW   - olive
]


def _get_action_cmap(
    num_actions: int,
    colors: Sequence[str] | None = None,
) -> mcolors.ListedColormap:
    """Build a discrete colormap for actions."""
    if colors is None:
        colors = DEFAULT_ACTION_COLORS[:num_actions]
    return mcolors.ListedColormap(colors[:num_actions])


def _resolve_player_actions(traj: Trajectory, player: int | None) -> np.ndarray:
    """Extract a (B, T) action array, selecting a player if multi-player."""
    if traj.is_multi_player:
        if player is None:
            player = 0
        return traj.player(player).actions
    return traj.actions


# ---------------------------------------------------------------------------
# Raster plot
# ---------------------------------------------------------------------------


def action_raster(
    traj: Trajectory,
    player: int | None = None,
    num_actions: int = 12,
    action_labels: list[str] | None = None,
    colors: Sequence[str] | None = None,
    episode_labels: list[str] | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (14, 6),
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot a raster of action sequences across episodes.

    Each row is an episode, each column is a timestep, and color encodes the
    action taken.  This is the single most informative view for spotting
    temporal structure: phase transitions, loops, stereotyped sub-behaviors.

    Parameters
    ----------
    traj : Trajectory
        Trajectory data.
    player : int, optional
        Player index for multi-player trajectories.  Defaults to 0.
    num_actions : int
        Total number of distinct actions.
    action_labels : list[str], optional
        Human-readable labels per action.
    colors : sequence of str, optional
        Hex colors per action.
    episode_labels : list[str], optional
        Labels for each episode (y-axis).
    ax : Axes, optional
        Existing matplotlib axes.  A new figure is created if *None*.
    figsize : tuple
        Figure size if creating a new figure.
    title : str, optional
        Plot title.

    Returns
    -------
    fig, ax : Figure, Axes
    """
    actions = _resolve_player_actions(traj, player)  # (B, T)
    B, T = actions.shape

    if action_labels is None:
        action_labels = DEFAULT_ACTION_LABELS[:num_actions]

    cmap = _get_action_cmap(num_actions, colors)
    norm = mcolors.BoundaryNorm(np.arange(-0.5, num_actions), num_actions)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

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
    num_actions: int = 12,
    normalize: bool = True,
    time_range: tuple[int, int] | None = None,
) -> np.ndarray:
    """Compute the action-to-action transition matrix.

    Parameters
    ----------
    traj : Trajectory
    player : int, optional
    num_actions : int
    normalize : bool
        If *True*, rows sum to 1 (transition probabilities).
        If *False*, raw counts.
    time_range : tuple[int, int], optional
        Restrict to a specific timestep window ``(start, end)``.

    Returns
    -------
    T : np.ndarray
        Shape ``(num_actions, num_actions)``.  ``T[i, j]`` is the
        probability (or count) of action *j* following action *i*.
    """
    actions = _resolve_player_actions(traj, player)  # (B, T)
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
    num_actions: int = 12,
    action_labels: list[str] | None = None,
    time_range: tuple[int, int] | None = None,
    normalize: bool = True,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (8, 7),
    cmap: str = "Blues",
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot the action transition matrix as a heatmap.

    Parameters
    ----------
    traj, player, num_actions, action_labels, time_range, normalize
        See :func:`transition_matrix`.
    ax, figsize, cmap, title
        Plotting options.

    Returns
    -------
    fig, ax : Figure, Axes
    """
    mat = transition_matrix(traj, player, num_actions, normalize, time_range)

    if action_labels is None:
        action_labels = DEFAULT_ACTION_LABELS[:num_actions]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

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
    num_actions: int = 12,
    action_labels: list[str] | None = None,
    figsize_per_phase: tuple[float, float] = (5, 4.5),
    cmap: str = "Blues",
    phase_labels: list[str] | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot transition matrices for multiple episode phases side-by-side.

    This reveals how the policy's sequential behavior changes over the
    course of an episode (e.g., exploring early, exploiting late).

    Parameters
    ----------
    traj : Trajectory
    phases : list of (start, end) tuples
        Timestep ranges for each phase.
    phase_labels : list[str], optional
        Label for each phase subplot.

    Returns
    -------
    fig, axes : Figure, ndarray of Axes
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
) -> list[tuple[tuple[str, ...], int]]:
    """Extract the most common action n-grams.

    Parameters
    ----------
    traj : Trajectory
    n : int
        Length of the subsequences (2 = bigrams, 3 = trigrams, etc.).
    player : int, optional
    top_k : int
        Number of top n-grams to return.
    action_labels : list[str], optional
        If provided, n-grams are returned as label tuples.
        Otherwise, integer tuples.

    Returns
    -------
    list of (ngram, count)
        Sorted by count descending.
    """
    actions = _resolve_player_actions(traj, player)  # (B, T)
    counter: Counter = Counter()

    for ep in actions:
        for t in range(len(ep) - n + 1):
            gram = tuple(ep[t : t + n].tolist())
            counter[gram] += 1

    if action_labels is not None:
        labeled: Counter = Counter()
        for gram, count in counter.items():
            labeled[tuple(action_labels[a] for a in gram)] = count
        return labeled.most_common(top_k)

    return counter.most_common(top_k)


def _blend_ngram_color(
    action_indices: tuple[int, ...],
    colors: Sequence[str],
) -> str:
    """Average the RGB values of the actions in an n-gram."""
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
    """Plot top-k n-grams for each n in a range, one row per n.

    Produces a vertically stacked set of horizontal bar charts. Each
    row shows the *top_k* most frequent n-grams for one value of *n*,
    making it easy to spot dominant action sequences at every scale
    from bigrams up to long motifs.

    Args:
        traj: Trajectory data.
        n_range: Inclusive ``(min_n, max_n)`` range for n-gram lengths.
        player: Player index for multi-player trajectories.
        top_k: Number of top n-grams per row.
        action_labels: Human-readable labels per action.
        colors: Hex colors per action, matching the raster palette.
            Each bar is colored by blending the colors of its
            constituent actions.
        figsize: Figure size. Defaults to a height scaled by the
            number of rows.
        title: Overall figure title.

    Returns:
        Tuple of ``(fig, axes)`` where *axes* is a 1-D array of
        ``Axes``, one per n value.
    """
    if action_labels is None:
        action_labels = DEFAULT_ACTION_LABELS
    if colors is None:
        colors = DEFAULT_ACTION_COLORS
    n_min, n_max = n_range
    n_values = list(range(n_min, n_max + 1))
    num_rows = len(n_values)

    if figsize is None:
        figsize = (12, 1.2 + 1.4 * num_rows)

    fig, axes = plt.subplots(
        num_rows, 1, figsize=figsize, squeeze=False,
    )
    axes = axes[:, 0]

    from matplotlib.patches import Patch, Rectangle
    from matplotlib.transforms import blended_transform_factory

    # Square size in axes-fraction (x) and data units (y).
    sq_w = 0.018
    sq_gap = 0.003
    sq_h = 0.7
    # Reserve left margin for colored boxes (widest n-gram determines width).
    box_margin = (n_max + 1) * (sq_w + sq_gap)

    for row, n in enumerate(n_values):
        ax = axes[row]
        grams_int = action_ngrams(traj, n, player, top_k)
        if not grams_int:
            ax.set_visible(False)
            continue
        labels = [
            " ".join(action_labels[a] for a in g)
            for g, _ in grams_int
        ]
        counts = [c for _, c in grams_int]
        bar_colors = [
            _blend_ngram_color(g, colors) for g, _ in grams_int
        ]
        ax.barh(
            range(len(labels)), counts, color=bar_colors, height=0.7,
        )
        # Action name text to the right of each bar.
        for idx, (label, count) in enumerate(zip(labels, counts)):
            ax.text(
                count, idx, f"  {label}", va="center", ha="left",
                fontsize=7, family="monospace",
            )
        ax.set_yticks([])
        ax.invert_yaxis()
        ax.set_ylabel(f"n={n}", fontsize=9, rotation=0, labelpad=30)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.tick_params(axis="x", labelsize=7)
        if row < num_rows - 1:
            ax.set_xticklabels([])
        # Expand x-axis to fit text labels.
        if counts:
            ax.set_xlim(right=max(counts) * 2.5)

        # Colored action squares in the left margin.
        trans = blended_transform_factory(ax.transAxes, ax.transData)
        for idx, (gram, _) in enumerate(grams_int):
            for j, a in enumerate(gram):
                x = -box_margin + j * (sq_w + sq_gap)
                ax.add_patch(Rectangle(
                    (x, idx - sq_h / 2), sq_w, sq_h,
                    facecolor=colors[min(a, len(colors) - 1)],
                    edgecolor="white", linewidth=0.3,
                    transform=trans, clip_on=False,
                ))

    # Action color legend at the top of the figure.
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
        bbox_to_anchor=(0.5, 1.0),
    )

    axes[-1].set_xlabel("Count")
    if title:
        fig.suptitle(title, fontsize=11, y=1.06)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig, axes


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
    """Plot a bar chart of the most common action n-grams.

    Parameters
    ----------
    traj, n, player, top_k, action_labels
        See :func:`action_ngrams`.
    ax, figsize, title
        Plotting options.

    Returns
    -------
    fig, ax : Figure, Axes
    """
    if action_labels is None:
        action_labels = DEFAULT_ACTION_LABELS
    grams = action_ngrams(traj, n, player, top_k, action_labels)

    labels = [" → ".join(g) for g, _ in grams]
    counts = [c for _, c in grams]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

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
    num_actions: int = 12,
    window: int = 1,
) -> np.ndarray:
    """Compute per-timestep action entropy across episodes.

    At each timestep, we compute the empirical distribution of actions
    across all episodes and return its entropy.  Higher entropy means the
    agent is less predictable at that point in the episode.

    Parameters
    ----------
    traj : Trajectory
    player : int, optional
    num_actions : int
    window : int
        Smoothing window.  If > 1, a rolling average is applied.

    Returns
    -------
    entropy : np.ndarray
        Shape ``(T,)`` — entropy in bits at each timestep.
    """
    actions = _resolve_player_actions(traj, player)  # (B, T)
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
    num_actions: int = 12,
    window: int = 5,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (12, 4),
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot action entropy over the course of an episode.

    Parameters
    ----------
    traj, player, num_actions, window
        See :func:`action_entropy`.
    ax, figsize, title
        Plotting options.

    Returns
    -------
    fig, ax : Figure, Axes
    """
    ent = action_entropy(traj, player, num_actions, window)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

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
    """Compute run-lengths of consecutive identical actions.

    Parameters
    ----------
    traj : Trajectory
    player : int, optional

    Returns
    -------
    dict mapping action_id -> list of run lengths
    """
    actions = _resolve_player_actions(traj, player)
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
    num_actions: int = 12,
    colors: Sequence[str] | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (10, 5),
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot run-length distributions as box plots per action.

    Reveals which actions tend to be repeated in long streaks (e.g., an
    agent walking in one direction for many steps) versus actions that
    rarely repeat (one-shot interactions).

    Parameters
    ----------
    traj, player, action_labels, num_actions, colors
        See other functions.
    ax, figsize, title
        Plotting options.

    Returns
    -------
    fig, ax : Figure, Axes
    """
    if action_labels is None:
        action_labels = DEFAULT_ACTION_LABELS[:num_actions]
    if colors is None:
        colors = DEFAULT_ACTION_COLORS[:num_actions]

    runs = run_lengths(traj, player)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

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
    num_actions: int = 12,
    action_labels: list[str] | None = None,
    colors: Sequence[str] | None = None,
    window: int = 10,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (14, 5),
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Stacked area chart showing how the action distribution evolves.

    At each timestep, the distribution of actions across episodes is
    computed and plotted as stacked areas.  A smoothing window makes
    the trends readable.

    Parameters
    ----------
    traj, player, num_actions, action_labels, colors
        Standard parameters.
    window : int
        Smoothing window size.
    ax, figsize, title
        Plotting options.

    Returns
    -------
    fig, ax : Figure, Axes
    """
    actions = _resolve_player_actions(traj, player)  # (B, T)
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

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

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
