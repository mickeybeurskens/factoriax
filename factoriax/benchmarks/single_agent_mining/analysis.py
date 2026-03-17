"""Analysis and visualisation for single-agent mining benchmark results.

Each function takes a ``BenchmarkResult`` and returns a Matplotlib figure.
Figures are returned rather than shown or saved so callers control I/O.
W&B logging is opt-in: pass a live ``wandb.Run`` object or ``None``.

All functions gracefully degrade when the result has zero items mined
(e.g. from a random policy) — they still produce valid plots.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from factoriax.benchmarks.core import BenchmarkResult
from factoriax.benchmarks.single_agent_mining.scoring import RESOURCE_WEIGHTS
from factoriax.constants import Action

if TYPE_CHECKING:
    pass

# Consistent colours for each resource type across all plots.
_RESOURCE_COLORS: dict[str, str] = {
    "coal": "#363636",
    "iron": "#9e9e9e",
    "copper": "#b26a00",
}

_ACTION_LABELS: list[str] = [a.name for a in Action]


def plot_level_scores(result: BenchmarkResult) -> plt.Figure:
    """Bar chart comparing per-level scores.

    Args:
        result: Benchmark result to visualise.

    Returns:
        Matplotlib figure with one bar per level, coloured by score.
    """
    names = [lr.level_name for lr in result.level_results]
    scores = [lr.weighted_score for lr in result.level_results]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(names, scores, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.axhline(result.aggregate_score, color="crimson", linestyle="--", linewidth=1.2,
               label=f"aggregate mean = {result.aggregate_score:.1f}")
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    ax.set_ylabel("Weighted score")
    ax.set_title(f"{result.benchmark_name} — per-level scores")
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(scores) * 1.2 if any(s > 0 for s in scores) else 1)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    return fig


def plot_resource_breakdown(result: BenchmarkResult) -> plt.Figure:
    """Stacked bar chart of resources collected per level.

    Each bar shows the contribution of coal, iron, and copper to the total
    score for that level. The bar height is the *weighted* contribution
    (item count × weight), so the full bar equals the level score.

    Args:
        result: Benchmark result to visualise.

    Returns:
        Matplotlib figure with stacked bars.
    """
    names = [lr.level_name for lr in result.level_results]
    resources = list(RESOURCE_WEIGHTS.keys())

    weighted: dict[str, list[float]] = {
        r: [
            lr.items_mined.get(r, 0) * RESOURCE_WEIGHTS[r]
            for lr in result.level_results
        ]
        for r in resources
    }

    fig, ax = plt.subplots(figsize=(8, 4))
    bottoms = np.zeros(len(names))
    for resource in resources:
        vals = np.array(weighted[resource], dtype=float)
        ax.bar(
            names,
            vals,
            bottom=bottoms,
            label=resource,
            color=_RESOURCE_COLORS[resource],
            edgecolor="white",
            linewidth=0.4,
        )
        bottoms += vals

    ax.set_ylabel("Weighted score contribution")
    ax.set_title(f"{result.benchmark_name} — resource breakdown")
    ax.legend(title="Resource", fontsize=9)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    return fig


def plot_action_distribution(result: BenchmarkResult) -> plt.Figure:
    """Grouped bar chart of action frequencies per level.

    Shows how often each action was taken across the episode. Useful for
    diagnosing pathological behaviours like excessive NOOPs or zero MINE
    actions.

    Args:
        result: Benchmark result to visualise.

    Returns:
        Matplotlib figure with one group of bars per action.
    """
    num_levels = len(result.level_results)
    num_actions = len(Action)
    counts = np.zeros((num_levels, num_actions), dtype=float)

    for i, lr in enumerate(result.level_results):
        if lr.actions.size > 0:
            for a in range(num_actions):
                counts[i, a] = float(np.sum(lr.actions == a))
            counts[i] /= lr.actions.size  # normalise to frequency

    x = np.arange(num_actions)
    width = 0.8 / num_levels
    level_names = [lr.level_name for lr in result.level_results]

    fig, ax = plt.subplots(figsize=(12, 4))
    for i, name in enumerate(level_names):
        offset = (i - num_levels / 2 + 0.5) * width
        ax.bar(x + offset, counts[i], width=width, label=name, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(_ACTION_LABELS, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Fraction of steps")
    ax.set_title(f"{result.benchmark_name} — action distribution")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def log_to_wandb(
    result: BenchmarkResult,
    wandb_run: Any | None,
    step: int | None = None,
) -> None:
    """Log benchmark results and analysis figures to Weights and Biases.

    Logs per-level scores, the aggregate score, a resource breakdown table,
    and all three analysis figures as ``wandb.Image`` objects.

    Args:
        result: Benchmark result to log.
        wandb_run: A live ``wandb.Run`` instance (from ``wandb.init()``) or
            ``None``. When ``None`` this function is a no-op.
        step: Optional global training step to associate with the log entry.
            Pass ``None`` to log without a step.
    """
    if wandb_run is None:
        return

    try:
        import wandb  # type: ignore[import-untyped]
    except ImportError:
        return

    prefix = f"{result.benchmark_name}/"
    log_data: dict[str, Any] = {
        f"{prefix}aggregate_score": result.aggregate_score,
    }
    for lr in result.level_results:
        lvl = lr.level_name
        log_data[f"{prefix}{lvl}/score"] = lr.weighted_score
        log_data[f"{prefix}{lvl}/coal"] = lr.items_mined.get("coal", 0)
        log_data[f"{prefix}{lvl}/iron"] = lr.items_mined.get("iron", 0)
        log_data[f"{prefix}{lvl}/copper"] = lr.items_mined.get("copper", 0)
        log_data[f"{prefix}{lvl}/steps"] = lr.timesteps_used

    fig_scores = plot_level_scores(result)
    fig_breakdown = plot_resource_breakdown(result)
    fig_actions = plot_action_distribution(result)

    log_data[f"{prefix}plots/level_scores"] = wandb.Image(fig_scores)
    log_data[f"{prefix}plots/resource_breakdown"] = wandb.Image(fig_breakdown)
    log_data[f"{prefix}plots/action_distribution"] = wandb.Image(fig_actions)

    plt.close(fig_scores)
    plt.close(fig_breakdown)
    plt.close(fig_actions)

    if step is not None:
        wandb_run.log(log_data, step=step)
    else:
        wandb_run.log(log_data)
