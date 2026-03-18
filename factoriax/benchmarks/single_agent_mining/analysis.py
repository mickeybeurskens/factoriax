"""Analysis and visualisation for single-agent mining benchmark results.

Each plot function takes a ``BenchmarkResult`` and returns a Matplotlib figure.
Figures are returned rather than shown or saved so callers control I/O.

``render_level_video`` re-runs a policy through one level and captures an RGB
frame at each step. ``save_mp4`` writes those frames to disk. Neither is
JIT-compatible — they are for offline visualisation only.

W&B logging is opt-in via ``log_to_wandb``. Pass video paths produced by
``save_mp4`` to have them uploaded alongside the scalar metrics and figures.

All functions degrade gracefully when the result has zero items mined.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from factoriax.benchmarks.core import BenchmarkLevel, BenchmarkResult, Policy
from factoriax.benchmarks.single_agent_mining.scoring import RESOURCE_WEIGHTS
from factoriax.constants import Action
from factoriax.state import EnvParams, EnvState

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


def render_level_video(
    bench_level: BenchmarkLevel,
    policy: Policy,
    seed: int = 0,
    obs_fn: Callable[[EnvState, EnvParams, int], jax.Array] | None = None,
) -> list[np.ndarray]:
    """Run a policy through one level and return rendered RGB frames.

    Captures one frame per step from the pre-step state so the video shows
    what the agent sees before each action. A final frame is appended after
    the last step. Not JIT-compatible; intended for offline visualisation only.

    Args:
        bench_level: Level to render.
        policy: Policy to evaluate. Receives a float32 JAX observation array
            and returns a JAX integer action scalar.
        seed: Random seed for ``step_env``. Does not affect the policy's own
            PRNG if it manages its own key.
        obs_fn: Observation extraction function with signature
            ``(state, env_params, player_idx) -> obs_array``. Defaults to
            ``global_array``. Pass a custom function to match the observation
            space used during training (e.g. ``local_array`` with a fixed
            radius for policies trained with local observations).

    Returns:
        List of ``(H, W, 3)`` uint8 numpy arrays, one per step plus one
        final frame.
    """
    from factoriax.envs import FactoriaXEnv
    from factoriax.levels import build_state
    from factoriax.observations import global_array
    from factoriax.renderer import render_pixels

    _obs_fn = obs_fn if obs_fn is not None else global_array
    env = FactoriaXEnv()
    jit_step = jax.jit(env.step_env)
    rng = jax.random.PRNGKey(seed)
    params = bench_level.env_params
    state = build_state(bench_level.level, params)

    frames: list[np.ndarray] = []
    for _ in range(params.max_timesteps):
        frames.append(render_pixels(state))
        obs = _obs_fn(state, params, 0)
        action = policy(obs)
        rng, subkey = jax.random.split(rng)
        _, state, _, done, _ = jit_step(subkey, state, action, params)
        if bool(done):
            break

    frames.append(render_pixels(state))
    return frames


def save_mp4(frames: list[np.ndarray], path: Path, fps: int = 10) -> None:
    """Write a list of RGB frames to an MP4 file.

    Args:
        frames: List of ``(H, W, 3)`` uint8 numpy arrays.
        path: Destination file path. Parent directories are created if absent.
        fps: Frames per second.

    Raises:
        ValueError: If ``frames`` is empty.
    """
    if not frames:
        raise ValueError("frames list is empty; cannot write MP4.")
    import warnings
    import imageio.v3 as iio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # imageio-ffmpeg spawns ffmpeg via os.fork()+exec. JAX registers an
    # os.register_at_fork() callback that warns whenever fork is called after
    # JAX's thread pool has started. The fork succeeds and no deadlock occurs
    # in practice, so we suppress the false-positive here.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="os.fork()")
        iio.imwrite(
            str(path),
            np.stack([f.astype(np.uint8) for f in frames]),
            plugin="FFMPEG",
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
        )


def log_to_wandb(
    result: BenchmarkResult,
    wandb_run: Any | None,
    step: int | None = None,
    videos: dict[str, Path] | None = None,
) -> None:
    """Log benchmark results, figures, and optional videos to Weights and Biases.

    Args:
        result: Benchmark result to log.
        wandb_run: A live ``wandb.Run`` instance or ``None`` (no-op).
        step: Optional global training step for the log entry.
        videos: Optional mapping of level name to MP4 file path. Each path
            is uploaded as a ``wandb.Video`` under
            ``{benchmark_name}/videos/{level_name}``.
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

    if videos:
        for level_name, mp4_path in videos.items():
            log_data[f"{prefix}videos/{level_name}"] = wandb.Video(
                str(mp4_path), fps=10, format="mp4"
            )

    if step is not None:
        wandb_run.log(log_data, step=step)
    else:
        wandb_run.log(log_data)
