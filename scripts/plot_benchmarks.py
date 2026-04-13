"""Generate benchmark progression plots from wandb data.

Fetches throughput data from the factoriax-benchmarks wandb project
and produces plots showing performance across commits for different
map sizes and entity configurations.

Usage:
    python scripts/plot_benchmarks.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("docs/profiling/plots")


def fetch_wandb_data() -> list[dict]:
    """Fetch all benchmark runs from wandb.

    Returns:
        List of run dicts with config and summary data.
    """
    import wandb  # type: ignore[import-untyped]

    api = wandb.Api()
    runs = api.runs("factoriax-benchmarks")

    data = []
    for run in runs:
        commit = run.config.get("commit", "?")
        message = run.config.get("commit_message", "")
        date = run.config.get("commit_date", "")
        summary = dict(run.summary)
        data.append({
            "commit": commit,
            "message": message,
            "date": date,
            "summary": summary,
        })

    data.sort(key=lambda r: r["date"])
    return data


def _short_label(commit: str, message: str) -> str:
    """Build a short x-axis label from commit and message.

    Args:
        commit: Short commit hash.
        message: Commit message.

    Returns:
        Abbreviated label.
    """
    words = message.split()[:4]
    short_msg = " ".join(words)
    if len(message.split()) > 4:
        short_msg += "..."
    return f"{commit}\n{short_msg}"


def plot_map_size_progression(
    data: list[dict],
    map_size: int,
    batch_size: int = 4096,
) -> None:
    """Plot throughput across commits for a given map size.

    Shows both auto and small max_machines modes.

    Args:
        data: Benchmark run data from wandb.
        map_size: Map size to plot.
        batch_size: Batch size to plot.
    """
    auto_key = f"sps/map{map_size}_batch{batch_size}_mmauto"
    small_key = f"sps/map{map_size}_batch{batch_size}_mmsmall"

    labels = []
    auto_vals = []
    small_vals = []

    for run in data:
        s = run["summary"]
        auto_v = s.get(auto_key, 0)
        small_v = s.get(small_key, 0)
        if auto_v == 0 and small_v == 0:
            continue
        labels.append(_short_label(run["commit"], run["message"]))
        auto_vals.append(auto_v / 1000)
        small_vals.append(small_v / 1000)

    if not labels:
        logger.warning("No data for map=%d batch=%d", map_size, batch_size)
        return

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.5), 6))
    bars_auto = ax.bar(
        x - width / 2, auto_vals, width,
        label="auto max_machines", color="#4a90d9", alpha=0.85,
    )
    bars_small = ax.bar(
        x + width / 2, small_vals, width,
        label="max_machines=8", color="#e8a838", alpha=0.85,
    )

    ax.set_ylabel("Throughput (k steps/s)")
    ax.set_title(
        f"Step Throughput: {map_size}x{map_size} map, "
        f"batch={batch_size}"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bars in [bars_auto, bars_small]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(
                    f"{h:.0f}k",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=7,
                )

    fig.tight_layout()
    path = OUTPUT_DIR / f"throughput_{map_size}x{map_size}_b{batch_size}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_entity_scaling(
    data: list[dict],
    batch_size: int = 4096,
) -> None:
    """Plot throughput vs map size for auto vs small entities.

    Uses the latest commit's data.

    Args:
        data: Benchmark run data from wandb.
        batch_size: Batch size to plot.
    """
    latest = data[-1] if data else None
    if not latest:
        return

    s = latest["summary"]
    map_sizes = [8, 16, 32, 64]
    auto_vals = []
    small_vals = []

    for ms in map_sizes:
        auto_key = f"sps/map{ms}_batch{batch_size}_mmauto"
        small_key = f"sps/map{ms}_batch{batch_size}_mmsmall"
        auto_vals.append(s.get(auto_key, 0) / 1000)
        small_vals.append(s.get(small_key, 0) / 1000)

    x = np.arange(len(map_sizes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        x - width / 2, auto_vals, width,
        label="auto max_machines", color="#4a90d9", alpha=0.85,
    )
    ax.bar(
        x + width / 2, small_vals, width,
        label="max_machines=8", color="#e8a838", alpha=0.85,
    )

    for i, (a, s_val) in enumerate(zip(auto_vals, small_vals)):
        if a > 0 and s_val > 0:
            ratio = s_val / a
            ax.annotate(
                f"{ratio:.1f}x",
                xy=(i + width / 2, s_val),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center", fontsize=9, fontweight="bold",
                color="#c0620a",
            )

    ax.set_ylabel("Throughput (k steps/s)")
    ax.set_title(
        f"Entity Array Impact on Throughput "
        f"(batch={batch_size}, commit {latest['commit']})"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{ms}x{ms}" for ms in map_sizes])
    ax.set_xlabel("Map Size")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = OUTPUT_DIR / f"entity_scaling_b{batch_size}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_peak_throughput(data: list[dict]) -> None:
    """Plot peak throughput across commits.

    Args:
        data: Benchmark run data from wandb.
    """
    labels = []
    peaks = []

    for run in data:
        peak = run["summary"].get("peak_steps_per_sec", 0)
        if peak == 0:
            continue
        labels.append(_short_label(run["commit"], run["message"]))
        peaks.append(peak / 1000)

    if not labels:
        return

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.5), 5))
    bars = ax.bar(
        range(len(labels)), peaks,
        color="#2ecc71", alpha=0.85, edgecolor="#27ae60",
    )

    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.0f}k",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_ylabel("Peak Throughput (k steps/s)")
    ax.set_title("Peak Step Throughput Across Commits")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = OUTPUT_DIR / "peak_throughput.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def main() -> None:
    """Fetch wandb data and generate all plots."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching wandb data...")
    data = fetch_wandb_data()
    print(f"Found {len(data)} runs")
    print()

    print("Generating plots:")
    for ms in [8, 16, 32]:
        plot_map_size_progression(data, ms, batch_size=4096)

    plot_entity_scaling(data, batch_size=4096)
    plot_peak_throughput(data)
    print()
    print(f"All plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
