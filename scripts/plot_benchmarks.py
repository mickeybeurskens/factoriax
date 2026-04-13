"""Generate benchmark progression plots and upload to wandb.

Fetches throughput data from the factoriax-benchmarks wandb project,
generates comparison plots across commits, and uploads them as a
separate wandb run for easy sharing.

Usage:
    python scripts/plot_benchmarks.py
    python scripts/plot_benchmarks.py --no-wandb   # Save locally only
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("docs/profiling/plots")


def fetch_wandb_data() -> list[dict]:
    """Fetch all benchmark runs from wandb.

    Returns:
        List of run dicts sorted by commit date, each containing
        commit, message, date, and summary metrics.
    """
    import wandb  # type: ignore[import-untyped]

    api = wandb.Api()
    runs = api.runs("factoriax-benchmarks")

    data = []
    for run in runs:
        if run.name and run.name.startswith("plots"):
            continue
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
    seen: set[str] = set()
    deduped = []
    for r in data:
        if r["commit"] not in seen:
            seen.add(r["commit"])
            deduped.append(r)
    return deduped


def _short_label(commit: str, message: str) -> str:
    """Build a short x-axis label from commit and message.

    Args:
        commit: Short commit hash.
        message: Commit message.

    Returns:
        Abbreviated label.
    """
    words = message.split()[:3]
    short_msg = " ".join(words)
    if len(message.split()) > 3:
        short_msg += "..."
    return f"{commit}\n{short_msg}"


def plot_map_throughput(
    data: list[dict],
    map_size: int,
    batch_size: int = 4096,
) -> plt.Figure | None:
    """Plot throughput across commits for a given map size.

    Args:
        data: Benchmark run data from wandb.
        map_size: Map size to plot.
        batch_size: Batch size to plot.

    Returns:
        Matplotlib figure, or None if no data.
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
        return None

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.4), 6))
    ax.bar(
        x - width / 2, auto_vals, width,
        label="auto max_machines", color="#4a90d9", alpha=0.85,
    )
    ax.bar(
        x + width / 2, small_vals, width,
        label="max_machines=8", color="#e8a838", alpha=0.85,
    )

    ax.set_ylabel("Throughput (k steps/s)")
    ax.set_title(
        f"Step Throughput: {map_size}x{map_size} map, batch={batch_size}"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_entity_scaling(data: list[dict], batch_size: int = 4096) -> plt.Figure | None:
    """Plot entity array impact on throughput for the latest commit.

    Args:
        data: Benchmark run data from wandb.
        batch_size: Batch size to plot.

    Returns:
        Matplotlib figure, or None if no data.
    """
    latest = data[-1] if data else None
    if not latest:
        return None

    s = latest["summary"]
    map_sizes = [8, 16, 32, 64]
    auto_vals = []
    small_vals = []

    for ms in map_sizes:
        auto_vals.append(
            s.get(f"sps/map{ms}_batch{batch_size}_mmauto", 0) / 1000
        )
        small_vals.append(
            s.get(f"sps/map{ms}_batch{batch_size}_mmsmall", 0) / 1000
        )

    if all(v == 0 for v in auto_vals + small_vals):
        return None

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

    for i, (a, sv) in enumerate(zip(auto_vals, small_vals)):
        if a > 0 and sv > 0:
            ax.annotate(
                f"{sv / a:.1f}x",
                xy=(i + width / 2, sv),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center", fontsize=9, fontweight="bold",
                color="#c0620a",
            )

    ax.set_ylabel("Throughput (k steps/s)")
    ax.set_title(
        f"Entity Array Impact (batch={batch_size}, "
        f"commit {latest['commit']})"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{ms}x{ms}" for ms in map_sizes])
    ax.set_xlabel("Map Size")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_peak_throughput(data: list[dict]) -> plt.Figure | None:
    """Plot peak throughput across commits.

    Args:
        data: Benchmark run data from wandb.

    Returns:
        Matplotlib figure, or None if no data.
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
        return None

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.4), 5))
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
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def main() -> None:
    """Fetch wandb data, generate plots, and upload."""
    parser = argparse.ArgumentParser(
        description="Generate and upload benchmark plots.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Save plots locally only, skip wandb upload.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching wandb data...")
    data = fetch_wandb_data()
    print(f"Found {len(data)} runs (deduplicated by commit)")
    print()

    figures: dict[str, plt.Figure] = {}

    for ms in [8, 16, 32]:
        fig = plot_map_throughput(data, ms, batch_size=4096)
        if fig:
            name = f"throughput_{ms}x{ms}_b4096"
            figures[name] = fig
            path = OUTPUT_DIR / f"{name}.png"
            fig.savefig(path, dpi=150)
            print(f"  Saved {path}")

    fig = plot_entity_scaling(data, batch_size=4096)
    if fig:
        figures["entity_scaling_b4096"] = fig
        path = OUTPUT_DIR / "entity_scaling_b4096.png"
        fig.savefig(path, dpi=150)
        print(f"  Saved {path}")

    fig = plot_peak_throughput(data)
    if fig:
        figures["peak_throughput"] = fig
        path = OUTPUT_DIR / "peak_throughput.png"
        fig.savefig(path, dpi=150)
        print(f"  Saved {path}")

    if not args.no_wandb and figures:
        _upload_wandb(figures, data)

    plt.close("all")
    print(f"\nAll plots saved to {OUTPUT_DIR}/")


def _upload_wandb(
    figures: dict[str, plt.Figure],
    data: list[dict],
) -> None:
    """Upload plots to wandb as a separate analysis run.

    Args:
        figures: Named matplotlib figures to upload.
        data: Source benchmark data for metadata.
    """
    try:
        import wandb  # type: ignore[import-untyped]
    except ImportError:
        logger.error("wandb not found. Install with: uv add wandb")
        return

    commits = [r["commit"] for r in data]
    run = wandb.init(
        project="factoriax-benchmarks",
        name=f"plots - {len(data)} commits",
        job_type="analysis",
        tags=["plots"],
        config={
            "commits_analyzed": commits,
            "num_commits": len(data),
        },
    )

    for name, fig in figures.items():
        run.log({name: wandb.Image(fig)})

    run.finish()
    print(f"\nwandb plots run: {run.url}")


if __name__ == "__main__":
    main()
