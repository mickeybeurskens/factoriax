"""Benchmark matrix: steps/s across map sizes and batch sizes.

Produces a table of throughput for every (map_size, batch_size) combination.
Results are printed and optionally saved to a JSON file for comparison.

Usage:
    python scripts/benchmark_matrix.py --label grid
    python scripts/benchmark_matrix.py --label entity
    python scripts/benchmark_matrix.py --compare grid entity
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from factoriax.constants import NUM_ACTIONS
from factoriax.envs.factoriax_env import FactoriaXEnv
from factoriax.state import EnvParams

RESULTS_DIR = Path("docs/profiling")
WARMUP = 5
ITERS = 50

MAP_SIZES = [16, 32, 64, 128]
BATCH_SIZES_BY_MAP = {
    16: [64, 256, 1024, 4096],
    32: [64, 256, 1024, 4096],
    64: [64, 256, 1024],
    128: [64, 256],
}


def benchmark_cell(
    map_size: int, batch_size: int,
) -> float:
    """Measure steps/s for one (map_size, batch_size) cell.

    Args:
        map_size: Map width and height.
        batch_size: Number of parallel environments.

    Returns:
        Steps per second.
    """
    env = FactoriaXEnv()
    params = EnvParams(
        map_width=map_size, map_height=map_size, num_players=1,
    )
    keys = jax.random.split(jax.random.key(42), batch_size)
    _, states = jax.vmap(env.reset_env, in_axes=(0, None))(keys, params)
    step_keys = jax.random.split(jax.random.key(99), batch_size)
    actions = jax.random.randint(
        jax.random.key(7), (batch_size,), 0, NUM_ACTIONS,
    )
    vmap_step = jax.jit(
        jax.vmap(env.step_env, in_axes=(0, 0, 0, None)),
    )

    for _ in range(WARMUP):
        _, states, _, _, _ = vmap_step(
            step_keys, states, actions, params,
        )
    jax.block_until_ready(jax.tree.leaves(states))

    t0 = time.perf_counter()
    for _ in range(ITERS):
        _, states, _, _, _ = vmap_step(
            step_keys, states, actions, params,
        )
    jax.block_until_ready(jax.tree.leaves(states))
    elapsed = time.perf_counter() - t0

    return batch_size * ITERS / elapsed


def run_matrix() -> dict[str, float]:
    """Run the full benchmark matrix.

    Returns:
        Dict mapping "MAPxMAP_BATCH" to steps/s.
    """
    results: dict[str, float] = {}
    for map_size in MAP_SIZES:
        for batch_size in BATCH_SIZES_BY_MAP[map_size]:
            key = f"{map_size}x{map_size}_b{batch_size}"
            try:
                sps = benchmark_cell(map_size, batch_size)
                results[key] = sps
                print(f"  {key}: {sps:>12,.0f} steps/s")
            except Exception as e:
                results[key] = -1
                print(f"  {key}: FAILED ({e})")
    return results


def save_results(label: str, results: dict[str, float]) -> Path:
    """Save results to a JSON file.

    Args:
        label: Name for this run (e.g., "grid", "entity").
        results: Benchmark results dict.

    Returns:
        Path to the saved file.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"matrix_{label}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {path}")
    return path


def compare_results(label_a: str, label_b: str) -> None:
    """Compare two saved benchmark results.

    Args:
        label_a: First label (baseline).
        label_b: Second label (candidate).
    """
    path_a = RESULTS_DIR / f"matrix_{label_a}.json"
    path_b = RESULTS_DIR / f"matrix_{label_b}.json"
    with open(path_a) as f:
        results_a = json.load(f)
    with open(path_b) as f:
        results_b = json.load(f)

    print(f"\nComparison: {label_a} vs {label_b}")
    print(f"{'Cell':<20s} {label_a:>12s} {label_b:>12s} {'Delta':>8s} {'Verdict':>8s}")
    print("-" * 62)

    for key in sorted(results_a.keys()):
        a = results_a.get(key, -1)
        b = results_b.get(key, -1)
        if a <= 0 or b <= 0:
            print(f"{key:<20s} {'FAIL':>12s} {'FAIL':>12s}")
            continue
        delta = (b - a) / a * 100
        if delta > 5:
            verdict = "GREEN"
        elif delta < -5:
            verdict = "RED"
        else:
            verdict = "YELLOW"
        print(
            f"{key:<20s} {a:>12,.0f} {b:>12,.0f} "
            f"{delta:>+7.1f}% {verdict:>8s}"
        )


def main() -> None:
    """Run benchmark matrix or compare results."""
    parser = argparse.ArgumentParser(
        description="Benchmark matrix across map and batch sizes.",
    )
    parser.add_argument(
        "--label", type=str, help="Run matrix and save with this label",
    )
    parser.add_argument(
        "--compare", nargs=2, metavar=("A", "B"),
        help="Compare two saved results",
    )
    args = parser.parse_args()

    if args.compare:
        compare_results(args.compare[0], args.compare[1])
    elif args.label:
        print(f"Running benchmark matrix (label={args.label}):")
        results = run_matrix()
        save_results(args.label, results)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
