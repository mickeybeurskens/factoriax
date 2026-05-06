"""Quick terminal profiler using lax.scan for accurate throughput.

Uses jax.lax.scan to fuse multiple steps into a single XLA call,
eliminating Python loop overhead. Step counts are tuned per config
so each measurement takes ~10s without requiring time-based loops.

Usage:
    python scripts/quick_profile_to_term.py
    python scripts/quick_profile_to_term.py --map-size 16 --batch 4096
"""

from __future__ import annotations

import argparse
import time

import jax
from jax import lax, random

from factoriax.achievements import core_game_conditions
from factoriax.constants import Action
from factoriax.envs.factoriax_env import FactoriaXEnv
from factoriax.state import EnvParams


def profile(
    map_size: int = 32,
    batch: int = 8192,
    scan_length: int = 128,
    num_scans: int = 3,
) -> dict[str, float]:
    """Profile one configuration with lax.scan and separated warmup.

    Args:
        map_size: Square map side length.
        batch: Number of parallel environments.
        scan_length: Steps fused per lax.scan call.
        num_scans: Number of scan calls for measurement.

    Returns:
        Dict with warmup_s, call_ms, steps_per_second, total_steps.
    """
    env = FactoriaXEnv(achievement_fn=core_game_conditions)
    params = EnvParams(map_width=map_size, map_height=map_size)

    keys = jax.random.split(jax.random.PRNGKey(0), batch)
    reset_fn = jax.vmap(env.reset_env, in_axes=(0, None))
    vmap_step = jax.vmap(env.step_env, in_axes=(0, 0, None, None))
    _, states = reset_fn(keys, params)

    def scan_body(carry, _):
        """One step inside lax.scan."""
        st, rng = carry
        rng, key = random.split(rng)
        sk = random.split(key, batch)
        _, st, _, _, _ = vmap_step(
            sk,
            st,
            int(Action.NOOP),
            params,
        )
        return (st, rng), None

    scan_fn = jax.jit(
        lambda s, r: lax.scan(
            scan_body,
            (s, r),
            None,
            length=scan_length,
        ),
    )

    # 1. Warmup (JIT compile).
    t0 = time.perf_counter()
    (states, _), _ = scan_fn(states, random.PRNGKey(99))
    jax.block_until_ready(states)
    warmup_s = time.perf_counter() - t0

    # 2. Single scan call (retrace check).
    t0 = time.perf_counter()
    (states, _), _ = scan_fn(states, random.PRNGKey(100))
    jax.block_until_ready(states)
    call_ms = (time.perf_counter() - t0) * 1000
    cache = scan_fn._cache_size()

    # 3. Throughput measurement.
    rng = random.PRNGKey(42)
    total = 0
    t0 = time.perf_counter()
    for _ in range(num_scans):
        (states, rng), _ = scan_fn(states, rng)
        jax.block_until_ready(states)
        total += batch * scan_length
    elapsed = time.perf_counter() - t0
    sps = total / elapsed

    return {
        "warmup_s": warmup_s,
        "call_ms": call_ms,
        "cache_size": cache,
        "steps_per_second": sps,
        "total_steps": total,
        "elapsed_s": elapsed,
        "scan_length": scan_length,
        "num_scans": num_scans,
    }


# Tuned configs: (map_size, batch, scan_length, num_scans)
# Each should take ~10s for the measurement phase.
DEFAULT_CONFIGS = [
    (8, 16384, 256, 3),  # small map, big batch, fast
    (16, 8192, 128, 3),  # medium
    (32, 4096, 64, 3),  # large map, smaller batch
    (64, 2048, 32, 3),  # biggest map, small batch
]


def main() -> None:
    """Run quick profile and print results."""
    parser = argparse.ArgumentParser(
        description="Quick lax.scan profiler.",
    )
    parser.add_argument(
        "--map-size",
        type=int,
        default=None,
        help="Single map size (overrides default sweep).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Batch size (overrides default).",
    )
    parser.add_argument(
        "--scan-length",
        type=int,
        default=128,
        help="Steps per lax.scan call (default: 128).",
    )
    parser.add_argument(
        "--num-scans",
        type=int,
        default=3,
        help="Scan calls for measurement (default: 3).",
    )
    args = parser.parse_args()

    if args.map_size is not None:
        configs = [
            (
                args.map_size,
                args.batch or 4096,
                args.scan_length,
                args.num_scans,
            )
        ]
    else:
        configs = DEFAULT_CONFIGS

    print(
        f"{'Map':>5} {'Batch':>7} {'Scan':>5} "
        f"{'JIT':>7} {'Call':>8} {'Steps/s':>12} "
        f"{'Cache':>5}"
    )
    print("-" * 60)

    for ms, bs, sl, ns in configs:
        print(
            f"{ms:>5} {bs:>7} {sl:>5}  ",
            end="",
            flush=True,
        )
        r = profile(ms, bs, sl, ns)
        print(
            f"{r['warmup_s']:>6.1f}s "
            f"{r['call_ms']:>7.0f}ms "
            f"{r['steps_per_second']:>11,.0f} "
            f"{r['cache_size']:>5}"
        )


if __name__ == "__main__":
    main()
