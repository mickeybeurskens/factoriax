"""Comprehensive step throughput benchmark for FactoriaX.

Measures vmapped step_env throughput across a matrix of map sizes,
batch sizes, and entity array capacities. Results are logged to wandb
for tracking performance across commits.

Usage:
    python scripts/benchmark.py                  # Run on current code, log to wandb
    python scripts/benchmark.py --no-wandb       # Print to stdout only
    python scripts/benchmark.py --map-size 32    # Single map size
    python scripts/benchmark.py --history        # Benchmark past week's commits
    python scripts/benchmark.py --dry-run        # Show matrix without running
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)

MAP_SIZES = [8, 16, 32, 64]
BATCH_SIZES = [1, 64, 1024, 4096]
MAX_MACHINES_MODES = ["auto", "small"]
WARMUP_ITERS = 5
TIMED_ITERS = 50

PERFORMANCE_COMMITS = [
    "b3d20b2",
    "ae44678",
    "2776db2",
    "c69cfe5",
    "61b44a1",
    "dd0c9c4",
    "ac31b2c",
]


def _resolve_max_machines(mode: str, map_size: int) -> int:
    """Resolve max_machines for a given mode and map size.

    Args:
        mode: Either "auto" or "small".
        map_size: Map width/height.

    Returns:
        Concrete max_machines value.
    """
    if mode == "small":
        return 8
    return max(64, map_size * map_size // 4)


def _measure_throughput(
    map_size: int,
    batch_size: int,
    max_machines: int,
) -> float | None:
    """Measure vmapped step_env throughput for one configuration.

    Args:
        map_size: Square map side length.
        batch_size: Number of parallel environments.
        max_machines: Entity array capacity.

    Returns:
        Steps per second, or None if the configuration failed.
    """
    import jax
    import jax.numpy as jnp

    from factoriax.envs.factoriax_env import FactoriaXEnv
    from factoriax.state import EnvParams

    try:
        num_actions = 80
        try:
            from factoriax.constants import NUM_ACTIONS

            num_actions = NUM_ACTIONS
        except ImportError:
            pass

        params = EnvParams(
            map_width=map_size,
            map_height=map_size,
            num_players=1,
            max_machines=max_machines,
        )
        env = FactoriaXEnv()

        keys = jax.random.split(jax.random.key(42), batch_size)
        _, states = jax.vmap(env.reset_env, in_axes=(0, None))(keys, params)
        vmap_step = jax.jit(
            jax.vmap(env.step_env, in_axes=(0, 0, 0, None))
        )

        rng = jax.random.key(77)
        actions = jnp.zeros(batch_size, dtype=jnp.int32)
        step_keys = jax.random.split(rng, batch_size)
        for _ in range(WARMUP_ITERS):
            _, states, _, _, _ = vmap_step(
                step_keys, states, actions, params
            )
        jax.block_until_ready(jax.tree.leaves(states))

        rng_t = jax.random.key(88)
        t0 = time.perf_counter()
        for _ in range(TIMED_ITERS):
            rng_t, k_a, k_s = jax.random.split(rng_t, 3)
            actions = jax.random.randint(
                k_a, (batch_size,), 0, num_actions
            )
            step_keys = jax.random.split(k_s, batch_size)
            _, states, _, _, _ = vmap_step(
                step_keys, states, actions, params
            )
        jax.block_until_ready(jax.tree.leaves(states))
        elapsed = time.perf_counter() - t0

        return batch_size * TIMED_ITERS / elapsed
    except Exception as exc:
        logger.warning(
            "Failed map=%d batch=%d mm=%d: %s",
            map_size, batch_size, max_machines, exc,
        )
        return None


def _get_commit_info() -> dict[str, str]:
    """Get current git commit metadata.

    Returns:
        Dict with commit, message, and date keys.
    """
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
        message = subprocess.check_output(
            ["git", "log", "-1", "--format=%s"],
            text=True,
        ).strip()
        date = subprocess.check_output(
            ["git", "log", "-1", "--format=%ci"],
            text=True,
        ).strip()
        return {"commit": commit, "message": message, "date": date}
    except Exception:
        return {"commit": "unknown", "message": "", "date": ""}


def _get_device_name() -> str:
    """Get JAX device name.

    Returns:
        Device description string.
    """
    import jax

    return str(jax.devices()[0])


def run_matrix(
    map_sizes: list[int] | None = None,
    use_wandb: bool = True,
    commit_info: dict[str, str] | None = None,
) -> list[dict]:
    """Run the full benchmark matrix and log results.

    Args:
        map_sizes: Map sizes to benchmark. Uses all sizes when None.
        use_wandb: Whether to log to wandb.
        commit_info: Override commit metadata (for history mode).

    Returns:
        List of result dicts.
    """
    sizes = map_sizes or MAP_SIZES
    info = commit_info or _get_commit_info()
    device = _get_device_name()

    results: list[dict] = []
    configs = []
    for ms in sizes:
        for bs in BATCH_SIZES:
            for mm_mode in MAX_MACHINES_MODES:
                mm = _resolve_max_machines(mm_mode, ms)
                configs.append((ms, bs, mm, mm_mode))

    print(f"Device: {device}")
    print(f"Commit: {info['commit']} - {info['message']}")
    print(f"Date:   {info['date']}")
    print(f"Matrix: {len(configs)} configurations")
    print()
    print(
        f"  {'map':>5s}  {'batch':>6s}  {'max_m':>6s}  "
        f"{'mode':>5s}  {'steps/s':>12s}"
    )
    print(f"  {'-' * 5}  {'-' * 6}  {'-' * 6}  {'-' * 5}  {'-' * 12}")

    for ms, bs, mm, mm_mode in configs:
        sps = _measure_throughput(ms, bs, mm)
        row = {
            "map_size": ms,
            "batch_size": bs,
            "max_machines": mm,
            "max_machines_mode": mm_mode,
            "steps_per_sec": sps or 0.0,
        }
        results.append(row)
        sps_str = f"{sps:>12.0f}" if sps else "       FAILED"
        print(
            f"  {ms:>5d}  {bs:>6d}  {mm:>6d}  "
            f"{mm_mode:>5s}  {sps_str}"
        )

    if use_wandb and results:
        _log_wandb(results, info, device)

    return results


def _log_wandb(
    results: list[dict],
    info: dict[str, str],
    device: str,
) -> None:
    """Log benchmark results to wandb.

    Args:
        results: List of measurement dicts.
        info: Commit metadata.
        device: JAX device name.
    """
    try:
        import wandb  # type: ignore[import-untyped]
    except ImportError:
        logger.error("wandb not found. Install with: uv add wandb")
        return

    run = wandb.init(
        project="factoriax-benchmarks",
        name=f"{info['commit']} - {info['message'][:50]}",
        tags=[info["commit"], device.split(":")[0]],
        config={
            "commit": info["commit"],
            "commit_message": info["message"],
            "commit_date": info["date"],
            "device": device,
            "map_sizes": sorted({r["map_size"] for r in results}),
            "batch_sizes": sorted({r["batch_size"] for r in results}),
            "warmup_iters": WARMUP_ITERS,
            "timed_iters": TIMED_ITERS,
        },
    )

    columns = [
        "map_size", "batch_size", "max_machines",
        "max_machines_mode", "steps_per_sec",
    ]
    table = wandb.Table(columns=columns)
    for r in results:
        table.add_data(*[r[c] for c in columns])

    run.log({"throughput_matrix": table})

    for r in results:
        key = (
            f"sps/map{r['map_size']}_batch{r['batch_size']}"
            f"_mm{r['max_machines_mode']}"
        )
        run.summary[key] = r["steps_per_sec"]

    peak = max(r["steps_per_sec"] for r in results)
    run.summary["peak_steps_per_sec"] = peak

    run.finish()
    print(f"\nwandb run: {run.url}")


def run_history(
    use_wandb: bool = True,
    commits: list[str] | None = None,
) -> None:
    """Benchmark across historical commits using git worktrees.

    For each commit, creates an isolated worktree, installs deps,
    copies the benchmark script in, and runs it as a subprocess.

    Args:
        use_wandb: Whether to log to wandb.
        commits: Commits to benchmark. Uses PERFORMANCE_COMMITS
            when None.
    """
    commit_list = commits or PERFORMANCE_COMMITS
    script_path = Path(__file__).resolve()
    repo_root = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            text=True,
        ).strip()
    )

    print(f"Benchmarking {len(commit_list)} historical commits")
    print()

    for commit_hash in commit_list:
        full_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", commit_hash],
            text=True,
            cwd=repo_root,
        ).strip()
        message = subprocess.check_output(
            ["git", "log", "-1", "--format=%s", commit_hash],
            text=True,
            cwd=repo_root,
        ).strip()
        date = subprocess.check_output(
            ["git", "log", "-1", "--format=%ci", commit_hash],
            text=True,
            cwd=repo_root,
        ).strip()

        print(f"{'=' * 70}")
        print(f"Commit: {full_hash} - {message}")
        print(f"Date:   {date}")
        print(f"{'=' * 70}")

        work_dir = Path(tempfile.mkdtemp(prefix=f"factoriax-bench-{full_hash}-"))
        try:
            subprocess.run(
                [
                    "git", "worktree", "add",
                    str(work_dir), commit_hash,
                ],
                cwd=repo_root,
                check=True,
                capture_output=True,
            )

            scripts_dir = work_dir / "scripts"
            scripts_dir.mkdir(exist_ok=True)
            shutil.copy2(script_path, scripts_dir / "benchmark.py")

            wandb_flag = [] if use_wandb else ["--no-wandb"]
            env_cmd = (
                ["uv", "run", "python", "scripts/benchmark.py"]
                + wandb_flag
                + [
                    "--commit-hash", full_hash,
                    "--commit-message", message,
                    "--commit-date", date,
                ]
            )

            result = subprocess.run(
                env_cmd,
                cwd=work_dir,
                timeout=600,
                capture_output=False,
            )

            if result.returncode != 0:
                logger.error(
                    "Benchmark failed for %s (exit %d)",
                    full_hash, result.returncode,
                )
        except subprocess.TimeoutExpired:
            logger.error("Benchmark timed out for %s", full_hash)
        except Exception as exc:
            logger.error("Failed to benchmark %s: %s", full_hash, exc)
        finally:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(work_dir)],
                cwd=repo_root,
                capture_output=True,
            )
            if work_dir.exists():
                shutil.rmtree(work_dir, ignore_errors=True)

        print()


def main() -> None:
    """CLI entry point for the benchmark script."""
    parser = argparse.ArgumentParser(
        description="FactoriaX step throughput benchmark.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Print results to stdout only, skip wandb logging.",
    )
    parser.add_argument(
        "--map-size",
        type=int,
        help="Benchmark a single map size instead of all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the benchmark matrix without running.",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Benchmark performance-relevant commits from the past week.",
    )
    parser.add_argument(
        "--commit-hash",
        help="Override commit hash (used by --history subprocess).",
    )
    parser.add_argument(
        "--commit-message",
        help="Override commit message (used by --history subprocess).",
    )
    parser.add_argument(
        "--commit-date",
        help="Override commit date (used by --history subprocess).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    use_wandb = not args.no_wandb

    if args.history:
        run_history(use_wandb=use_wandb)
        return

    if args.dry_run:
        sizes = [args.map_size] if args.map_size else MAP_SIZES
        print("Benchmark matrix (dry run):")
        for ms in sizes:
            for bs in BATCH_SIZES:
                for mm_mode in MAX_MACHINES_MODES:
                    mm = _resolve_max_machines(mm_mode, ms)
                    print(
                        f"  map={ms:>3d}  batch={bs:>5d}  "
                        f"max_machines={mm:>3d} ({mm_mode})"
                    )
        return

    commit_info = None
    if args.commit_hash:
        commit_info = {
            "commit": args.commit_hash,
            "message": args.commit_message or "",
            "date": args.commit_date or "",
        }

    map_sizes = [args.map_size] if args.map_size else None
    run_matrix(
        map_sizes=map_sizes,
        use_wandb=use_wandb,
        commit_info=commit_info,
    )


if __name__ == "__main__":
    main()
