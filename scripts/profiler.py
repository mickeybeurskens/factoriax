"""JAX profiler: throughput, memory, and Perfetto traces for FactoriaX.

Measures vmapped step throughput across map sizes, automatically finds
the max batch size per map before OOM, captures GPU memory stats, and
optionally records Perfetto traces for kernel-level analysis.

Results are saved to ``profiled.json`` with the current git commit hash
and timestamp. Perfetto traces go to ``profiles/<commit>/``.

Usage::

    python scripts/profiler.py
    python scripts/profiler.py --map-sizes 16 64
    python scripts/profiler.py --no-trace
    python scripts/profiler.py --duration 5
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import jax
import orjson

from factoriax.constants import Action
from factoriax.envs.factoriax_env import make_factoriax_env
from factoriax.state import EnvParams

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _get_commit_hash() -> str:
    """Return the short git commit hash of the working tree."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _get_gpu_name() -> str:
    """Return the name of the first JAX device."""
    devices = jax.local_devices()
    if not devices:
        return "no device"
    d = devices[0]
    return f"{d.platform}:{d.device_kind}"


def _find_max_batch(
    env: object,
    params: EnvParams,
    cap: int = 65536,
) -> int:
    """Find the largest batch size that fits in GPU memory.

    Doubles from 256 until reset OOMs, then returns the last
    successful size.

    Args:
        env: FactoriaX environment instance.
        params: Environment parameters.
        cap: Maximum batch size to try.

    Returns:
        Largest batch size that succeeded.
    """
    reset_fn = jax.vmap(env.reset_env, in_axes=(0, None))
    last_good = 256
    bs = 256
    while bs <= cap:
        try:
            keys = jax.random.split(jax.random.PRNGKey(0), bs)
            _, states = reset_fn(keys, params)
            jax.block_until_ready(states)

            # Also try a few steps to check for fragmentation.
            step_fn = jax.jit(
                jax.vmap(env.step_env, in_axes=(0, 0, None, None)),
            )
            step_keys = jax.random.split(jax.random.PRNGKey(1), bs)
            for _ in range(3):
                _ = step_fn(step_keys, states, int(Action.NOOP), params)
                jax.block_until_ready(_)

            last_good = bs
            logger.info(
                "  batch=%d OK (map %dx%d)",
                bs, params.map_width, params.map_height,
            )
            bs *= 2
        except (jax.errors.JaxRuntimeError, RuntimeError):
            logger.info(
                "  batch=%d OOM (map %dx%d), using %d",
                bs, params.map_width, params.map_height, last_good,
            )
            break
    return last_good


def _profile_config(
    env: object,
    params: EnvParams,
    batch_size: int,
    duration: float,
    trace_dir: str | None,
) -> dict:
    """Profile one map_size + batch_size configuration.

    Args:
        env: FactoriaX environment instance.
        params: Environment parameters.
        batch_size: Number of parallel environments.
        duration: Seconds to measure throughput.
        trace_dir: Path for Perfetto trace, or None to skip.

    Returns:
        Dict with profiling results.
    """
    ms = params.map_width
    bs = batch_size
    device = jax.local_devices()[0]

    reset_fn = jax.vmap(env.reset_env, in_axes=(0, None))
    step_fn = jax.jit(
        jax.vmap(env.step_env, in_axes=(0, 0, None, None)),
    )

    keys = jax.random.split(jax.random.PRNGKey(0), bs)
    _, states = reset_fn(keys, params)
    action = int(Action.NOOP)
    step_keys = jax.random.split(jax.random.PRNGKey(1), bs)

    # Warmup JIT.
    t_warmup = time.perf_counter()
    _ = step_fn(step_keys, states, action, params)
    jax.block_until_ready(_)
    warmup_s = time.perf_counter() - t_warmup

    # Memory stats after warmup.
    stats = device.memory_stats()
    mem_in_use = stats.get("bytes_in_use", 0)
    mem_peak = stats.get("peak_bytes_in_use", 0)
    mem_limit = stats.get("bytes_limit", 0)

    logger.info(
        "  %dx%d b=%d: warmup=%.1fs, mem=%dMB/%dMB",
        ms, ms, bs, warmup_s,
        mem_in_use // (1024 * 1024),
        mem_limit // (1024 * 1024),
    )

    # Perfetto trace (short burst).
    if trace_dir is not None:
        Path(trace_dir).mkdir(parents=True, exist_ok=True)
        logger.info("  Capturing Perfetto trace -> %s", trace_dir)
        with jax.profiler.trace(
            trace_dir, create_perfetto_trace=True,
        ):
            for _ in range(5):
                _ = step_fn(step_keys, states, action, params)
                jax.block_until_ready(_)

    # Throughput measurement.
    total = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < duration:
        _ = step_fn(step_keys, states, action, params)
        jax.block_until_ready(_)
        total += bs
    elapsed = time.perf_counter() - t0
    sps = total / elapsed

    logger.info(
        "  %dx%d b=%d: %s steps/s (%d steps in %.1fs)",
        ms, ms, bs, f"{sps:,.0f}", total, elapsed,
    )

    return {
        "map_size": ms,
        "batch_size": bs,
        "steps_per_second": round(sps),
        "memory_bytes_in_use": mem_in_use,
        "memory_peak_bytes": mem_peak,
        "memory_limit_bytes": mem_limit,
        "warmup_seconds": round(warmup_s, 2),
        "measure_seconds": round(elapsed, 2),
        "total_steps": total,
        "trace_dir": trace_dir,
    }


def main() -> None:
    """Run the profiler across map sizes and save results."""
    parser = argparse.ArgumentParser(
        description="Profile FactoriaX step throughput and GPU usage.",
    )
    parser.add_argument(
        "--map-sizes",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64],
        help="Map sizes to profile (default: 8 16 32 64).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Seconds per throughput measurement (default: 10).",
    )
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="Skip Perfetto trace capture.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="profiled.json",
        help="Output JSON file (default: profiled.json).",
    )
    args = parser.parse_args()

    commit = _get_commit_hash()
    gpu = _get_gpu_name()
    ts = datetime.now(tz=UTC).isoformat(timespec="seconds")

    logger.info("Commit: %s", commit)
    logger.info("GPU: %s", gpu)
    logger.info("Map sizes: %s", args.map_sizes)
    logger.info("Duration: %.0fs per config", args.duration)

    env, _ = make_factoriax_env()
    configs: list[dict] = []

    for ms in args.map_sizes:
        params = EnvParams(map_width=ms, map_height=ms)

        logger.info("Finding max batch for %dx%d...", ms, ms)
        max_bs = _find_max_batch(env, params)

        # Try profiling at max batch; fall back to half on OOM.
        bs = max_bs
        while bs >= 256:
            try:
                trace_dir = None
                if not args.no_trace:
                    trace_dir = f"profiles/{commit}/{ms}x{ms}_b{bs}"
                result = _profile_config(
                    env, params, bs, args.duration, trace_dir,
                )
                configs.append(result)
                break
            except (jax.errors.JaxRuntimeError, RuntimeError):
                logger.info(
                    "  OOM during profiling at b=%d, retrying b=%d",
                    bs, bs // 2,
                )
                bs //= 2
        else:
            logger.error("  Could not profile %dx%d at any batch size", ms, ms)

    output = {
        "commit": commit,
        "date": ts,
        "gpu": gpu,
        "configs": configs,
    }

    out_path = Path(args.output)
    out_path.write_bytes(
        orjson.dumps(output, option=orjson.OPT_INDENT_2),
    )
    logger.info("Results saved to %s", out_path)

    # Summary table.
    print(f"\n{'=' * 70}")
    print(f"  Profile: {commit} @ {ts}")
    print(f"  GPU: {gpu}")
    print(f"{'=' * 70}")
    print(
        f"{'Map':>6} {'Batch':>7} {'Steps/s':>12} "
        f"{'Mem MB':>8} {'Peak MB':>8} {'Warmup':>7}"
    )
    print(f"{'-' * 6} {'-' * 7} {'-' * 12} {'-' * 8} {'-' * 8} {'-' * 7}")
    for c in configs:
        print(
            f"{c['map_size']:>6} {c['batch_size']:>7} "
            f"{c['steps_per_second']:>12,} "
            f"{c['memory_bytes_in_use'] // (1024 * 1024):>8} "
            f"{c['memory_peak_bytes'] // (1024 * 1024):>8} "
            f"{c['warmup_seconds']:>6.1f}s"
        )


if __name__ == "__main__":
    main()
