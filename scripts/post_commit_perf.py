"""Post-commit performance benchmark for FactoriaX.

Measures vmapped step_env throughput across a matrix of map sizes
and batch sizes using lax.scan for accurate, step-count-based
measurement. Each configuration runs exactly 100 environment steps
per batch element.

Includes an addition baseline that replaces game logic with
jax.tree.map(x+1) on the same state pytree, giving a practical
upper bound on dispatch + memory bandwidth.

Results are logged to WandB and a 7-commit historical summary is
printed to stdout.

Usage:
    uv run python scripts/post_commit_perf.py
    uv run python scripts/post_commit_perf.py --no-wandb
    uv run python scripts/post_commit_perf.py --no-history
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import threading
import time

import jax
from jax import lax, random

from factoriax.engine.constants import NUM_ACTIONS
from factoriax.engine.envs.factoriax_env import FactoriaXEnv
from factoriax.engine.state import EnvParams

logger = logging.getLogger(__name__)

MAP_SIZES = [8, 32, 128]
BATCH_SIZES = [256, 1024, 4096]
STEPS_PER_MAP: dict[int, int] = {8: 100, 32: 100, 128: 20}


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
    return str(jax.devices()[0])


class _GpuStateSampler:
    """Background sampler that captures the GPU's peak load state.

    Polling ``nvidia-smi`` once after the benchmark is misleading:
    the driver drops back to P8 (idle) within milliseconds of the
    compute kernels finishing, so the snapshot nearly always reads
    P8 regardless of what the benchmark actually ran under. Instead,
    this sampler polls in a background thread at 500 ms cadence and
    tracks the highest-performance state observed (smallest P-number)
    plus the maximum SM clock / power / temperature. Those are the
    numbers that matter when comparing run-to-run throughput.
    """

    def __init__(self, interval_sec: float = 0.5) -> None:
        self._interval = interval_sec
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._best_pstate_num: int | None = None  # lower = faster
        self._peak: dict[str, float] = {}

    def _poll_once(self) -> dict[str, str] | None:
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=pstate,clocks.current.sm,"
                    "power.draw,temperature.gpu,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=2,
            ).strip()
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            return None
        parts = [p.strip() for p in out.split("\n", 1)[0].split(",")]
        if len(parts) < 5:
            return None
        return {
            "pstate": parts[0],
            "sm_clock_mhz": parts[1],
            "power_w": parts[2],
            "temp_c": parts[3],
            "utilization_pct": parts[4],
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = self._poll_once()
            if sample is not None:
                # P-state is "P" followed by a digit; lower digit = better.
                try:
                    pnum = int(sample["pstate"].lstrip("pP"))
                except ValueError:
                    pnum = 9
                if self._best_pstate_num is None or pnum < self._best_pstate_num:
                    self._best_pstate_num = pnum
                for key in (
                    "sm_clock_mhz",
                    "power_w",
                    "temp_c",
                    "utilization_pct",
                ):
                    try:
                        v = float(sample[key])
                    except (ValueError, KeyError):
                        continue
                    if v > self._peak.get(key, float("-inf")):
                        self._peak[key] = v
            self._stop.wait(self._interval)

    def __enter__(self) -> _GpuStateSampler:
        self._thread = threading.Thread(
            target=self._run,
            name="gpu-state-sampler",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def summary(self) -> dict[str, str]:
        """Return the peak-load GPU state as strings, ``'unknown'`` if unseen.

        On some consumer GPUs ``pstate`` sticks at P8 regardless of load
        (NVIDIA driver quirk). ``utilization_pct`` and ``sm_clock_mhz``
        are more reliable load indicators — log both so the wandb entry
        is interpretable on any hardware.
        """
        if self._best_pstate_num is None:
            return {
                "pstate": "unknown",
                "sm_clock_mhz": "unknown",
                "power_w": "unknown",
                "temp_c": "unknown",
                "utilization_pct": "unknown",
            }
        return {
            "pstate": f"P{self._best_pstate_num}",
            "sm_clock_mhz": (
                f"{self._peak['sm_clock_mhz']:.0f}"
                if "sm_clock_mhz" in self._peak
                else "unknown"
            ),
            "power_w": (
                f"{self._peak['power_w']:.1f}" if "power_w" in self._peak else "unknown"
            ),
            "temp_c": (
                f"{self._peak['temp_c']:.0f}" if "temp_c" in self._peak else "unknown"
            ),
            "utilization_pct": (
                f"{self._peak['utilization_pct']:.0f}"
                if "utilization_pct" in self._peak
                else "unknown"
            ),
        }


def _get_gpu_perf_state() -> dict[str, str]:
    """Query nvidia-smi for GPU perf state, SM clock, power, temperature.

    The P-state (P0-P8) is the most useful: P0 is "max performance",
    P8 is "min / idle". A benchmark that thinks it's measuring peak
    throughput while the GPU is sitting at P5 is measuring the wrong
    thing — so we log it alongside the throughput numbers so the wandb
    history is interpretable even when the GPU clocked down between
    runs.

    Returns:
        Dict with keys pstate, sm_clock_mhz, power_w, temp_c. Values
        default to "unknown" when nvidia-smi isn't available.
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=pstate,clocks.current.sm,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
        # First line / first GPU only.
        line = out.split("\n", 1)[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            return {
                "pstate": parts[0],
                "sm_clock_mhz": parts[1],
                "power_w": parts[2],
                "temp_c": parts[3],
            }
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        pass
    return {
        "pstate": "unknown",
        "sm_clock_mhz": "unknown",
        "power_w": "unknown",
        "temp_c": "unknown",
    }


def _measure_config(
    env: FactoriaXEnv,
    map_size: int,
    batch_size: int,
    num_steps: int,
) -> dict[str, float] | None:
    """Measure throughput for one (map_size, batch_size) config.

    Uses lax.scan to fuse steps into a single XLA call.
    Warmup (JIT compilation) is measured separately.

    Args:
        env: FactoriaX environment instance.
        map_size: Square map side length.
        batch_size: Number of parallel environments.
        num_steps: Steps per environment to run.

    Returns:
        Dict with steps_per_sec, jit_warmup_sec, vram_bytes,
        vram_peak_bytes, vram_limit_bytes. None if OOM.
    """
    try:
        params = EnvParams(
            map_width=map_size,
            map_height=map_size,
            num_players=1,
        )

        keys = random.split(random.PRNGKey(0), batch_size)
        reset_fn = jax.vmap(env.reset_env, in_axes=(0, None))
        _, states = reset_fn(keys, params)

        vmap_step = jax.vmap(env.step_env, in_axes=(0, 0, 0, None))

        def scan_body(
            carry: tuple[object, jax.Array],
            _: None,
        ) -> tuple[tuple[object, jax.Array], None]:
            """One step inside lax.scan."""
            state, rng = carry
            rng, k_action, k_step = random.split(rng, 3)
            step_keys = random.split(k_step, batch_size)
            actions = random.randint(
                k_action,
                (batch_size,),
                0,
                NUM_ACTIONS,
            )
            _, state, _, _, _ = vmap_step(
                step_keys,
                state,
                actions,
                params,
            )
            return (state, rng), None

        scan_fn = jax.jit(
            lambda s, r: lax.scan(
                scan_body,
                (s, r),
                None,
                length=num_steps,
            ),
        )

        # JIT warmup.
        t0 = time.perf_counter()
        (states, _), _ = scan_fn(states, random.PRNGKey(99))
        jax.block_until_ready(jax.tree.leaves(states))
        jit_warmup_sec = time.perf_counter() - t0

        # VRAM measurement.
        device = jax.local_devices()[0]
        mem = device.memory_stats() or {}
        vram_bytes = mem.get("bytes_in_use", 0)
        vram_peak_bytes = mem.get("peak_bytes_in_use", 0)
        vram_limit_bytes = mem.get("bytes_limit", 0)

        # Throughput measurement (second call, no JIT).
        t0 = time.perf_counter()
        (states, _), _ = scan_fn(states, random.PRNGKey(42))
        jax.block_until_ready(jax.tree.leaves(states))
        elapsed = time.perf_counter() - t0
        total_steps = num_steps * batch_size
        steps_per_sec = total_steps / elapsed

        return {
            "steps_per_sec": steps_per_sec,
            "jit_warmup_sec": jit_warmup_sec,
            "vram_bytes": vram_bytes,
            "vram_peak_bytes": vram_peak_bytes,
            "vram_limit_bytes": vram_limit_bytes,
        }
    except (jax.errors.JaxRuntimeError, RuntimeError) as exc:
        logger.warning(
            "SKIPPED map=%dx%d batch=%d: %s",
            map_size,
            map_size,
            batch_size,
            exc,
        )
        return None


def _measure_baseline(
    env: FactoriaXEnv,
    map_size: int,
    batch_size: int,
    num_steps: int,
) -> dict[str, float] | None:
    """Measure addition baseline for one config.

    Same scan structure as the real benchmark but replaces game
    logic with jax.tree.map(x+1) on the real state pytree.

    Args:
        env: FactoriaX environment instance.
        map_size: Square map side length.
        batch_size: Number of parallel environments.
        num_steps: Steps per environment to run.

    Returns:
        Dict with steps_per_sec and jit_warmup_sec. None if OOM.
    """
    try:
        params = EnvParams(
            map_width=map_size,
            map_height=map_size,
            num_players=1,
        )

        keys = random.split(random.PRNGKey(0), batch_size)
        reset_fn = jax.vmap(env.reset_env, in_axes=(0, None))
        _, states = reset_fn(keys, params)

        # Vmap the addition to match the real benchmark's
        # scan-over-vmap structure. Use lax.full_like so the 1
        # constant matches each leaf's dtype (avoids bool->int32
        # promotion that breaks scan type matching).
        vmap_add = jax.vmap(
            lambda s: jax.tree.map(
                lambda x: x + lax.full_like(x, 1),
                s,
            ),
        )

        def baseline_body(
            carry: tuple[object, jax.Array],
            _: None,
        ) -> tuple[tuple[object, jax.Array], None]:
            """One addition step inside lax.scan."""
            state, rng = carry
            rng, _ = random.split(rng)
            state = vmap_add(state)
            return (state, rng), None

        baseline_fn = jax.jit(
            lambda s, r: lax.scan(
                baseline_body,
                (s, r),
                None,
                length=num_steps,
            ),
        )

        # JIT warmup.
        t0 = time.perf_counter()
        (states, _), _ = baseline_fn(states, random.PRNGKey(99))
        jax.block_until_ready(jax.tree.leaves(states))
        jit_warmup_sec = time.perf_counter() - t0

        # Throughput measurement.
        t0 = time.perf_counter()
        (states, _), _ = baseline_fn(states, random.PRNGKey(42))
        jax.block_until_ready(jax.tree.leaves(states))
        elapsed = time.perf_counter() - t0
        total_steps = num_steps * batch_size
        steps_per_sec = total_steps / elapsed

        return {
            "baseline_steps_per_sec": steps_per_sec,
            "baseline_jit_warmup_sec": jit_warmup_sec,
        }
    except (jax.errors.JaxRuntimeError, RuntimeError) as exc:
        logger.warning(
            "SKIPPED baseline map=%dx%d batch=%d: %s",
            map_size,
            map_size,
            batch_size,
            exc,
        )
        return None


def run_benchmark() -> list[dict]:
    """Run the full benchmark matrix (env + baseline).

    Returns:
        List of result dicts, one per configuration.
    """
    env = FactoriaXEnv()
    results: list[dict] = []

    for ms in MAP_SIZES:
        ns = STEPS_PER_MAP[ms]
        for bs in BATCH_SIZES:
            print(
                f"  Benchmarking {ms}x{ms} batch={bs} ({ns} steps)...",
                end="",
                flush=True,
            )
            env_result = _measure_config(env, ms, bs, ns)
            base_result = _measure_baseline(env, ms, bs, ns)

            if env_result is None:
                print(" SKIPPED (OOM)")
                results.append(
                    {
                        "map_size": ms,
                        "batch_size": bs,
                        "skipped": True,
                    }
                )
                continue

            row: dict = {
                "map_size": ms,
                "batch_size": bs,
                "skipped": False,
                **env_result,
            }

            if base_result is not None:
                row.update(base_result)
                ceiling = base_result["baseline_steps_per_sec"]
                if ceiling > 0:
                    row["pct_of_ceiling"] = env_result["steps_per_sec"] / ceiling * 100
                else:
                    row["pct_of_ceiling"] = 0.0
            else:
                row["baseline_steps_per_sec"] = 0.0
                row["baseline_jit_warmup_sec"] = 0.0
                row["pct_of_ceiling"] = 0.0

            results.append(row)
            print(
                f" {env_result['steps_per_sec']:,.0f} sps"
                f" ({row['pct_of_ceiling']:.1f}% ceiling)"
            )

    return results


def _print_results(
    results: list[dict],
    info: dict[str, str],
    device: str,
    gpu_state: dict[str, str],
) -> None:
    """Print benchmark results table to stdout.

    Args:
        results: List of result dicts from run_benchmark.
        info: Commit metadata.
        device: JAX device name.
        gpu_state: Dict from :func:`_get_gpu_perf_state`. The P-state
            clarifies whether the GPU was clocked up during the run.
    """
    print(f"\n{'=' * 80}")
    print(f"  Commit: {info['commit']} - {info['message']}")
    print(f"  Device: {device}")
    print(
        f"  GPU:    pstate={gpu_state['pstate']}  "
        f"util={gpu_state['utilization_pct']}%  "
        f"sm_clock={gpu_state['sm_clock_mhz']} MHz  "
        f"power={gpu_state['power_w']} W  "
        f"temp={gpu_state['temp_c']} C"
    )
    steps_desc = ", ".join(f"{ms}x{ms}={ns}" for ms, ns in STEPS_PER_MAP.items())
    print(f"  Steps per env: {steps_desc}")
    print(f"{'=' * 80}")

    # Environment throughput table.
    print("\n  Environment")
    print(
        f"  {'Map':>7}  {'Batch':>6}  {'Steps/s':>12}  "
        f"{'Ceiling%':>8}  {'Warmup':>7}  {'VRAM MB':>8}"
    )
    print(f"  {'-' * 7}  {'-' * 6}  {'-' * 12}  {'-' * 8}  {'-' * 7}  {'-' * 8}")
    for r in results:
        if r.get("skipped"):
            print(
                f"  {r['map_size']:>3}x{r['map_size']:<3}  "
                f"{r['batch_size']:>6}  "
                f"{'SKIPPED':>12}"
            )
            continue
        vram_mb = r.get("vram_bytes", 0) // (1024 * 1024)
        print(
            f"  {r['map_size']:>3}x{r['map_size']:<3}  "
            f"{r['batch_size']:>6}  "
            f"{r['steps_per_sec']:>12,.0f}  "
            f"{r.get('pct_of_ceiling', 0):>7.1f}%  "
            f"{r.get('jit_warmup_sec', 0):>6.1f}s  "
            f"{vram_mb:>8}"
        )

    # Addition baseline table.
    baseline_rows = [
        r
        for r in results
        if not r.get("skipped") and r.get("baseline_steps_per_sec", 0) > 0
    ]
    if baseline_rows:
        print("\n  Addition baseline (pytree x+1, same vmap/scan)")
        print(f"  {'Map':>7}  {'Batch':>6}  {'Steps/s':>12}  {'Warmup':>7}")
        print(f"  {'-' * 7}  {'-' * 6}  {'-' * 12}  {'-' * 7}")
        for r in baseline_rows:
            print(
                f"  {r['map_size']:>3}x{r['map_size']:<3}  "
                f"{r['batch_size']:>6}  "
                f"{r['baseline_steps_per_sec']:>12,.0f}  "
                f"{r.get('baseline_jit_warmup_sec', 0):>6.1f}s"
            )


def _log_wandb(
    results: list[dict],
    info: dict[str, str],
    device: str,
    gpu_state: dict[str, str],
) -> None:
    """Log benchmark results to WandB.

    Args:
        results: List of result dicts.
        info: Commit metadata.
        device: JAX device name.
        gpu_state: Snapshot from :func:`_get_gpu_perf_state`. Logged as
            both config (for filtering) and summary (for quick scan in
            the run list).
    """
    try:
        import wandb  # type: ignore[import-untyped]
    except ImportError:
        logger.error("wandb not found. Install with: uv add wandb")
        return

    run = wandb.init(
        project="factoriax-benchmarks",
        name=f"{info['commit']} - {info['message'][:50]}",
        tags=[
            info["commit"],
            device.split(":")[0],
            "post-commit",
            f"gpu_{gpu_state['pstate']}",
        ],
        config={
            "commit_hash": info["commit"],
            "commit_message": info["message"],
            "commit_date": info["date"],
            "device": device,
            "gpu_pstate": gpu_state["pstate"],
            "gpu_utilization_pct": gpu_state["utilization_pct"],
            "gpu_sm_clock_mhz": gpu_state["sm_clock_mhz"],
            "gpu_power_w": gpu_state["power_w"],
            "gpu_temp_c": gpu_state["temp_c"],
            "map_sizes": MAP_SIZES,
            "batch_sizes": BATCH_SIZES,
            "steps_per_map": STEPS_PER_MAP,
        },
    )

    columns = [
        "commit_hash",
        "map_size",
        "batch_size",
        "steps_per_sec",
        "baseline_steps_per_sec",
        "pct_of_ceiling",
        "jit_warmup_sec",
        "vram_bytes",
        "vram_peak_bytes",
    ]
    table = wandb.Table(columns=columns)
    for r in results:
        if r.get("skipped"):
            continue
        table.add_data(
            info["commit"],
            r["map_size"],
            r["batch_size"],
            r["steps_per_sec"],
            r.get("baseline_steps_per_sec", 0),
            r.get("pct_of_ceiling", 0),
            r.get("jit_warmup_sec", 0),
            r.get("vram_bytes", 0),
            r.get("vram_peak_bytes", 0),
        )
    run.log({"perf_matrix": table})

    run.summary["commit_hash"] = info["commit"]
    run.summary["gpu_pstate"] = gpu_state["pstate"]
    run.summary["gpu_utilization_pct"] = gpu_state["utilization_pct"]
    run.summary["gpu_sm_clock_mhz"] = gpu_state["sm_clock_mhz"]
    for r in results:
        if r.get("skipped"):
            continue
        ms = r["map_size"]
        bs = r["batch_size"]
        key = f"map{ms}_batch{bs}"
        run.summary[f"sps/{key}"] = r["steps_per_sec"]
        run.summary[f"baseline/{key}"] = r.get(
            "baseline_steps_per_sec",
            0,
        )
        run.summary[f"warmup/{key}"] = r.get("jit_warmup_sec", 0)
        run.summary[f"vram/{key}"] = r.get("vram_bytes", 0)

    successful = [
        r["steps_per_sec"]
        for r in results
        if not r.get("skipped") and r["steps_per_sec"] > 0
    ]
    if successful:
        run.summary["peak_steps_per_sec"] = max(successful)

    run.finish()
    print(f"\nwandb run: {run.url}")


def _print_history() -> None:
    """Query WandB for the last 7 post-commit runs and print summary."""
    try:
        import wandb  # type: ignore[import-untyped]
    except ImportError:
        logger.error("wandb not found, skipping history.")
        return

    try:
        api = wandb.Api()
        runs = api.runs(
            "factoriax-benchmarks",
            filters={"tags": {"$in": ["post-commit"]}},
            order="-created_at",
            per_page=7,
        )
        run_list = list(runs)
    except Exception as exc:
        logger.error("Failed to fetch WandB history: %s", exc)
        return

    if not run_list:
        print("\nNo historical runs found in WandB.")
        return

    print(f"\n{'=' * 90}")
    print(f"  Performance history (last {len(run_list)} commits)")
    print(f"{'=' * 90}")

    # Environment throughput history.
    print("\n  Environment")
    print(
        f"  {'Commit':>8}  {'Map':>7}  {'Batch':>6}  "
        f"{'Steps/s':>12}  {'Ceiling%':>8}  "
        f"{'Warmup':>7}  {'VRAM MB':>8}"
    )
    print(
        f"  {'-' * 8}  {'-' * 7}  {'-' * 6}  "
        f"{'-' * 12}  {'-' * 8}  {'-' * 7}  {'-' * 8}"
    )
    for run in run_list:
        commit = run.summary.get(
            "commit_hash",
            run.config.get("commit_hash", "?"),
        )
        for ms in MAP_SIZES:
            for bs in BATCH_SIZES:
                key = f"map{ms}_batch{bs}"
                sps = run.summary.get(f"sps/{key}")
                if sps is None:
                    continue
                baseline = run.summary.get(f"baseline/{key}", 0)
                warmup = run.summary.get(f"warmup/{key}", 0)
                vram = run.summary.get(f"vram/{key}", 0)
                pct = (sps / baseline * 100) if baseline > 0 else 0
                vram_mb = int(vram) // (1024 * 1024) if vram else 0
                print(
                    f"  {commit:>8}  {ms:>3}x{ms:<3}  {bs:>6}  "
                    f"{sps:>12,.0f}  {pct:>7.1f}%  "
                    f"{warmup:>6.1f}s  {vram_mb:>8}"
                )

    # Addition baseline history.
    print("\n  Addition baseline (pytree x+1, same vmap/scan)")
    print(f"  {'Commit':>8}  {'Map':>7}  {'Batch':>6}  {'Steps/s':>12}")
    print(f"  {'-' * 8}  {'-' * 7}  {'-' * 6}  {'-' * 12}")
    for run in run_list:
        commit = run.summary.get(
            "commit_hash",
            run.config.get("commit_hash", "?"),
        )
        for ms in MAP_SIZES:
            for bs in BATCH_SIZES:
                key = f"map{ms}_batch{bs}"
                baseline = run.summary.get(f"baseline/{key}")
                if baseline is None or baseline <= 0:
                    continue
                print(f"  {commit:>8}  {ms:>3}x{ms:<3}  {bs:>6}  {baseline:>12,.0f}")


def main() -> None:
    """CLI entry point for the post-commit benchmark."""
    parser = argparse.ArgumentParser(
        description="Post-commit performance benchmark for FactoriaX.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Print results to stdout only, skip WandB logging.",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Skip printing the historical summary.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    info = _get_commit_info()
    device = _get_device_name()

    print("FactoriaX Post-Commit Benchmark")
    print(f"Commit: {info['commit']} - {info['message']}")
    print(f"Device: {device}")
    print(
        f"Matrix: {len(MAP_SIZES)}x{len(BATCH_SIZES)} configs, "
        f"{STEPS_PER_MAP} steps per map"
    )
    print()

    with _GpuStateSampler() as sampler:
        results = run_benchmark()
    gpu_state = sampler.summary()
    _print_results(results, info, device, gpu_state)

    if not args.no_wandb:
        _log_wandb(results, info, device, gpu_state)

    if not args.no_history:
        _print_history()


if __name__ == "__main__":
    main()
