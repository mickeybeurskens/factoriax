"""Throughput benchmark: engine steps/sec across map sizes and batch sizes.

Sweeps ``map_size`` x ``num_envs`` x ``entity_multiplier`` and times
a JIT-compiled, vmapped, ``lax.scan`` driven inner loop of the bare
FactoriaX engine on whatever JAX device is available. Reports the
median steps/sec across many trials per configuration and logs every
individual trial to Weights & Biases so the dispersion is auditable,
not just summarised.

The entity-limit dimension matters because the agent literally
cannot place machines beyond it: sweeping at 0.5x, 1x, and 2x the
engine's standard cap (``max(64, map_area // 4)``) makes both the
throughput cost AND the placement headroom visible in §3.5.

This script lives in the FactoriaX engine repo (not the paper) so it
is available on Snellius without checking out the paper. The same
script runs locally on a laptop GPU and on a Snellius A100 node via
``scripts/snellius_submit_throughput.sh``. The ``--device-label``
argument tags the output so multiple devices' measurements can live
side by side in ``paper/data/`` for the figure builder.

Measurement methodology
-----------------------

* **Engine under test**: a bare :class:`FactoriaXEnv` (not a
  registered scenario) with ``obs="x_ray_global"``, easy-rocket's
  achievement hook (``easy_rocket_conditions``), and easy-rocket's
  reward function (``easy_rocket_reward``). All three are attached
  because a real training pipeline pays each of their costs every
  step; running without any of them would inflate the headline
  number.

* **DCE protection**: the scan body folds the observation, reward,
  and done outputs of ``step_env`` into a scalar sink that is part
  of the scan carry and returned alongside the final state. Without
  this, XLA would dead-code-eliminate those computations (the
  outputs are otherwise unused after the step), and the headline
  throughput would only reflect the state-transition cost. The sink
  is discarded after ``block_until_ready``; its only job is to keep
  XLA honest.

* **For each (map_size, batch_size)**:

  1. **Calibrate** ``inner_steps`` (unless ``--inner-steps`` is
     given). Build a small probe scan at
     ``--probe-inner-steps`` (default 64), warm it once to absorb
     the JIT compile, time a second call, derive a per-tick wall
     estimate, and choose ``inner_steps`` so each timed trial
     targets ``--target-trial-seconds`` (default 0.3 s). This keeps
     small batch sizes out of the CPU-scheduling noise floor and
     stops large batch sizes from spending minutes per trial.

  2. **Startup** is timed separately. The first call to the
     calibrated scan pays the JIT compile and the first kernel
     launch; that wall time is reported as ``startup_seconds`` and
     is NOT folded into the throughput numbers.

  3. **Trials**: ``--trials`` (default 20) runs at the calibrated
     ``inner_steps`` with the JIT cached. Each trial is a fresh
     ``reset_env`` followed by ``inner_steps`` batched random ticks;
     wall time covers everything from reset through the final
     ``block_until_ready``. Steps/sec for a trial is
     ``num_envs * inner_steps / wall``. Every trial sample is
     stored in the JSON and logged individually to wandb.

* **Optional addition baseline** (``--baseline``): replaces the real
  step with ``jax.tree.map(x + 1)`` on the state pytree. Same
  vmap/scan structure, no engine logic. Gives a practical dispatch +
  memory-bandwidth ceiling; the figure can show real throughput as
  "% of ceiling".

* **GPU state sampling**: a background thread polls ``nvidia-smi``
  every 500 ms and records the peak P-state, SM clock, power,
  temperature and utilisation reached during the run. Useful for
  detecting silent clock-down regressions between laptop runs.

Output schema (``./throughput_<device-label>.json`` by default)::

    {
      "device_label":            "laptop",
      "device_full":             "NVIDIA GeForce RTX 3050 6GB Laptop GPU",
      "obs":                     "x_ray_global",
      "achievement_fn":          "easy_rocket_conditions",
      "trials":                  50,
      "target_trial_seconds":    1.0,
      "override_inner_steps":    null,
      "jax_version":             "0.4.x",
      "gpu_state":               {"pstate": "P0", ...},
      "measurements": [
        {
          "device":               "laptop",
          "device_full":          "NVIDIA GeForce RTX 3050 6GB Laptop GPU",
          "map_size":             16,
          "batch_size":           256,
          "inner_steps":          4096,
          "startup_seconds":      5.2,
          "skipped":              false,
          "steps_per_sec":        1.21e5,
          "steps_per_sec_median": 1.21e5,
          "steps_per_sec_mean":   1.20e5,
          "steps_per_sec_std":    3.4e3,
          "steps_per_sec_min":    1.15e5,
          "steps_per_sec_max":    1.25e5,
          "steps_per_sec_p05":    1.17e5,
          "steps_per_sec_p95":    1.24e5,
          "trial_samples":        [1.20e5, 1.21e5, ..., 1.22e5],
          "baseline_steps_per_sec": 1.5e6,
          "baseline_startup_seconds": 3.0
        },
        ...
      ]
    }

The paper-side script ``paper/scripts/experiments/throughput.py``
ingests these per-device JSON files (downloaded from wandb) and
merges them into the consolidated ``paper/data/throughput.json``
that the figure builder reads.
"""

from __future__ import annotations

import argparse
import logging
import statistics
import subprocess
import threading
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import orjson
from jax import lax, random

from factoriax.engine.constants import NUM_ACTIONS
from factoriax.engine.envs.factoriax_env import FactoriaXEnv
from factoriax.engine.scenarios.easy_rocket import (
    easy_rocket_conditions,
    easy_rocket_reward,
)
from factoriax.engine.state import EnvParams

logger = logging.getLogger("factoriax.throughput_bench")

#: Default map sizes to sweep (square; ``map_width = map_height``).
_DEFAULT_MAP_SIZES: tuple[int, ...] = (16, 32, 128)
#: Default batch sizes; large entries are skipped if they OOM.
_DEFAULT_NUM_ENVS: tuple[int, ...] = (64, 256, 1024, 4096)
#: Default entity-limit multipliers relative to the engine standard.
#: A multiplier of 1.0 keeps ``EnvParams.max_machines = 0`` (the engine
#: derives ``max(64, map_area // 4)``); other multipliers scale that
#: implicit default. The standard is on the agent's side of fairness:
#: a hard cap on how many machines can coexist matters because the
#: agent literally cannot place beyond it.
_DEFAULT_ENTITY_MULTIPLIERS: tuple[float, ...] = (0.5, 1.0, 2.0)
#: Probe scan length used for the autoscale calibration.
_DEFAULT_PROBE_INNER_STEPS: int = 64


def _standard_max_machines(map_size: int) -> int:
    """Match :meth:`EnvParams.effective_max_machines` for a square map."""
    return max(64, map_size * map_size // 4)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device-label",
        required=True,
        help=(
            'Short tag for the device, e.g. "laptop" or "a100". Used '
            "for the output filename and the device column in the JSON."
        ),
    )
    parser.add_argument(
        "--map-sizes",
        type=int,
        nargs="+",
        default=None,
        help=f"Square map sizes to sweep (default {list(_DEFAULT_MAP_SIZES)}).",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        nargs="+",
        default=None,
        help=f"Batch sizes to sweep (default {list(_DEFAULT_NUM_ENVS)}).",
    )
    parser.add_argument(
        "--entity-multipliers",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Entity-limit multipliers relative to the engine standard "
            "(max(64, map_area // 4)). Default "
            f"{list(_DEFAULT_ENTITY_MULTIPLIERS)} so the figure can "
            "show throughput at half, standard, and double the cap."
        ),
    )
    parser.add_argument(
        "--inner-steps",
        type=int,
        default=None,
        help=(
            "Fixed inner_steps. If unset, autoscale to target "
            "--target-trial-seconds per trial."
        ),
    )
    parser.add_argument(
        "--probe-inner-steps",
        type=int,
        default=_DEFAULT_PROBE_INNER_STEPS,
        help="Probe scan length used during autoscale calibration.",
    )
    parser.add_argument(
        "--target-trial-seconds",
        type=float,
        default=0.3,
        help=(
            "Autoscale target wall time per trial. The default is "
            "set short enough that the full sweep finishes in a few "
            "minutes; raise to 1.0 for a slower, more precise run."
        ),
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help=(
            "Timing trials per (map_size, batch_size). Bump to 50 "
            "for tighter dispersion estimates."
        ),
    )
    parser.add_argument(
        "--obs",
        default="x_ray_global",
        help="Observation variant for the bare env.",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help=(
            "Also measure the addition baseline (pytree x+1) per "
            "config; gives a dispatch+memory ceiling for the real "
            "engine throughput."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="PRNG seed.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSON path. Defaults to "
            "./throughput_<device-label>.json in the current "
            "working directory."
        ),
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging (useful for smoke tests).",
    )
    parser.add_argument(
        "--wandb-project",
        default="factoriax_throughput",
        help="wandb project name.",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Override wandb run name (default: throughput_<device-label>).",
    )
    return parser.parse_args()


def _detect_device_full() -> str:
    devices = jax.devices()
    if not devices:
        return "no-devices"
    return str(devices[0])


class _GpuStateSampler:
    """Background sampler that captures the GPU's peak load state.

    Adapted from :mod:`scripts.post_commit_perf`. Polling
    ``nvidia-smi`` once after a benchmark is misleading because the
    driver drops back to P8 within milliseconds of the kernels
    finishing. This sampler polls in a background thread at 500 ms
    cadence and tracks the highest-performance state observed.
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
            target=self._run, name="gpu-state-sampler", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def summary(self) -> dict[str, str]:
        if self._best_pstate_num is None:
            return {
                k: "unknown"
                for k in (
                    "pstate",
                    "sm_clock_mhz",
                    "power_w",
                    "temp_c",
                    "utilization_pct",
                )
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


def _build_step_scan(
    env: FactoriaXEnv,
    params: EnvParams,
    batch_size: int,
    inner_steps: int,
):
    """JIT a ``vmap+lax.scan`` over ``inner_steps`` engine ticks.

    The scan body folds the observation, reward, and done outputs of
    ``step_env`` into a scalar sink that is part of the scan carry.
    Without this, XLA would dead-code-eliminate the observation /
    reward / done computations because none of them affect the final
    returned ``state``, and the resulting throughput would only
    reflect the state transition cost. Folding them into a returned
    scalar costs roughly one reduce per step but forces XLA to keep
    all four step outputs in the compiled graph, so the numbers
    reflect what a real training loop actually pays.
    """
    vmap_reset = jax.vmap(env.reset_env, in_axes=(0, None))
    vmap_step = jax.vmap(env.step_env, in_axes=(0, 0, 0, None))

    def scan_body(carry, _):
        rng, state, sink = carry
        rng, key_actions, key_step = random.split(rng, 3)
        actions = random.randint(
            key_actions, (batch_size,), 0, NUM_ACTIONS, dtype=jnp.int32
        )
        step_keys = random.split(key_step, batch_size)
        obs, state, reward, done, _info = vmap_step(step_keys, state, actions, params)
        # Force obs/reward/done into the dependency graph so XLA cannot
        # DCE them. A scalar sum keeps the accumulator cheap.
        sink = (
            sink
            + jnp.sum(obs.astype(jnp.float32))
            + jnp.sum(reward.astype(jnp.float32))
            + jnp.sum(done.astype(jnp.float32))
        )
        return (rng, state, sink), None

    @jax.jit
    def run(rng: jax.Array):
        rng, reset_rng = random.split(rng)
        reset_keys = random.split(reset_rng, batch_size)
        _, state = vmap_reset(reset_keys, params)
        sink = jnp.float32(0.0)
        (_, state, sink), _ = lax.scan(
            scan_body, (rng, state, sink), None, length=inner_steps
        )
        # Return sink alongside state so the caller's
        # block_until_ready also has to materialise the sink (and
        # therefore everything that feeds it).
        return state, sink

    return run


def _build_baseline_scan(
    env: FactoriaXEnv,
    params: EnvParams,
    batch_size: int,
    inner_steps: int,
):
    """JIT a ``vmap+lax.scan`` of ``jax.tree.map(x + 1)`` over the state."""
    vmap_reset = jax.vmap(env.reset_env, in_axes=(0, None))

    def add_one(state):
        return jax.tree.map(lambda x: x + lax.full_like(x, 1), state)

    vmap_add = jax.vmap(add_one)

    def scan_body(carry, _):
        rng, state = carry
        rng, _ = random.split(rng)
        state = vmap_add(state)
        return (rng, state), None

    @jax.jit
    def run(rng: jax.Array):
        rng, reset_rng = random.split(rng)
        reset_keys = random.split(reset_rng, batch_size)
        _, state = vmap_reset(reset_keys, params)
        (_, state), _ = lax.scan(scan_body, (rng, state), None, length=inner_steps)
        return state

    return run


def _time_run(run_fn, rng: jax.Array) -> float:
    """Single timing trial. Returns wall-clock seconds."""
    start = time.perf_counter()
    state = run_fn(rng)
    jax.block_until_ready(jax.tree.leaves(state))
    return time.perf_counter() - start


def _calibrate_inner_steps(
    env: FactoriaXEnv,
    params: EnvParams,
    batch_size: int,
    target_seconds: float,
    probe_steps: int,
    rng: jax.Array,
) -> tuple[int, jax.Array]:
    """Run a probe scan, measure per-tick wall, return target inner_steps."""
    probe_fn = _build_step_scan(env, params, batch_size, probe_steps)

    warm_rng, rng = random.split(rng)
    _time_run(probe_fn, warm_rng)  # JIT compile

    measure_rng, rng = random.split(rng)
    wall = _time_run(probe_fn, measure_rng)
    per_tick = wall / probe_steps
    target = max(probe_steps, int(target_seconds / per_tick))
    return target, rng


def _percentile(samples: list[float], pct: float) -> float:
    """Inclusive percentile via index interpolation."""
    if not samples:
        return float("nan")
    s = sorted(samples)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _summarise_samples(samples: list[float]) -> dict[str, float]:
    median = statistics.median(samples)
    return {
        "steps_per_sec": median,
        "steps_per_sec_median": median,
        "steps_per_sec_mean": statistics.mean(samples),
        "steps_per_sec_std": (statistics.stdev(samples) if len(samples) > 1 else 0.0),
        "steps_per_sec_min": min(samples),
        "steps_per_sec_max": max(samples),
        "steps_per_sec_p05": _percentile(samples, 5),
        "steps_per_sec_p95": _percentile(samples, 95),
    }


def _measure(
    env: FactoriaXEnv,
    params: EnvParams,
    batch_size: int,
    trials: int,
    inner_steps: int,
    rng: jax.Array,
) -> tuple[dict[str, object], jax.Array]:
    """Compile, warm (startup_seconds), then run trials. Returns row + rng.

    Steady-state SPS contract: ``startup_seconds`` covers ONLY the first
    call (JIT compile + first kernel launch + autotune) and is NEVER
    folded into ``trial_samples``. Every entry in ``trial_samples`` is
    a post-JIT call with the kernel already cached, so the median /
    mean / std summarise steady-state throughput.
    """
    run_fn = _build_step_scan(env, params, batch_size, inner_steps)

    # --- Warmup. Not counted in trial samples or in SPS summaries. ---
    startup_rng, rng = random.split(rng)
    startup_seconds = _time_run(run_fn, startup_rng)

    # --- Steady-state trials. JIT is cached from the warmup above. ---
    samples: list[float] = []
    for _ in range(trials):
        trial_rng, rng = random.split(rng)
        wall = _time_run(run_fn, trial_rng)
        samples.append(batch_size * inner_steps / wall)

    summary = _summarise_samples(samples)
    return {
        "inner_steps": inner_steps,
        "startup_seconds": startup_seconds,
        "trial_samples": samples,
        **summary,
    }, rng


def _measure_baseline(
    env: FactoriaXEnv,
    params: EnvParams,
    batch_size: int,
    inner_steps: int,
    rng: jax.Array,
) -> tuple[dict[str, float], jax.Array]:
    """Build, warm, time once. Baseline doesn't need 50 trials for a ceiling."""
    run_fn = _build_baseline_scan(env, params, batch_size, inner_steps)

    startup_rng, rng = random.split(rng)
    startup_seconds = _time_run(run_fn, startup_rng)

    trial_rng, rng = random.split(rng)
    wall = _time_run(run_fn, trial_rng)
    sps = batch_size * inner_steps / wall
    return {
        "baseline_steps_per_sec": sps,
        "baseline_startup_seconds": startup_seconds,
    }, rng


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    args = _parse_args()

    map_sizes = tuple(args.map_sizes) if args.map_sizes else _DEFAULT_MAP_SIZES
    batch_sizes = tuple(args.num_envs) if args.num_envs else _DEFAULT_NUM_ENVS
    entity_multipliers = (
        tuple(args.entity_multipliers)
        if args.entity_multipliers
        else _DEFAULT_ENTITY_MULTIPLIERS
    )

    device_full = _detect_device_full()
    logger.info(
        "device_label=%s device_full=%s obs=%s map_sizes=%s batch_sizes=%s "
        "entity_multipliers=%s trials=%d target_seconds=%.2f baseline=%s",
        args.device_label,
        device_full,
        args.obs,
        list(map_sizes),
        list(batch_sizes),
        list(entity_multipliers),
        args.trials,
        args.target_trial_seconds,
        args.baseline,
    )

    wandb_run = None
    if not args.no_wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"throughput_{args.device_label}",
            config={
                "device_label": args.device_label,
                "device_full": device_full,
                "obs": args.obs,
                "achievement_fn": "easy_rocket_conditions",
                "reward_fn": "easy_rocket_reward",
                "trials": args.trials,
                "target_trial_seconds": args.target_trial_seconds,
                "override_inner_steps": args.inner_steps,
                "probe_inner_steps": args.probe_inner_steps,
                "map_sizes": list(map_sizes),
                "batch_sizes": list(batch_sizes),
                "entity_multipliers": list(entity_multipliers),
                "baseline": args.baseline,
                "jax_version": jax.__version__,
            },
        )

    env = FactoriaXEnv(
        achievement_fn=easy_rocket_conditions,
        reward_fn=easy_rocket_reward,
        obs=args.obs,
    )
    rng = random.PRNGKey(args.seed)

    measurements: list[dict[str, object]] = []
    with _GpuStateSampler() as sampler:
        for map_size in map_sizes:
            standard = _standard_max_machines(map_size)
            for entity_multiplier in entity_multipliers:
                entity_limit = max(1, int(round(entity_multiplier * standard)))
                params = EnvParams(
                    map_width=map_size,
                    map_height=map_size,
                    num_players=1,
                    max_machines=entity_limit,
                )
                for batch_size in batch_sizes:
                    row: dict[str, object] = {
                        "device": args.device_label,
                        "device_full": device_full,
                        "map_size": map_size,
                        "batch_size": batch_size,
                        "entity_multiplier": entity_multiplier,
                        "entity_limit": entity_limit,
                        "entity_limit_standard": standard,
                    }
                    try:
                        if args.inner_steps is not None:
                            inner_steps = args.inner_steps
                        else:
                            rng, sub_rng = random.split(rng)
                            inner_steps, _ = _calibrate_inner_steps(
                                env,
                                params,
                                batch_size,
                                args.target_trial_seconds,
                                args.probe_inner_steps,
                                sub_rng,
                            )
                        rng, sub_rng = random.split(rng)
                        result, _ = _measure(
                            env,
                            params,
                            batch_size,
                            args.trials,
                            inner_steps,
                            sub_rng,
                        )
                        row.update(result)
                        row["skipped"] = False
                        if args.baseline:
                            rng, sub_rng = random.split(rng)
                            base, _ = _measure_baseline(
                                env,
                                params,
                                batch_size,
                                inner_steps,
                                sub_rng,
                            )
                            row.update(base)
                    except (jax.errors.JaxRuntimeError, RuntimeError) as exc:
                        logger.error(
                            "SKIPPED map=%d batch=%d entities=%d: %s",
                            map_size,
                            batch_size,
                            entity_limit,
                            exc,
                        )
                        row["skipped"] = True
                        row["skip_reason"] = str(exc)
                    measurements.append(row)

                    if row["skipped"]:
                        logger.info(
                            "map=%d batch=%d entities=%d (x%.2f) SKIPPED",
                            map_size,
                            batch_size,
                            entity_limit,
                            entity_multiplier,
                        )
                    else:
                        logger.info(
                            "map=%d batch=%d entities=%d (x%.2f) inner=%d "
                            "startup=%.2fs median_sps=%.3e std=%.3e",
                            map_size,
                            batch_size,
                            entity_limit,
                            entity_multiplier,
                            row["inner_steps"],
                            row["startup_seconds"],
                            row["steps_per_sec_median"],
                            row["steps_per_sec_std"],
                        )
                    if wandb_run is not None and not row["skipped"]:
                        samples = row["trial_samples"]
                        assert isinstance(samples, list)
                        for trial_idx, sps in enumerate(samples):
                            wandb_run.log(
                                {
                                    "map_size": map_size,
                                    "batch_size": batch_size,
                                    "entity_limit": entity_limit,
                                    "entity_multiplier": entity_multiplier,
                                    "trial_idx": trial_idx,
                                    "steps_per_sec": sps,
                                }
                            )
                        # Per-config summary event so warmup time and
                        # median/std appear as their own metric streams
                        # in wandb, separable from per-trial sps.
                        summary_event: dict[str, object] = {
                            "config/map_size": map_size,
                            "config/batch_size": batch_size,
                            "config/entity_limit": entity_limit,
                            "config/entity_multiplier": entity_multiplier,
                            "config/inner_steps": row["inner_steps"],
                            "config/startup_seconds": row["startup_seconds"],
                            "config/median_steps_per_sec": (
                                row["steps_per_sec_median"]
                            ),
                            "config/std_steps_per_sec": (row["steps_per_sec_std"]),
                        }
                        if args.baseline and "baseline_steps_per_sec" in row:
                            summary_event["config/baseline_steps_per_sec"] = row[
                                "baseline_steps_per_sec"
                            ]
                            summary_event["config/baseline_startup_seconds"] = row.get(
                                "baseline_startup_seconds", 0.0
                            )
                        wandb_run.log(summary_event)
                        key = f"map{map_size}_batch{batch_size}_ent{entity_limit}"
                        wandb_run.summary[f"sps/{key}/median"] = row[
                            "steps_per_sec_median"
                        ]
                        wandb_run.summary[f"sps/{key}/std"] = row["steps_per_sec_std"]
                        wandb_run.summary[f"startup/{key}"] = row["startup_seconds"]
                        wandb_run.summary[f"inner_steps/{key}"] = row["inner_steps"]
                        if args.baseline and "baseline_steps_per_sec" in row:
                            wandb_run.summary[f"baseline/{key}"] = row[
                                "baseline_steps_per_sec"
                            ]
                            wandb_run.summary[f"baseline_startup/{key}"] = row.get(
                                "baseline_startup_seconds", 0.0
                            )

    gpu_state = sampler.summary()
    logger.info("gpu_state=%s", gpu_state)

    output_path = args.output or Path.cwd() / f"throughput_{args.device_label}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "device_label": args.device_label,
        "device_full": device_full,
        "obs": args.obs,
        "achievement_fn": "easy_rocket_conditions",
        "reward_fn": "easy_rocket_reward",
        "trials": args.trials,
        "target_trial_seconds": args.target_trial_seconds,
        "override_inner_steps": args.inner_steps,
        "probe_inner_steps": args.probe_inner_steps,
        "entity_multipliers": list(entity_multipliers),
        "baseline": args.baseline,
        "jax_version": jax.__version__,
        "gpu_state": gpu_state,
        "measurements": measurements,
    }
    output_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    logger.info("wrote %s", output_path)

    if wandb_run is not None:
        for key, value in gpu_state.items():
            wandb_run.summary[f"gpu/{key}"] = value
        wandb_run.save(str(output_path), policy="now")
        wandb_run.finish()

    return 0 if any(not m.get("skipped") for m in measurements) else 1


if __name__ == "__main__":
    raise SystemExit(main())
