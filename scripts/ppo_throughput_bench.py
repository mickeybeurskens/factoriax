"""PPO throughput bench: measure training-loop steps/sec on easy-rocket.

Runs the same PPO inner loop that ``baselines.easy_rocket.ppo.train_ppo``
uses (rollout + GAE + clipped surrogate update), times one warmup
iteration (JIT compile + first kernel launch -> ``startup_seconds``),
then a number of steady-state iterations sized so the measurement
covers a fixed ``--measure-budget-steps`` of env work. Reports the
steady-state rate plus the **wall-clock per 100M steps**, which is a
common reporting unit so a laptop and a cluster number compare
directly regardless of how many steps each one actually timed.

Why easy-rocket only: this is the only scenario with a shipped PPO
baseline, and §3.5 reports a single concrete PPO training number for
the paper's benchmark of record. Other scenarios will need their own
bench when their baselines land.

This script lives in the FactoriaX engine repo (not the paper) so it
is available on Snellius without checking out the paper. The same
script runs locally on a laptop GPU and on a Snellius A100 node via
``scripts/snellius_submit_ppo_throughput.sh``.

Output schema (``./ppo_throughput_<device-label>.json`` by default)::

    {
      "device_label":          "laptop",
      "device_full":           "NVIDIA GeForce RTX 3050 6GB Laptop GPU",
      "scenario":              "EasyRocket-v1",
      "num_envs":              64,
      "rollout_steps":         128,
      "update_epochs":         4,
      "num_minibatches":       4,
      "warmup_iters":          1,
      "measure_iters":         1220,
      "measure_budget_steps":  10000000,
      "measured_steps":        9994240,
      "reference_steps":       100000000,
      "jax_version":           "0.4.x",
      "gpu_state":             {"pstate": "P0", ...},
      "runs": [
        {
          "device":                       "laptop",
          "device_full":                  "NVIDIA GeForce RTX 3050 6GB Laptop GPU",
          "measured_steps":               9994240,
          "reference_steps":              100000000,
          "steady_state_sps":             3.8e4,
          "steady_state_sps_mean":        3.8e4,
          "wallclock_per_100M_seconds":   2630.0,
          "wallclock_seconds":            2630.0,
          "startup_seconds":              42.0,
          "iter_samples":                 [0.21, 0.20, 0.21, ...]
        }
      ]
    }

The ``wallclock_seconds`` field is a backward-compatible alias for
``wallclock_per_100M_seconds`` so the existing figure builder reads
the per-100M number directly.

The paper-side figure ``paper/scripts/figures/throughput.py`` reads
the merged copy at ``paper/data/ppo_throughput.json``; concatenate
the per-device files (or just include them as separate runs in a
single JSON) before placing the file there.
"""

from __future__ import annotations

import argparse
import logging
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import optax
import orjson
from jax import lax, random

import factoriax
from baselines.easy_rocket.ppo.config import PPOConfig
from baselines.easy_rocket.ppo.gae import Transition, compute_gae
from baselines.easy_rocket.ppo.loss import make_update_fn
from baselines.easy_rocket.ppo.network import ActorCritic
from baselines.easy_rocket.ppo.normalization import (
    RunningStats,
    init_running_stats,
    normalize_obs,
    update_running_stats,
)
from factoriax.engine.constants import NUM_ACTIONS

logger = logging.getLogger("factoriax.ppo_throughput_bench")

_SCENARIO_ID: str = "EasyRocket-v1"


class _GpuStateSampler:
    """Background sampler for peak GPU state. Mirrors throughput_bench.py."""

    def __init__(self, interval_sec: float = 0.5) -> None:
        self._interval = interval_sec
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._best_pstate_num: int | None = None
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device-label",
        required=True,
        help='Short tag for the device (e.g. "laptop", "a100").',
    )
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[64, 64],
        help="MLP hidden layer widths.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=1,
        help="Iterations to run before timing (counted as startup).",
    )
    parser.add_argument(
        "--measure-iters",
        type=int,
        default=None,
        help=(
            "Fixed steady-state iteration count. If unset, the iter "
            "count is derived from --measure-budget-steps so the "
            "measurement covers a fixed env-step budget regardless "
            "of how fast the hardware is."
        ),
    )
    parser.add_argument(
        "--measure-budget-steps",
        type=int,
        default=10_000_000,
        help=(
            "Total env steps to time across all measurement "
            "iterations. Default 10M is a reasonable laptop budget; "
            "set to 500M on cluster hardware so the wall time "
            "dominates timer noise. Ignored when --measure-iters is "
            "passed."
        ),
    )
    parser.add_argument(
        "--log-every-iter",
        type=int,
        default=0,
        help=(
            "Log every Nth iteration to wandb (0 = auto; picks a "
            "rate that keeps the total log call count near 200). "
            "Useful on fast hardware where per-iter wandb logging "
            "would otherwise become a bottleneck."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=("Output JSON path. Defaults to ./ppo_throughput_<device-label>.json."),
    )
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="factoriax_ppo_throughput")
    parser.add_argument("--wandb-run-name", default=None)
    return parser.parse_args()


def _detect_device_full() -> str:
    devices = jax.devices()
    if not devices:
        return "no-devices"
    return str(devices[0])


def _build_train_step(
    env: Any,
    env_params: Any,
    network: ActorCritic,
    optimizer: optax.GradientTransformation,
    ppo: PPOConfig,
    update_fn: Any,
    reset_states: Any,
    obs_dim: int,
):
    """Build a JIT-compiled rollout + GAE + PPO update closure.

    Mirrors the inner ``train_step`` from
    ``baselines.easy_rocket.ppo.train_ppo`` but strips the metric and
    logging plumbing so the timed loop is just the training kernel.
    """
    vmap_step = jax.vmap(env.step_env, in_axes=(0, 0, 0, None))
    vmap_obs = jax.vmap(env.get_obs, in_axes=(0, None))
    steps_per_iter = ppo.num_envs * ppo.rollout_steps

    @jax.jit
    def train_step(
        params: Any,
        opt_state: optax.OptState,
        obs_stats: RunningStats,
        env_states: Any,
        obs: jax.Array,
        rng: jax.Array,
    ):
        rng, key_collect, key_update = random.split(rng, 3)

        def _rollout_step(carry, _):
            states, cur_obs, key = carry
            key, key_act, key_step = random.split(key, 3)
            norm = normalize_obs(obs_stats, cur_obs) if ppo.normalize_obs else cur_obs
            logits, values = network.apply(params, norm)
            actions = random.categorical(key_act, logits)
            log_probs = jax.nn.log_softmax(logits)[jnp.arange(ppo.num_envs), actions]
            keys = random.split(key_step, ppo.num_envs)
            _, next_states, rewards, dones, _ = vmap_step(
                keys, states, actions, env_params
            )

            def _where(r, s):
                mask = dones.reshape((-1,) + (1,) * (s.ndim - 1))
                return jnp.where(mask, r, s)

            next_states = jax.tree.map(_where, reset_states, next_states)
            next_obs = vmap_obs(next_states, env_params)
            return (next_states, next_obs, key), Transition(
                obs=cur_obs,
                action=actions,
                log_prob=log_probs,
                value=values,
                reward=rewards,
                done=dones,
            )

        (env_states, obs, _), traj = lax.scan(
            _rollout_step,
            (env_states, obs, key_collect),
            None,
            length=ppo.rollout_steps,
        )
        norm_last = normalize_obs(obs_stats, obs) if ppo.normalize_obs else obs
        _, last_vals = network.apply(params, norm_last)

        adv, ret = compute_gae(
            traj.reward,
            traj.value,
            traj.done,
            last_vals,
            ppo.gamma,
            ppo.gae_lambda,
        )

        flat_obs = traj.obs.reshape((-1, obs_dim))
        if ppo.normalize_obs:
            obs_stats = update_running_stats(obs_stats, flat_obs)

        flat_actions = traj.action.reshape(-1)
        flat_lp = traj.log_prob.reshape(-1)
        flat_adv = adv.reshape(-1)
        flat_ret = ret.reshape(-1)

        params, opt_state, _, _ = update_fn(
            params,
            opt_state,
            obs_stats,
            flat_obs,
            flat_actions,
            flat_lp,
            flat_adv,
            flat_ret,
            key_update,
        )
        return params, opt_state, obs_stats, env_states, obs, rng

    return train_step, steps_per_iter


def _time_iter(train_step, params, opt_state, obs_stats, env_states, obs, rng):
    """Run one timed iteration. Returns (wall_seconds, new_carry)."""
    start = time.perf_counter()
    params, opt_state, obs_stats, env_states, obs, rng = train_step(
        params, opt_state, obs_stats, env_states, obs, rng
    )
    jax.block_until_ready(jax.tree.leaves((params, opt_state, env_states, obs)))
    wall = time.perf_counter() - start
    return wall, (params, opt_state, obs_stats, env_states, obs, rng)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    args = _parse_args()

    device_full = _detect_device_full()
    logger.info(
        "device_label=%s device_full=%s num_envs=%d rollout=%d epochs=%d mb=%d",
        args.device_label,
        device_full,
        args.num_envs,
        args.rollout_steps,
        args.update_epochs,
        args.num_minibatches,
    )

    ppo = PPOConfig(
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        num_minibatches=args.num_minibatches,
        hidden_dims=tuple(args.hidden_dims),
        use_wandb=False,
    )
    steps_per_iter = ppo.num_envs * ppo.rollout_steps
    if args.measure_iters is not None:
        measure_iters = args.measure_iters
    else:
        measure_iters = max(1, args.measure_budget_steps // steps_per_iter)
    measured_steps = measure_iters * steps_per_iter
    if args.log_every_iter > 0:
        log_every = args.log_every_iter
    else:
        log_every = max(1, measure_iters // 200)

    wandb_run = None
    if not args.no_wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"ppo_throughput_{args.device_label}",
            config={
                "device_label": args.device_label,
                "device_full": device_full,
                "scenario": _SCENARIO_ID,
                "num_envs": args.num_envs,
                "rollout_steps": args.rollout_steps,
                "update_epochs": args.update_epochs,
                "num_minibatches": args.num_minibatches,
                "hidden_dims": list(args.hidden_dims),
                "warmup_iters": args.warmup_iters,
                "measure_iters": measure_iters,
                "measure_budget_steps": args.measure_budget_steps,
                "measured_steps": measured_steps,
                "log_every_iter": log_every,
                "jax_version": jax.__version__,
            },
        )

    env, env_params = factoriax.make(_SCENARIO_ID)
    obs_dim = int(env.observation_space(env_params).shape[0])
    logger.info("obs_dim=%d num_actions=%d", obs_dim, NUM_ACTIONS)

    network = ActorCritic(
        hidden_dims=ppo.hidden_dims,
        num_actions=NUM_ACTIONS,
    )
    rng = random.PRNGKey(args.seed)
    rng, key_init = random.split(rng)
    params = network.init(key_init, jnp.zeros(obs_dim))

    optimizer = optax.chain(
        optax.clip_by_global_norm(ppo.max_grad_norm),
        optax.adam(ppo.learning_rate, eps=1e-5),
    )
    opt_state = optimizer.init(params)

    update_fn = make_update_fn(network, optimizer, ppo)
    obs_stats = init_running_stats(obs_dim)

    vmap_reset = jax.vmap(env.reset_env, in_axes=(0, None))
    vmap_obs = jax.vmap(env.get_obs, in_axes=(0, None))
    rng, key_reset = random.split(rng)
    env_reset_keys = random.split(key_reset, ppo.num_envs)
    _, reset_states = vmap_reset(env_reset_keys, env_params)
    env_states = reset_states
    obs = vmap_obs(env_states, env_params)

    train_step, _ = _build_train_step(
        env, env_params, network, optimizer, ppo, update_fn, reset_states, obs_dim
    )
    logger.info(
        "steps_per_iter=%d measure_iters=%d measured_steps=%d log_every=%d",
        steps_per_iter,
        measure_iters,
        measured_steps,
        log_every,
    )

    iter_samples: list[float] = []
    startup_seconds = 0.0

    with _GpuStateSampler() as sampler:
        for warm_idx in range(args.warmup_iters):
            wall, carry = _time_iter(
                train_step, params, opt_state, obs_stats, env_states, obs, rng
            )
            params, opt_state, obs_stats, env_states, obs, rng = carry
            if warm_idx == 0:
                startup_seconds = wall
            logger.info("warmup iter %d wall=%.3fs", warm_idx, wall)

        for trial_idx in range(measure_iters):
            wall, carry = _time_iter(
                train_step, params, opt_state, obs_stats, env_states, obs, rng
            )
            params, opt_state, obs_stats, env_states, obs, rng = carry
            sps = steps_per_iter / wall
            iter_samples.append(wall)
            if trial_idx % log_every == 0 or trial_idx == measure_iters - 1:
                logger.info(
                    "iter %d/%d wall=%.3fs sps=%.3e",
                    trial_idx,
                    measure_iters,
                    wall,
                    sps,
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "trial_idx": trial_idx,
                            "iter_wall_seconds": wall,
                            "iter_steps_per_sec": sps,
                        }
                    )

    gpu_state = sampler.summary()
    logger.info("gpu_state=%s", gpu_state)

    median_iter_wall = statistics.median(iter_samples)
    mean_iter_wall = statistics.mean(iter_samples)
    steady_state_sps_median = steps_per_iter / median_iter_wall
    steady_state_sps_mean = steps_per_iter / mean_iter_wall
    # Wall-clock per 100M steps. Common reporting unit so laptop and
    # cluster numbers compare directly regardless of how many env steps
    # each one actually timed.
    reference_steps = 100_000_000
    # Name matches the JSON output-schema key the figure builder reads.
    wallclock_per_100M_seconds = reference_steps / steady_state_sps_median  # noqa: N806

    output_path = args.output or Path.cwd() / f"ppo_throughput_{args.device_label}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "device_label": args.device_label,
        "device_full": device_full,
        "scenario": _SCENARIO_ID,
        "num_envs": args.num_envs,
        "rollout_steps": args.rollout_steps,
        "update_epochs": args.update_epochs,
        "num_minibatches": args.num_minibatches,
        "hidden_dims": list(args.hidden_dims),
        "warmup_iters": args.warmup_iters,
        "measure_iters": measure_iters,
        "measure_budget_steps": args.measure_budget_steps,
        "measured_steps": measured_steps,
        "log_every_iter": log_every,
        "reference_steps": reference_steps,
        "jax_version": jax.__version__,
        "gpu_state": gpu_state,
        "runs": [
            {
                "device": args.device_label,
                "device_full": device_full,
                "measured_steps": measured_steps,
                "reference_steps": reference_steps,
                "steady_state_sps": steady_state_sps_median,
                "steady_state_sps_mean": steady_state_sps_mean,
                "wallclock_per_100M_seconds": wallclock_per_100M_seconds,
                "wallclock_seconds": wallclock_per_100M_seconds,
                "startup_seconds": startup_seconds,
                "iter_samples": iter_samples,
            }
        ],
    }
    output_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    logger.info(
        "wrote %s steady_state_sps=%.3e startup=%.2fs wallclock_per_100M=%.0fs",
        output_path,
        steady_state_sps_median,
        startup_seconds,
        wallclock_per_100M_seconds,
    )

    if wandb_run is not None:
        wandb_run.summary["steady_state_sps"] = steady_state_sps_median
        wandb_run.summary["steady_state_sps_mean"] = steady_state_sps_mean
        wandb_run.summary["startup_seconds"] = startup_seconds
        wandb_run.summary["wallclock_per_100M_seconds"] = wallclock_per_100M_seconds
        wandb_run.summary["measured_steps"] = measured_steps
        for key, value in gpu_state.items():
            wandb_run.summary[f"gpu/{key}"] = value
        wandb_run.save(str(output_path), policy="now")
        wandb_run.finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
