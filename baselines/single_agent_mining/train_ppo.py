"""PPO curriculum training on the single-agent mining benchmark.

Trains a shared policy across the five mining benchmark levels using either
sequential or mixed sampling, then evaluates the final policy on the full
benchmark via ``SingleAgentMiningBenchmark``.

Sequential sampling trains all steps on one level before progressing to the
next. The policy warm-starts from the previous level's weights, which acts as
a simple curriculum. Mixed sampling collects rollouts from all five levels each
iteration and performs a single joint PPO update on the combined batch.

All five levels have different map sizes, so local observations with a fixed
radius are used throughout. This keeps the observation dimension constant
regardless of map size, letting the same network train and evaluate across
all levels without recompilation.

The action space is restricted to the first six actions: NOOP, LEFT, RIGHT,
UP, DOWN, MINE.

Usage::

    python -m baselines.single_agent_mining.train_ppo
    python -m baselines.single_agent_mining.train_ppo --sampling-mode sequential
    python -m baselines.single_agent_mining.train_ppo --use-wandb --total-steps 5000000
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import pickle
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from baselines.ppo import (
    ActorCritic,
    PPOConfig,
    RunningStats,
    Transition,
    compute_gae,
    init_running_stats,
    make_update_fn,
    normalize_obs,
    update_running_stats,
)
from baselines.ppo.cli import add_ppo_args, ppo_config_from_args
from benchmarks.core import BenchmarkLevel
from benchmarks.runner import BenchmarkRunner
from benchmarks.single_agent_mining.analysis import (
    log_to_wandb,
    render_level_video,
    save_mp4,
)
from benchmarks.single_agent_mining.benchmark import SingleAgentMiningBenchmark
from factoriax.envs import FactoriaXEnv
from factoriax.levels import build_state
from factoriax.observations import local_array
from factoriax.rewards import mining_reward
from factoriax.state import EnvParams, EnvState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# NOOP=0, LEFT=1, RIGHT=2, UP=3, DOWN=4, MINE=5
_NUM_ACTIONS = 6


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Config:
    """Hyperparameters for PPO curriculum training on the mining benchmark.

    Embeds a ``PPOConfig`` for shared PPO hyperparameters and adds
    domain-specific fields for the mining curriculum.

    Attributes:
        ppo: Shared PPO hyperparameters (network, optimizer, logging, etc.).
        sampling_mode: ``"sequential"`` trains all steps on one level before
            progressing; ``"mixed"`` collects from all levels each iteration.
        eval_seeds: Number of independent evaluation seeds for benchmark
            scoring.
    """

    ppo: PPOConfig = dataclasses.field(default_factory=PPOConfig)
    sampling_mode: str = "mixed"
    eval_seeds: int = 10


# ---------------------------------------------------------------------------
# Per-level collection
# ---------------------------------------------------------------------------


def make_collect_fn(
    env: FactoriaXEnv,
    bench_level: BenchmarkLevel,
    network: ActorCritic,
    config: Config,
) -> tuple[Callable, Callable]:
    """Build a JIT-compiled collection function for one benchmark level.

    The level always resets to the same fixed initial state so episodes are
    deterministic in layout. Episodes that reach ``done`` are immediately
    reset to this fixed state. The obs function uses ``local_array`` with the
    configured radius, giving a constant observation dimension across all
    levels regardless of map size.

    Args:
        env: FactoriaX environment instance.
        bench_level: Level to build the collector for.
        network: Shared actor-critic network.
        config: Training configuration.

    Returns:
        Tuple ``(collect_fn, init_fn)``. ``init_fn()`` returns the initial
        ``(env_states, obs)`` for this level. ``collect_fn(params, obs_stats,
        env_states, obs, rng)`` returns ``(trajectories, next_env_states,
        next_obs, last_values, next_rng)`` with trajectory fields of shape
        ``(T, N, ...)``.
    """
    env_params = bench_level.env_params
    level_state = build_state(bench_level.level, env_params)
    num_envs = config.ppo.num_envs
    obs_radius = config.ppo.obs_radius

    vmap_step = jax.vmap(env.step_env, in_axes=(0, 0, 0, None))
    vmap_reward = jax.vmap(mining_reward, in_axes=(0, 0, None))

    def _obs(state: EnvState) -> jax.Array:
        return local_array(state, env_params, state.selected_player, obs_radius)

    vmap_obs = jax.vmap(_obs)

    def _broadcast(x: jax.Array) -> jax.Array:
        a = jnp.asarray(x)
        return jnp.broadcast_to(a[None], (num_envs,) + a.shape)

    fixed_states = jax.tree_util.tree_map(_broadcast, level_state)

    def init() -> tuple[EnvState, jax.Array]:
        """Return initial env states and observations for this level.

        Returns:
            Tuple ``(env_states, obs)`` with shapes ``(N, ...)`` and
            ``(N, obs_dim)``.
        """
        return fixed_states, vmap_obs(fixed_states)

    @jax.jit
    def collect(
        params: Any,
        obs_stats: RunningStats,
        env_states: EnvState,
        obs: jax.Array,
        rng: jax.Array,
    ) -> tuple[Transition, EnvState, jax.Array, jax.Array, jax.Array]:
        """Collect ``rollout_steps`` transitions from ``num_envs`` environments.

        Args:
            params: Current network parameters.
            obs_stats: Running observation statistics for normalization.
            env_states: Current vectorized env state, shape ``(N, ...)``.
            obs: Current observations, shape ``(N, obs_dim)``.
            rng: PRNG key.

        Returns:
            Tuple ``(trajectories, next_env_states, next_obs, last_values,
            next_rng)``. Trajectory fields have shape ``(T, N, ...)``.
        """

        def rollout_step(
            carry: tuple[EnvState, jax.Array, jax.Array],
            _: None,
        ) -> tuple[tuple[EnvState, jax.Array, jax.Array], Transition]:
            states, cur_obs, step_rng = carry
            step_rng, key_act, key_step = jax.random.split(step_rng, 3)

            norm = (
                normalize_obs(obs_stats, cur_obs)
                if config.ppo.normalize_obs
                else cur_obs
            )
            logits, values = network.apply(params, norm)

            actions = jax.random.categorical(key_act, logits)
            log_probs = jax.nn.log_softmax(logits)[jnp.arange(num_envs), actions]

            keys_step = jax.random.split(key_step, num_envs)
            prev_states = states
            _, next_states, _, dones, _ = vmap_step(
                keys_step, states, actions, env_params
            )
            rewards = vmap_reward(prev_states, next_states, env_params)

            def _where(r: jax.Array, s: jax.Array) -> jax.Array:
                pad = dones.reshape((-1,) + (1,) * (s.ndim - 1))
                return jnp.where(pad, r, s)

            next_states = jax.tree_util.tree_map(_where, fixed_states, next_states)
            next_obs = vmap_obs(next_states)

            return (next_states, next_obs, step_rng), Transition(
                obs=cur_obs,
                action=actions,
                log_prob=log_probs,
                value=values,
                reward=rewards,
                done=dones,
            )

        (next_states, next_obs, rng), trajectories = jax.lax.scan(
            rollout_step,
            (env_states, obs, rng),
            None,
            length=config.ppo.rollout_steps,
        )
        norm_last = (
            normalize_obs(obs_stats, next_obs) if config.ppo.normalize_obs else next_obs
        )
        _, last_values = network.apply(params, norm_last)
        return trajectories, next_states, next_obs, last_values, rng

    return collect, init


# ---------------------------------------------------------------------------
# Sequential training
# ---------------------------------------------------------------------------


def _run_sequential(
    levels: list[BenchmarkLevel],
    env: FactoriaXEnv,
    network: ActorCritic,
    update_fn: Callable,
    params: Any,
    obs_stats: RunningStats,
    opt_state: optax.OptState,
    obs_dim: int,
    config: Config,
    rng: jax.Array,
    wandb_run: Any | None,
) -> tuple[Any, RunningStats]:
    """Train sequentially: all steps on one level before progressing.

    Params and observation statistics carry over to the next level,
    acting as a warm start. Each level's collect function is freshly
    JIT-compiled for its map shape.

    Args:
        levels: Benchmark levels in training order.
        env: FactoriaX environment instance.
        network: Shared actor-critic network.
        update_fn: JIT-compiled PPO update function.
        params: Initial network parameters.
        obs_stats: Initial running observation statistics.
        opt_state: Initial optimizer state.
        obs_dim: Observation vector length.
        config: Training configuration.
        rng: PRNG key.
        wandb_run: Live W&B run or None.

    Returns:
        Tuple ``(params, obs_stats)`` after training all levels.
    """
    steps_per_level = config.ppo.total_steps // len(levels)
    steps_per_iter = config.ppo.num_envs * config.ppo.rollout_steps
    iters_per_level = max(1, steps_per_level // steps_per_iter)
    current_step = 0

    for level_idx, bl in enumerate(levels):
        logger.info(
            "--- Level %d/%d: %s  (%d iters x %d steps) ---",
            level_idx + 1,
            len(levels),
            bl.name,
            iters_per_level,
            steps_per_iter,
        )
        collect_fn, init_fn = make_collect_fn(env, bl, network, config)
        env_states, obs = init_fn()

        t_start = time.time()
        completed_returns: deque[float] = deque(maxlen=1000)
        running_ep_return = np.zeros(config.ppo.num_envs, dtype=np.float32)
        running_ep_length = np.zeros(config.ppo.num_envs, dtype=np.int32)

        for it in range(iters_per_level):
            rng, key_collect, key_update = jax.random.split(rng, 3)

            trajectories, env_states, obs, last_values, _ = collect_fn(
                params, obs_stats, env_states, obs, key_collect
            )

            flat_obs = trajectories.obs.reshape(-1, obs_dim)
            obs_stats = update_running_stats(obs_stats, flat_obs)

            adv, ret = compute_gae(
                trajectories.reward,
                trajectories.value,
                trajectories.done,
                last_values,
                config.ppo.gamma,
                config.ppo.gae_lambda,
            )

            params, opt_state, metrics, _ = update_fn(
                params,
                opt_state,
                obs_stats,
                flat_obs,
                trajectories.action.reshape(-1),
                trajectories.log_prob.reshape(-1),
                adv.reshape(-1),
                ret.reshape(-1),
                key_update,
            )
            current_step += steps_per_iter

            rewards_np = np.array(trajectories.reward)
            dones_np = np.array(trajectories.done)
            for t in range(rewards_np.shape[0]):
                running_ep_return += rewards_np[t]
                running_ep_length += 1
                for n in np.where(dones_np[t])[0]:
                    completed_returns.append(float(running_ep_return[n]))
                    running_ep_return[n] = 0.0
                    running_ep_length[n] = 0

            if (it + 1) % config.ppo.log_interval == 0 or it == iters_per_level - 1:
                elapsed = time.time() - t_start
                sps = (it + 1) * steps_per_iter / elapsed
                mean_ret = (
                    float(np.mean(list(completed_returns)))
                    if completed_returns
                    else 0.0
                )
                logger.info(
                    "  [%s] iter=%d/%d  step=%d  sps=%.0f  "
                    "mean_ret=%.3f  loss=%.4f  entropy=%.4f",
                    bl.name,
                    it + 1,
                    iters_per_level,
                    current_step,
                    sps,
                    mean_ret,
                    float(metrics["loss/total"]),
                    float(metrics["loss/entropy"]),
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            f"train/{bl.name}/step": float(current_step),
                            f"train/{bl.name}/sps": sps,
                            f"train/{bl.name}/mean_ep_return": mean_ret,
                            **{
                                f"train/{bl.name}/{k}": float(v)
                                for k, v in metrics.items()
                            },
                        },
                        step=current_step,
                    )

    return params, obs_stats


# ---------------------------------------------------------------------------
# Mixed training
# ---------------------------------------------------------------------------


def _run_mixed(
    levels: list[BenchmarkLevel],
    env: FactoriaXEnv,
    network: ActorCritic,
    update_fn: Callable,
    params: Any,
    obs_stats: RunningStats,
    opt_state: optax.OptState,
    obs_dim: int,
    config: Config,
    rng: jax.Array,
    wandb_run: Any | None,
) -> tuple[Any, RunningStats]:
    """Train with mixed sampling: collect from all levels each iteration.

    Each update draws an equal number of transitions from every level and
    trains on the concatenated batch. This means the policy sees a balanced
    distribution of level difficulty throughout training rather than mastering
    easy levels before seeing hard ones.

    Args:
        levels: Benchmark levels to collect from each iteration.
        env: FactoriaX environment instance.
        network: Shared actor-critic network.
        update_fn: JIT-compiled PPO update function.
        params: Initial network parameters.
        obs_stats: Initial running observation statistics.
        opt_state: Initial optimizer state.
        obs_dim: Observation vector length.
        config: Training configuration.
        rng: PRNG key.
        wandb_run: Live W&B run or None.

    Returns:
        Tuple ``(params, obs_stats)`` after training.
    """
    num_levels = len(levels)
    steps_per_iter_per_level = config.ppo.num_envs * config.ppo.rollout_steps
    steps_per_iter_total = steps_per_iter_per_level * num_levels
    num_iters = max(1, config.ppo.total_steps // steps_per_iter_total)

    logger.info(
        "Mixed: %d levels x %d steps/iter = %d combined steps/iter  (%d iters total)",
        num_levels,
        steps_per_iter_per_level,
        steps_per_iter_total,
        num_iters,
    )

    collect_fns: list[Callable] = []
    level_env_states: list[EnvState] = []
    level_obs: list[jax.Array] = []
    for bl in levels:
        collect_fn, init_fn = make_collect_fn(env, bl, network, config)
        env_states_l, obs_l = init_fn()
        collect_fns.append(collect_fn)
        level_env_states.append(env_states_l)
        level_obs.append(obs_l)

    t_start = time.time()
    completed_returns: deque[float] = deque(maxlen=1000)
    running_ep_return = np.zeros(config.ppo.num_envs * num_levels, dtype=np.float32)
    running_ep_length = np.zeros(config.ppo.num_envs * num_levels, dtype=np.int32)
    current_step = 0

    for it in range(num_iters):
        rng, key_update = jax.random.split(rng)

        all_obs: list[jax.Array] = []
        all_actions: list[jax.Array] = []
        all_log_probs: list[jax.Array] = []
        all_adv: list[jax.Array] = []
        all_ret: list[jax.Array] = []

        for l_idx, (collect_fn, bl) in enumerate(zip(collect_fns, levels)):
            rng, key_l = jax.random.split(rng)
            (
                traj,
                level_env_states[l_idx],
                level_obs[l_idx],
                last_vals,
                _,
            ) = collect_fn(
                params,
                obs_stats,
                level_env_states[l_idx],
                level_obs[l_idx],
                key_l,
            )

            adv, ret = compute_gae(
                traj.reward,
                traj.value,
                traj.done,
                last_vals,
                config.ppo.gamma,
                config.ppo.gae_lambda,
            )

            all_obs.append(traj.obs.reshape(-1, obs_dim))
            all_actions.append(traj.action.reshape(-1))
            all_log_probs.append(traj.log_prob.reshape(-1))
            all_adv.append(adv.reshape(-1))
            all_ret.append(ret.reshape(-1))

            rewards_np = np.array(traj.reward)
            dones_np = np.array(traj.done)
            offset = l_idx * config.ppo.num_envs
            for t in range(rewards_np.shape[0]):
                running_ep_return[offset : offset + config.ppo.num_envs] += rewards_np[
                    t
                ]
                running_ep_length[offset : offset + config.ppo.num_envs] += 1
                for n in np.where(dones_np[t])[0]:
                    completed_returns.append(float(running_ep_return[offset + n]))
                    running_ep_return[offset + n] = 0.0
                    running_ep_length[offset + n] = 0

        flat_obs = jnp.concatenate(all_obs)
        obs_stats = update_running_stats(obs_stats, flat_obs)

        params, opt_state, metrics, _ = update_fn(
            params,
            opt_state,
            obs_stats,
            flat_obs,
            jnp.concatenate(all_actions),
            jnp.concatenate(all_log_probs),
            jnp.concatenate(all_adv),
            jnp.concatenate(all_ret),
            key_update,
        )
        current_step += steps_per_iter_total

        if (it + 1) % config.ppo.log_interval == 0 or it == num_iters - 1:
            elapsed = time.time() - t_start
            sps = current_step / elapsed
            mean_ret = (
                float(np.mean(list(completed_returns))) if completed_returns else 0.0
            )
            logger.info(
                "iter=%d/%d  step=%d  sps=%.0f  mean_ret=%.3f  "
                "loss=%.4f  entropy=%.4f  kl=%.5f",
                it + 1,
                num_iters,
                current_step,
                sps,
                mean_ret,
                float(metrics["loss/total"]),
                float(metrics["loss/entropy"]),
                float(metrics["misc/approx_kl"]),
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/step": float(current_step),
                        "train/sps": sps,
                        "train/mean_ep_return": mean_ret,
                        **{f"train/{k}": float(v) for k, v in metrics.items()},
                    },
                    step=current_step,
                )

    return params, obs_stats


# ---------------------------------------------------------------------------
# Benchmark evaluation helpers
# ---------------------------------------------------------------------------


def _plot_benchmark_boxplots(
    results: list,
) -> plt.Figure:
    """Box plot of per-level and aggregate scores across multiple evaluation seeds.

    Each box spans seeds min-max with median marked, making it easy to see
    which levels are solved consistently versus which are volatile.

    Args:
        results: List of ``BenchmarkResult``, one per evaluation seed.

    Returns:
        Matplotlib figure with one box per level plus one for the aggregate.
    """
    level_names = [lr.level_name for lr in results[0].level_results]
    labels = level_names + ["aggregate"]

    level_scores = np.array(
        [[lr.weighted_score for lr in r.level_results] for r in results]
    )  # (n_seeds, n_levels)
    aggregate_scores = np.array([r.aggregate_score for r in results])  # (n_seeds,)
    all_scores = np.hstack(
        [level_scores, aggregate_scores[:, None]]
    )  # (n_seeds, n_levels+1)

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.6), 5))
    bp = ax.boxplot(
        all_scores,
        labels=labels,
        patch_artist=True,
        medianprops={"color": "crimson", "linewidth": 1.5},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor("steelblue" if i < len(level_names) else "seagreen")
        patch.set_alpha(0.65)

    ax.set_ylabel("Weighted score")
    ax.set_title(
        f"Benchmark score distribution  ({len(results)} seeds)"
        f"  --  aggregate median ="
        f" {float(np.median(aggregate_scores)):.1f}"
    )
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------


def train(config: Config) -> None:
    """Run PPO curriculum training and evaluate on the full benchmark.

    Builds the shared network once, trains with the configured sampling mode,
    optionally saves a checkpoint, then evaluates the final policy through
    ``BenchmarkRunner`` using local observations to match training.

    Args:
        config: Training configuration.
    """
    rng = jax.random.PRNGKey(config.ppo.seed)

    wandb_run = None
    if config.ppo.use_wandb:
        try:
            import wandb  # type: ignore[import-untyped]

            wandb_run = wandb.init(
                project=config.ppo.wandb_project,
                name=config.ppo.wandb_run_name,
                config=dataclasses.asdict(config),
                tags=[
                    f"mode_{config.sampling_mode}",
                    f"obs_local_r{config.ppo.obs_radius}",
                ],
            )
        except ImportError:
            logger.error("wandb not found. Install with: uv add wandb")

    benchmark = SingleAgentMiningBenchmark()
    levels = benchmark.levels()
    env = FactoriaXEnv()

    # Derive obs_dim from a sample build -- consistent across all levels.
    _sample_state = build_state(levels[0].level, levels[0].env_params)
    obs_dim = int(
        local_array(
            _sample_state,
            levels[0].env_params,
            0,
            config.ppo.obs_radius,
        ).shape[0]
    )
    logger.info(
        "obs_dim=%d (local obs, radius=%d)  num_actions=%d  mode=%s",
        obs_dim,
        config.ppo.obs_radius,
        _NUM_ACTIONS,
        config.sampling_mode,
    )

    network = ActorCritic(hidden_dims=config.ppo.hidden_dims, num_actions=_NUM_ACTIONS)
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.ppo.max_grad_norm),
        optax.adam(config.ppo.learning_rate),
    )

    rng, key_init = jax.random.split(rng)
    params = network.init(key_init, jnp.zeros(obs_dim))
    opt_state = optimizer.init(params)
    obs_stats = init_running_stats(obs_dim)
    update_fn = make_update_fn(network, optimizer, config.ppo)

    logger.info(
        "Starting training: %d total steps  %d levels  %d envs/level",
        config.ppo.total_steps,
        len(levels),
        config.ppo.num_envs,
    )

    rng, key_train = jax.random.split(rng)
    train_kwargs: dict[str, Any] = dict(
        levels=levels,
        env=env,
        network=network,
        update_fn=update_fn,
        params=params,
        obs_stats=obs_stats,
        opt_state=opt_state,
        obs_dim=obs_dim,
        config=config,
        rng=key_train,
        wandb_run=wandb_run,
    )

    if config.sampling_mode == "sequential":
        params, obs_stats = _run_sequential(**train_kwargs)
    else:
        params, obs_stats = _run_mixed(**train_kwargs)

    if config.ppo.save_path is not None:
        out_dir = Path(config.ppo.save_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = out_dir / "final.pkl"
        data = {
            "params": jax.device_get(jax.tree_util.tree_map(np.array, params)),
            "obs_mean": np.array(obs_stats.mean),
            "obs_var": np.array(obs_stats.var),
            "obs_count": int(obs_stats.count),
            "config": dataclasses.asdict(config),
        }
        with open(ckpt_path, "wb") as f:
            pickle.dump(data, f)
        logger.info("Checkpoint saved -> %s", ckpt_path)

    # Wrap the trained network as a Policy and evaluate on the full benchmark.
    # The runner is configured with the same local obs function used in training
    # so the policy sees the same observation format it was trained on.
    jit_apply = jax.jit(network.apply)
    _obs_stats = obs_stats
    _normalize = config.ppo.normalize_obs

    def trained_policy(obs: jax.Array) -> jax.Array:
        """Greedy policy: pick the highest-logit action.

        Args:
            obs: Observation array matching the training obs format.

        Returns:
            Integer action scalar.
        """
        norm = normalize_obs(_obs_stats, obs) if _normalize else obs
        logits, _ = jit_apply(params, norm)
        return jnp.argmax(logits)

    def eval_obs_fn(
        state: EnvState, env_params: EnvParams, player_idx: int
    ) -> jax.Array:
        """Extract local observations for the benchmark runner.

        Args:
            state: Current environment state.
            env_params: Environment parameters for this level.
            player_idx: Player index.

        Returns:
            Local observation array.
        """
        return local_array(state, env_params, player_idx, config.ppo.obs_radius)

    # Run the benchmark eval_seeds times for statistical robustness. Each run
    # uses a different PRNG seed for step_env so stochastic episode transitions
    # are sampled independently. The trained policy is deterministic (greedy),
    # so variance comes entirely from environment randomness.
    logger.info(
        "Evaluating trained policy on the full benchmark (%d seeds)...",
        config.eval_seeds,
    )
    eval_results = []
    for seed in range(config.eval_seeds):
        runner = BenchmarkRunner(seed=seed)
        eval_results.append(
            runner.run(benchmark, policies=[trained_policy], obs_fn=eval_obs_fn)
        )

    level_names = [lr.level_name for lr in eval_results[0].level_results]
    level_scores = np.array(
        [[lr.weighted_score for lr in r.level_results] for r in eval_results]
    )  # (eval_seeds, n_levels)
    aggregate_scores = np.array([r.aggregate_score for r in eval_results])

    # Log summary statistics and the box plot under the benchmark/ W&B category.
    if wandb_run is not None:
        try:
            import wandb  # type: ignore[import-untyped]

            score_data: dict[str, object] = {
                "benchmark/aggregate_mean": float(np.mean(aggregate_scores)),
                "benchmark/aggregate_median": float(np.median(aggregate_scores)),
                "benchmark/aggregate_std": float(np.std(aggregate_scores)),
            }
            for j, name in enumerate(level_names):
                score_data[f"benchmark/{name}/mean"] = float(
                    np.mean(level_scores[:, j])
                )
                score_data[f"benchmark/{name}/median"] = float(
                    np.median(level_scores[:, j])
                )
                score_data[f"benchmark/{name}/std"] = float(np.std(level_scores[:, j]))

            fig_box = _plot_benchmark_boxplots(eval_results)
            score_data["benchmark/score_distribution"] = wandb.Image(fig_box)
            plt.close(fig_box)
            wandb_run.log(score_data)
        except ImportError:
            pass

    print(
        f"\nBenchmark : {eval_results[0].benchmark_name}  ({config.eval_seeds} seeds)"
    )
    print(
        f"Aggregate : {np.mean(aggregate_scores):.2f}"
        f" +/- {np.std(aggregate_scores):.2f}"
        f"  (median {np.median(aggregate_scores):.2f})"
    )
    print()
    for j, name in enumerate(level_names):
        s = level_scores[:, j]
        print(
            f"  {name:<28}  mean={np.mean(s):6.1f}"
            f" +/- {np.std(s):5.1f}"
            f"  median={np.median(s):6.1f}"
            f"  [{np.min(s):.0f}, {np.max(s):.0f}]"
        )

    # Use seed-0 result for scalar W&B plots and video rendering so the logged
    # figures and videos correspond to a single consistent episode set.
    result = eval_results[0]

    # Render one video per level using the trained policy with local observations.
    # Videos are saved alongside the checkpoint (or the current directory if no
    # save_path was set) and uploaded to W&B under a dedicated videos/ category.
    logger.info("Rendering final-policy videos for each level...")
    out_dir = (
        Path(config.ppo.save_path) if config.ppo.save_path is not None else Path(".")
    )
    videos: dict[str, Path] = {}
    for bl in benchmark.levels():
        frames = render_level_video(
            bl,
            trained_policy,
            seed=config.ppo.seed,
            obs_fn=eval_obs_fn,
        )
        path = out_dir / f"{bl.name}.mp4"
        save_mp4(frames, path)
        videos[bl.name] = path
        logger.info("  Video saved: %s  (%d frames)", path, len(frames))

    log_to_wandb(result, wandb_run, videos=videos)
    if wandb_run is not None:
        wandb_run.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> Config:
    """Parse command-line arguments into a Config.

    Returns:
        Populated Config dataclass.
    """
    p = argparse.ArgumentParser(
        description=("PPO curriculum training on the single-agent mining benchmark"),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--sampling-mode",
        choices=["mixed", "sequential"],
        default="mixed",
        help=(
            "mixed: collect from all levels each iteration,"
            " single joint update. "
            "sequential: train all steps on one level"
            " before progressing."
        ),
    )
    p.add_argument("--eval-seeds", type=int, default=10)
    add_ppo_args(p)

    args = p.parse_args()
    return Config(
        ppo=ppo_config_from_args(args, wandb_project="factoriax-mining"),
        sampling_mode=args.sampling_mode,
        eval_seeds=args.eval_seeds,
    )


if __name__ == "__main__":
    train(_parse_args())
