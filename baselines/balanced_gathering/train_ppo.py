"""PPO training for the balanced gathering benchmark with constraint penalties.

Trains a single agent to mine all three resource types while maintaining
balance between them.  The reward is ``mining_reward - lambda * sum(balance_cost)``,
where lambda controls the constraint penalty strength.

Supports three training modes:
- ``independent``: Train on each level separately.  Measures per-level
  performance without any transfer.
- ``sequential``: Train on level 1, warm-start on level 2, etc.
  Tests whether constraint-aware behaviour transfers across layouts
  (the core hypothesis of RQ3).
- ``mixed``: Sample from all levels each iteration.  Single joint PPO
  update.

All levels have different map sizes, so local observations with a fixed
radius are used.  The action space is restricted to NOOP + movement + MINE.

Usage::

    python -m baselines.balanced_gathering.train_ppo
    python -m baselines.balanced_gathering.train_ppo --mode sequential
    python -m baselines.balanced_gathering.train_ppo --use-wandb --lambda-cost 0.5
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import logging
import pickle
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax
from flax import struct

matplotlib.use("Agg")

from factoriax.benchmarks.balanced_gathering import BalancedGatheringBenchmark
from factoriax.benchmarks.balanced_gathering.scoring import constraint_summary
from factoriax.benchmarks.core import BenchmarkLevel
from factoriax.benchmarks.runner import BenchmarkRunner
from factoriax.benchmarks.single_agent_mining.analysis import (
    render_level_video,
    save_mp4,
)
from factoriax.constraints import balance_cost
from factoriax.envs import FactoriaXEnv
from factoriax.levels import build_state
from factoriax.observations import local_array
from factoriax.rewards import sparse_mining_reward
from factoriax.state import EnvParams, EnvState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_NUM_ACTIONS = 6


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Config:
    """Training configuration for balanced gathering PPO.

    Attributes:
        hidden_dims: Shared MLP hidden layer sizes.
        num_envs: Parallel training environments.
        rollout_steps: Steps per env per iteration.
        total_timesteps: Total env steps (per level in sequential mode).
        learning_rate: Adam learning rate.
        gamma: Discount factor.
        gae_lambda: GAE smoothing.
        clip_eps: PPO clipping.
        value_coef: Value loss weight.
        entropy_coef: Entropy bonus weight.
        update_epochs: PPO epochs per batch.
        num_minibatches: Minibatches per epoch.
        max_grad_norm: Gradient clipping norm.
        normalize_obs: Online observation normalization.
        obs_radius: Local observation radius.
        lambda_cost: Lagrange multiplier for balance constraint penalty.
        balance_threshold: Ratio threshold for balance_cost.
        mode: Training mode: "independent", "sequential", or "mixed".
        seed: Random seed.
        save_path: Checkpoint directory.
        log_interval: Iterations between log lines.
        use_wandb: Enable W&B logging.
        wandb_project: W&B project name.
        wandb_run_name: W&B run name.
    """

    hidden_dims: tuple[int, ...] = (256, 256)
    num_envs: int = 64
    rollout_steps: int = 128
    total_timesteps: int = 5_000_000
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    update_epochs: int = 4
    num_minibatches: int = 8
    max_grad_norm: float = 0.5
    normalize_obs: bool = True
    obs_radius: int = 7
    lambda_cost: float = 0.5
    balance_threshold: float = 3.0
    mode: str = "sequential"
    seed: int = 0
    save_path: str | None = None
    log_interval: int = 10
    use_wandb: bool = False
    wandb_project: str = "factoriax-balanced"
    wandb_run_name: str | None = None


# ---------------------------------------------------------------------------
# Network (same as single_agent_ppo)
# ---------------------------------------------------------------------------


class ActorCritic(nn.Module):
    """Shared-trunk MLP with separate policy and value heads."""

    hidden_dims: tuple[int, ...]
    num_actions: int

    @nn.compact
    def __call__(self, obs: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Compute action logits and state value.

        Args:
            obs: Observation array.

        Returns:
            ``(logits, value)`` tuple.
        """
        x = obs.astype(jnp.float32)
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.LayerNorm()(x)
            x = nn.tanh(x)
        logits = nn.Dense(self.num_actions)(x)
        value = nn.Dense(1)(x).squeeze(-1)
        return logits, value


# ---------------------------------------------------------------------------
# Observation normalization
# ---------------------------------------------------------------------------


@struct.dataclass
class RunningStats:
    """Welford online mean/variance tracker."""

    mean: jax.Array
    var: jax.Array
    count: jax.Array


def init_running_stats(obs_dim: int) -> RunningStats:
    """Create zero-initialized running statistics."""
    return RunningStats(
        mean=jnp.zeros(obs_dim, dtype=jnp.float32),
        var=jnp.ones(obs_dim, dtype=jnp.float32),
        count=jnp.array(0, dtype=jnp.int32),
    )


def update_running_stats(
    stats: RunningStats,
    batch: jax.Array,
) -> RunningStats:
    """Welford parallel batch update."""
    n = batch.shape[0]
    batch_mean = batch.mean(axis=0)
    batch_var = batch.var(axis=0)
    total = stats.count + n
    delta = batch_mean - stats.mean
    new_mean = stats.mean + delta * (n / total)
    new_var = (
        stats.var * stats.count + batch_var * n + delta**2 * stats.count * n / total
    ) / total
    return RunningStats(mean=new_mean, var=new_var, count=total)


def normalize_obs(stats: RunningStats, obs: jax.Array) -> jax.Array:
    """Normalize and clip observations."""
    return jnp.clip(
        (obs - stats.mean) / jnp.sqrt(stats.var + 1e-8),
        -10.0,
        10.0,
    )


# ---------------------------------------------------------------------------
# Rollout storage
# ---------------------------------------------------------------------------


class Transition(NamedTuple):
    """One step of experience."""

    obs: jax.Array
    action: jax.Array
    log_prob: jax.Array
    value: jax.Array
    reward: jax.Array
    done: jax.Array


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------


def compute_gae(
    rewards: jax.Array,
    values: jax.Array,
    dones: jax.Array,
    last_value: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    """Compute GAE advantages and value targets."""
    not_done = 1.0 - dones.astype(jnp.float32)
    next_values = jnp.concatenate(
        [values[1:], last_value[None]],
        axis=0,
    )

    def _step(
        gae: jax.Array,
        xs: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        r, v, nd, nv = xs
        delta = r + gamma * nv * nd - v
        gae = delta + gamma * gae_lambda * nd * gae
        return gae, gae

    _, advantages = jax.lax.scan(
        _step,
        jnp.zeros_like(last_value),
        (
            rewards[::-1],
            values[::-1],
            not_done[::-1],
            next_values[::-1],
        ),
    )
    advantages = advantages[::-1]
    return advantages, advantages + values


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------


def make_train_fns(
    env: FactoriaXEnv,
    env_params: EnvParams,
    network: ActorCritic,
    optimizer: optax.GradientTransformation,
    config: Config,
    obs_fn: Callable[[EnvState, EnvParams], jax.Array],
    reset_fn: Callable[[jax.Array, EnvParams], EnvState],
    reward_fn: Callable[[EnvState, EnvState, EnvParams], jax.Array],
) -> tuple[Any, Any]:
    """Build JIT-compiled collect and update functions.

    Args:
        env: Environment instance.
        env_params: Environment parameters.
        network: Actor-critic module.
        optimizer: Optax optimizer.
        config: Training config.
        obs_fn: Observation function.
        reset_fn: Reset function for vectorized envs.
        reward_fn: Reward function (should include constraint penalty).

    Returns:
        ``(collect_fn, update_fn)`` closures.
    """
    num_envs = config.num_envs
    rollout_steps = config.rollout_steps

    vmap_step = jax.vmap(env.step_env, in_axes=(0, 0, 0, None))
    vmap_obs_fn = jax.vmap(obs_fn, in_axes=(0, None))
    vmap_reward_fn = jax.vmap(reward_fn, in_axes=(0, 0, None))

    def _select_states(
        dones: jax.Array,
        step_states: EnvState,
        reset_states: EnvState,
    ) -> EnvState:
        def _where(r: jax.Array, s: jax.Array) -> jax.Array:
            pad = dones.reshape((-1,) + (1,) * (s.ndim - 1))
            return jnp.where(pad, r, s)

        return jax.tree_util.tree_map(_where, reset_states, step_states)

    @jax.jit
    def collect(
        params: Any,
        obs_stats: RunningStats,
        env_states: EnvState,
        obs: jax.Array,
        rng: jax.Array,
    ) -> tuple[Transition, EnvState, jax.Array, jax.Array, jax.Array]:
        """Collect rollout_steps transitions from num_envs environments."""

        def rollout_step(
            carry: tuple[EnvState, jax.Array, jax.Array],
            _: None,
        ) -> tuple[
            tuple[EnvState, jax.Array, jax.Array],
            Transition,
        ]:
            states, cur_obs, step_rng = carry
            step_rng, key_act, key_step, key_reset = jax.random.split(
                step_rng,
                4,
            )
            norm = (
                normalize_obs(obs_stats, cur_obs) if config.normalize_obs else cur_obs
            )
            logits, values = network.apply(params, norm)
            actions = jax.random.categorical(key_act, logits)
            log_probs = jax.nn.log_softmax(logits)[jnp.arange(num_envs), actions]

            keys_step = jax.random.split(key_step, num_envs)
            prev_states = states
            _, next_states, _, dones, _ = vmap_step(
                keys_step,
                states,
                actions,
                env_params,
            )
            rewards = vmap_reward_fn(
                prev_states,
                next_states,
                env_params,
            )

            keys_reset = jax.random.split(key_reset, num_envs)
            reset_states = reset_fn(keys_reset, env_params)
            next_states = _select_states(
                dones,
                next_states,
                reset_states,
            )
            next_obs = vmap_obs_fn(next_states, env_params)

            transition = Transition(
                obs=cur_obs,
                action=actions,
                log_prob=log_probs,
                value=values,
                reward=rewards,
                done=dones,
            )
            return (next_states, next_obs, step_rng), transition

        (next_states, next_obs, rng), trajectories = jax.lax.scan(
            rollout_step,
            (env_states, obs, rng),
            None,
            length=rollout_steps,
        )
        norm_last = (
            normalize_obs(obs_stats, next_obs) if config.normalize_obs else next_obs
        )
        _, last_values = network.apply(params, norm_last)
        return trajectories, next_states, next_obs, last_values, rng

    @jax.jit
    def update(
        params: Any,
        opt_state: optax.OptState,
        obs_stats: RunningStats,
        flat_obs: jax.Array,
        flat_actions: jax.Array,
        flat_log_probs: jax.Array,
        flat_advantages: jax.Array,
        flat_returns: jax.Array,
        rng: jax.Array,
    ) -> tuple[Any, optax.OptState, dict[str, jax.Array], jax.Array]:
        """Run PPO epochs on a flattened trajectory batch."""
        batch_size = flat_obs.shape[0]
        mb_size = batch_size // config.num_minibatches

        def _loss(
            p: Any,
            obs: jax.Array,
            actions: jax.Array,
            old_lp: jax.Array,
            adv: jax.Array,
            rets: jax.Array,
        ) -> tuple[jax.Array, dict[str, jax.Array]]:
            norm = normalize_obs(obs_stats, obs) if config.normalize_obs else obs
            logits, values = network.apply(p, norm)
            log_probs_all = jax.nn.log_softmax(logits)
            lp = log_probs_all[jnp.arange(obs.shape[0]), actions]
            probs = jax.nn.softmax(logits)
            entropy = -(probs * log_probs_all).sum(axis=-1).mean()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            ratio = jnp.exp(lp - old_lp)
            pg_loss = -jnp.minimum(
                ratio * adv,
                jnp.clip(
                    ratio,
                    1.0 - config.clip_eps,
                    1.0 + config.clip_eps,
                )
                * adv,
            ).mean()
            value_loss = 0.5 * ((values - rets) ** 2).mean()
            total = (
                pg_loss + config.value_coef * value_loss - config.entropy_coef * entropy
            )
            metrics = {
                "loss/total": total,
                "loss/policy": pg_loss,
                "loss/value": value_loss,
                "loss/entropy": entropy,
                "misc/approx_kl": ((ratio - 1.0) - jnp.log(ratio)).mean(),
                "misc/clip_frac": (
                    (jnp.abs(ratio - 1.0) > config.clip_eps).astype(
                        jnp.float32,
                    )
                ).mean(),
            }
            return total, metrics

        def _minibatch_step(
            carry: tuple[Any, optax.OptState],
            mb: tuple[
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
            ],
        ) -> tuple[
            tuple[Any, optax.OptState],
            dict[str, jax.Array],
        ]:
            p, os = carry
            (loss, m), grads = jax.value_and_grad(_loss, has_aux=True)(
                p,
                *mb,
            )
            updates, new_os = optimizer.update(grads, os, p)
            return (optax.apply_updates(p, updates), new_os), m

        def _epoch(
            carry: tuple[Any, optax.OptState, jax.Array],
            _: None,
        ) -> tuple[
            tuple[Any, optax.OptState, jax.Array],
            dict[str, jax.Array],
        ]:
            p, os, epoch_rng = carry
            epoch_rng, key_perm = jax.random.split(epoch_rng)
            perm = jax.random.permutation(key_perm, batch_size)

            def _reshape(x: jax.Array) -> jax.Array:
                return x[perm].reshape(
                    (config.num_minibatches, mb_size) + x.shape[1:],
                )

            mbs = (
                _reshape(flat_obs),
                _reshape(flat_actions),
                _reshape(flat_log_probs),
                _reshape(flat_advantages),
                _reshape(flat_returns),
            )
            (p, os), metrics = jax.lax.scan(
                _minibatch_step,
                (p, os),
                mbs,
            )
            return (p, os, epoch_rng), metrics

        (params, opt_state, rng), metrics = jax.lax.scan(
            _epoch,
            (params, opt_state, rng),
            None,
            length=config.update_epochs,
        )
        metrics = jax.tree_util.tree_map(lambda x: x.mean(), metrics)
        return params, opt_state, metrics, rng

    return collect, update


# ---------------------------------------------------------------------------
# Training on a single level
# ---------------------------------------------------------------------------


def train_on_level(
    bench_level: BenchmarkLevel,
    config: Config,
    params: Any | None = None,
    opt_state: optax.OptState | None = None,
    obs_stats: RunningStats | None = None,
    rng: jax.Array | None = None,
    wandb_run: Any | None = None,
    level_tag: str = "",
) -> tuple[Any, optax.OptState, RunningStats, jax.Array]:
    """Train PPO on a single benchmark level.

    Args:
        bench_level: Level to train on.
        config: Training configuration.
        params: Network params to warm-start from, or None for fresh init.
        opt_state: Optimizer state to resume, or None for fresh init.
        obs_stats: Observation stats to resume, or None for fresh init.
        rng: PRNG key, or None to create from config.seed.
        wandb_run: Active W&B run for logging, or None.
        level_tag: Prefix for log keys (e.g. "L1/" for level 1).

    Returns:
        ``(params, opt_state, obs_stats, rng)`` after training.
    """
    if rng is None:
        rng = jax.random.PRNGKey(config.seed)

    env = FactoriaXEnv()
    env_params = bench_level.env_params

    level_state = build_state(bench_level.level, env_params)

    _obs_radius = config.obs_radius

    def _obs_fn(state: EnvState, params: EnvParams) -> jax.Array:
        return local_array(
            state,
            params,
            state.selected_player,
            _obs_radius,
        )

    obs_dim = int(_obs_fn(level_state, env_params).shape[0])

    # Constrained reward: mining reward minus lambda * balance penalty
    _lam = config.lambda_cost
    _threshold = config.balance_threshold
    _constraint_fn = functools.partial(
        balance_cost,
        threshold=_threshold,
    )

    def _reward_fn(
        prev: EnvState,
        new: EnvState,
        params: EnvParams,
    ) -> jax.Array:
        r = sparse_mining_reward(prev, new, params)
        cost = _constraint_fn(prev, new, params)
        return r - _lam * jnp.sum(cost)

    network = ActorCritic(
        hidden_dims=config.hidden_dims,
        num_actions=_NUM_ACTIONS,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.learning_rate),
    )

    if params is None:
        rng, key_init = jax.random.split(rng)
        params = network.init(key_init, jnp.zeros(obs_dim))
    if opt_state is None:
        opt_state = optimizer.init(params)
    if obs_stats is None:
        obs_stats = init_running_stats(obs_dim)

    def _reset_fn(
        keys: jax.Array,
        params: EnvParams,
    ) -> EnvState:
        def _broadcast(x: jax.Array) -> jax.Array:
            a = jnp.asarray(x)
            return jnp.broadcast_to(a[None], (config.num_envs,) + a.shape)

        return jax.tree_util.tree_map(_broadcast, level_state)

    collect_fn, update_fn = make_train_fns(
        env,
        env_params,
        network,
        optimizer,
        config,
        _obs_fn,
        _reset_fn,
        _reward_fn,
    )

    steps_per_iter = config.num_envs * config.rollout_steps
    total_iters = max(1, config.total_timesteps // steps_per_iter)
    current_step = 0

    rng, key_envs = jax.random.split(rng)
    keys_envs = jax.random.split(key_envs, config.num_envs)
    env_states = _reset_fn(keys_envs, env_params)
    obs = jax.vmap(_obs_fn, in_axes=(0, None))(env_states, env_params)

    logger.info(
        "Training on '%s' for %d iters (%s steps)",
        bench_level.name,
        total_iters,
        f"{total_iters * steps_per_iter:,}",
    )

    running_returns: deque[float] = deque(maxlen=200)
    running_ep_return = np.zeros(config.num_envs, dtype=np.float32)
    t_start = time.time()

    for it in range(total_iters):
        rng, key_collect, key_update = jax.random.split(rng, 3)

        trajectories, env_states, obs, last_values, _ = collect_fn(
            params,
            obs_stats,
            env_states,
            obs,
            key_collect,
        )

        flat_obs = trajectories.obs.reshape(-1, obs_dim)
        obs_stats = update_running_stats(obs_stats, flat_obs)

        advantages, returns = compute_gae(
            trajectories.reward,
            trajectories.value,
            trajectories.done,
            last_values,
            config.gamma,
            config.gae_lambda,
        )

        params, opt_state, metrics, _ = update_fn(
            params,
            opt_state,
            obs_stats,
            flat_obs,
            trajectories.action.reshape(-1),
            trajectories.log_prob.reshape(-1),
            advantages.reshape(-1),
            returns.reshape(-1),
            key_update,
        )

        current_step += steps_per_iter

        rewards_np = np.array(trajectories.reward)
        dones_np = np.array(trajectories.done)
        for t in range(rewards_np.shape[0]):
            running_ep_return += rewards_np[t]
            done_envs = np.where(dones_np[t])[0]
            for n in done_envs:
                running_returns.append(float(running_ep_return[n]))
                running_ep_return[n] = 0.0

        if (it + 1) % config.log_interval == 0 or it == total_iters - 1:
            elapsed = time.time() - t_start
            sps = current_step / elapsed
            mean_ret = float(np.mean(running_returns)) if running_returns else 0.0
            logger.info(
                "%s step=%d  sps=%.0f  mean_ret=%.3f  loss=%.4f  entropy=%.4f",
                level_tag,
                current_step,
                sps,
                mean_ret,
                float(metrics["loss/total"]),
                float(metrics["loss/entropy"]),
            )
            if wandb_run is not None:
                log_data = {
                    f"{level_tag}train/step": float(current_step),
                    f"{level_tag}train/sps": sps,
                    f"{level_tag}train/mean_return": mean_ret,
                    **{f"{level_tag}{k}": float(v) for k, v in metrics.items()},
                }
                wandb_run.log(log_data)

    return params, opt_state, obs_stats, rng


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    params: Any,
    network: ActorCritic,
    obs_stats: RunningStats,
    config: Config,
    wandb_run: Any | None = None,
) -> None:
    """Evaluate the trained policy on all benchmark levels.

    Uses stochastic sampling from the policy (not greedy argmax) to
    avoid degenerate behaviour when entropy is low.  Renders a video
    of each level and uploads to W&B if enabled.

    Args:
        params: Trained network parameters.
        network: Actor-critic module.
        obs_stats: Observation normalization stats.
        config: Training config (for obs settings).
        wandb_run: W&B run for logging, or None.
    """
    benchmark = BalancedGatheringBenchmark()
    _obs_radius = config.obs_radius
    _rng = jax.random.PRNGKey(config.seed + 999)

    def _obs_fn(
        state: EnvState,
        params: EnvParams,
        player_idx: int,
    ) -> jax.Array:
        return local_array(state, params, player_idx, _obs_radius)

    def _policy(obs: jax.Array) -> jax.Array:
        nonlocal _rng
        norm = normalize_obs(obs_stats, obs) if config.normalize_obs else obs
        logits, _ = network.apply(params, norm)
        _rng, key = jax.random.split(_rng)
        return jax.random.categorical(key, logits)

    constraint_fn = functools.partial(
        balance_cost,
        threshold=config.balance_threshold,
    )

    runner = BenchmarkRunner(seed=config.seed)
    result = runner.run(
        benchmark,
        policies=[_policy],
        obs_fn=_obs_fn,
        constraint_fn=constraint_fn,
    )

    out_dir = Path(config.save_path) if config.save_path else Path(".")

    logger.info("=== Evaluation Results ===")
    logger.info("Aggregate score: %.3f", result.aggregate_score)
    for lr in result.level_results:
        summary = {}
        if lr.constraint_costs is not None:
            summary = constraint_summary(lr.constraint_costs)
        logger.info(
            "  %s: score=%.1f  mined=%s  violations=%.0f",
            lr.level_name,
            lr.weighted_score,
            lr.items_mined,
            summary.get("steps_violated", 0),
        )
        if wandb_run is not None:
            log_data = {
                f"eval/{lr.level_name}/score": lr.weighted_score,
                f"eval/{lr.level_name}/coal": lr.items_mined.get("coal", 0),
                f"eval/{lr.level_name}/iron": lr.items_mined.get("iron", 0),
                f"eval/{lr.level_name}/copper": lr.items_mined.get("copper", 0),
            }
            if summary:
                log_data[f"eval/{lr.level_name}/total_cost"] = summary["total_cost"]
                log_data[f"eval/{lr.level_name}/steps_violated"] = summary[
                    "steps_violated"
                ]
                log_data[f"eval/{lr.level_name}/frac_violated"] = summary[
                    "fraction_violated"
                ]
            wandb_run.log(log_data)

    if wandb_run is not None:
        wandb_run.log({"eval/aggregate_score": result.aggregate_score})

    # Render per-level videos
    logger.info("Rendering per-level videos...")
    levels = benchmark.levels()
    for bl in levels:
        logger.info("  Rendering %s...", bl.name)
        frames = render_level_video(bl, _policy, seed=config.seed, obs_fn=_obs_fn)
        mp4_path = out_dir / f"eval_{bl.name}.mp4"
        save_mp4(frames, mp4_path)
        logger.info("  Saved %s (%d frames)", mp4_path, len(frames))
        if wandb_run is not None:
            try:
                import wandb

                wandb_run.log(
                    {
                        f"eval/{bl.name}/video": wandb.Video(str(mp4_path), fps=10),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("W&B video upload failed for %s: %s", bl.name, exc)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train(config: Config) -> None:
    """Run the full training pipeline.

    Args:
        config: Training configuration.
    """
    wandb_run = None
    if config.use_wandb:
        try:
            import wandb

            wandb_run = wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=dataclasses.asdict(config),
                tags=[
                    f"mode_{config.mode}",
                    f"lambda_{config.lambda_cost}",
                    f"threshold_{config.balance_threshold}",
                ],
            )
        except ImportError:
            logger.error("wandb not found, continuing without it")

    benchmark = BalancedGatheringBenchmark()
    levels = benchmark.levels()
    rng = jax.random.PRNGKey(config.seed)

    network = ActorCritic(
        hidden_dims=config.hidden_dims,
        num_actions=_NUM_ACTIONS,
    )

    if config.mode == "sequential":
        params = None
        opt_state = None
        obs_stats = None
        for i, bl in enumerate(levels):
            tag = f"{bl.name}/"
            logger.info(
                "=== Sequential: Level %d/%d — %s ===",
                i + 1,
                len(levels),
                bl.name,
            )
            params, opt_state, obs_stats, rng = train_on_level(
                bl,
                config,
                params,
                opt_state,
                obs_stats,
                rng,
                wandb_run,
                tag,
            )
    elif config.mode == "independent":
        params = None
        obs_stats = None
        for i, bl in enumerate(levels):
            tag = f"{bl.name}/"
            logger.info(
                "=== Independent: Level %d/%d — %s ===",
                i + 1,
                len(levels),
                bl.name,
            )
            params, _, obs_stats, rng = train_on_level(
                bl,
                config,
                None,
                None,
                None,
                rng,
                wandb_run,
                tag,
            )
    elif config.mode == "mixed":
        bl = levels[-1]
        tag = f"{bl.name}/"
        logger.info("=== Mixed: Training on %s ===", bl.name)
        params, _, obs_stats, rng = train_on_level(
            bl,
            config,
            None,
            None,
            None,
            rng,
            wandb_run,
            tag,
        )
    else:
        raise ValueError(f"Unknown mode: {config.mode!r}")

    logger.info("Training complete. Running evaluation...")
    evaluate(params, network, obs_stats, config, wandb_run)

    if config.save_path is not None:
        ckpt = Path(config.save_path) / "final.pkl"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "params": jax.device_get(
                jax.tree_util.tree_map(np.array, params),
            ),
            "obs_mean": np.array(obs_stats.mean),
            "obs_var": np.array(obs_stats.var),
            "obs_count": int(obs_stats.count),
        }
        with open(ckpt, "wb") as f:
            pickle.dump(data, f)
        logger.info("Checkpoint saved -> %s", ckpt)

    if wandb_run is not None:
        wandb_run.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments and run training."""
    p = argparse.ArgumentParser(
        description="Balanced gathering PPO training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--rollout-steps", type=int, default=128)
    p.add_argument("--total-steps", type=int, default=5_000_000)
    p.add_argument("--lr", type=float, default=2.5e-4, dest="learning_rate")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--lambda-cost", type=float, default=0.5)
    p.add_argument("--balance-threshold", type=float, default=3.0)
    p.add_argument("--obs-radius", type=int, default=7)
    p.add_argument(
        "--mode",
        type=str,
        default="sequential",
        choices=["independent", "sequential", "mixed"],
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-path", type=str, default=None)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--use-wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="factoriax-balanced")
    p.add_argument("--wandb-run-name", type=str, default=None)

    args = p.parse_args()
    config = Config(
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        total_timesteps=args.total_steps,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        entropy_coef=args.entropy_coef,
        lambda_cost=args.lambda_cost,
        balance_threshold=args.balance_threshold,
        obs_radius=args.obs_radius,
        mode=args.mode,
        seed=args.seed,
        save_path=args.save_path,
        log_interval=args.log_interval,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )
    train(config)


if __name__ == "__main__":
    main()
