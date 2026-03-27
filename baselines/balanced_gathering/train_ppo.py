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
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax

matplotlib.use("Agg")

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
from benchmarks.balanced_gathering import BalancedGatheringBenchmark
from benchmarks.balanced_gathering.scoring import constraint_summary
from benchmarks.core import BenchmarkLevel
from benchmarks.runner import BenchmarkRunner
from benchmarks.single_agent_mining.analysis import (
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

    Embeds a ``PPOConfig`` for shared hyperparameters and adds
    domain-specific fields for the balanced gathering benchmark.

    Attributes:
        ppo: Shared PPO hyperparameters.
        lambda_cost: Lagrange multiplier for balance constraint penalty.
        balance_threshold: Ratio threshold for balance_cost.
        mode: Training mode: "independent", "sequential", or "mixed".
    """

    ppo: PPOConfig = dataclasses.field(default_factory=PPOConfig)
    lambda_cost: float = 0.5
    balance_threshold: float = 3.0
    mode: str = "sequential"


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------


def make_collect_fn(
    env: FactoriaXEnv,
    env_params: EnvParams,
    network: ActorCritic,
    config: Config,
    obs_fn: Callable[[EnvState, EnvParams], jax.Array],
    reset_fn: Callable[[jax.Array, EnvParams], EnvState],
    reward_fn: Callable[[EnvState, EnvState, EnvParams], jax.Array],
) -> Any:
    """Build a JIT-compiled collect function for rollout collection.

    Args:
        env: Environment instance.
        env_params: Environment parameters.
        network: Actor-critic module.
        config: Training config.
        obs_fn: Observation function.
        reset_fn: Reset function for vectorized envs.
        reward_fn: Reward function (should include constraint penalty).

    Returns:
        ``collect_fn`` closure.
    """
    ppo = config.ppo
    num_envs = ppo.num_envs
    rollout_steps = ppo.rollout_steps

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
            norm = normalize_obs(obs_stats, cur_obs) if ppo.normalize_obs else cur_obs
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
            normalize_obs(obs_stats, next_obs) if ppo.normalize_obs else next_obs
        )
        _, last_values = network.apply(params, norm_last)
        return trajectories, next_states, next_obs, last_values, rng

    return collect


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
        rng: PRNG key, or None to create from config.ppo.seed.
        wandb_run: Active W&B run for logging, or None.
        level_tag: Prefix for log keys (e.g. "L1/" for level 1).

    Returns:
        ``(params, opt_state, obs_stats, rng)`` after training.
    """
    ppo = config.ppo

    if rng is None:
        rng = jax.random.PRNGKey(ppo.seed)

    env = FactoriaXEnv()
    env_params = bench_level.env_params

    level_state = build_state(bench_level.level, env_params)

    _obs_radius = ppo.obs_radius

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
        hidden_dims=ppo.hidden_dims,
        num_actions=_NUM_ACTIONS,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(ppo.max_grad_norm),
        optax.adam(ppo.learning_rate),
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
            return jnp.broadcast_to(a[None], (ppo.num_envs,) + a.shape)

        return jax.tree_util.tree_map(_broadcast, level_state)

    collect_fn = make_collect_fn(
        env,
        env_params,
        network,
        config,
        _obs_fn,
        _reset_fn,
        _reward_fn,
    )
    update_fn = make_update_fn(network, optimizer, ppo)

    steps_per_iter = ppo.num_envs * ppo.rollout_steps
    total_iters = max(1, ppo.total_steps // steps_per_iter)
    current_step = 0

    rng, key_envs = jax.random.split(rng)
    keys_envs = jax.random.split(key_envs, ppo.num_envs)
    env_states = _reset_fn(keys_envs, env_params)
    obs = jax.vmap(_obs_fn, in_axes=(0, None))(env_states, env_params)

    logger.info(
        "Training on '%s' for %d iters (%s steps)",
        bench_level.name,
        total_iters,
        f"{total_iters * steps_per_iter:,}",
    )

    running_returns: deque[float] = deque(maxlen=200)
    running_ep_return = np.zeros(ppo.num_envs, dtype=np.float32)
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
            ppo.gamma,
            ppo.gae_lambda,
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

        if (it + 1) % ppo.log_interval == 0 or it == total_iters - 1:
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
    ppo = config.ppo
    benchmark = BalancedGatheringBenchmark()
    _obs_radius = ppo.obs_radius
    _rng = jax.random.PRNGKey(ppo.seed + 999)

    def _obs_fn(
        state: EnvState,
        params: EnvParams,
        player_idx: int,
    ) -> jax.Array:
        return local_array(state, params, player_idx, _obs_radius)

    def _policy(obs: jax.Array) -> jax.Array:
        nonlocal _rng
        norm = normalize_obs(obs_stats, obs) if ppo.normalize_obs else obs
        logits, _ = network.apply(params, norm)
        _rng, key = jax.random.split(_rng)
        return jax.random.categorical(key, logits)

    constraint_fn = functools.partial(
        balance_cost,
        threshold=config.balance_threshold,
    )

    runner = BenchmarkRunner(seed=ppo.seed)
    result = runner.run(
        benchmark,
        policies=[_policy],
        obs_fn=_obs_fn,
        constraint_fn=constraint_fn,
    )

    out_dir = Path(ppo.save_path) if ppo.save_path else Path(".")

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
        frames = render_level_video(bl, _policy, seed=ppo.seed, obs_fn=_obs_fn)
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
    ppo = config.ppo
    wandb_run = None
    if ppo.use_wandb:
        try:
            import wandb

            wandb_run = wandb.init(
                project=ppo.wandb_project,
                name=ppo.wandb_run_name,
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
    rng = jax.random.PRNGKey(ppo.seed)

    network = ActorCritic(
        hidden_dims=ppo.hidden_dims,
        num_actions=_NUM_ACTIONS,
    )

    if config.mode == "sequential":
        params = None
        opt_state = None
        obs_stats = None
        for i, bl in enumerate(levels):
            tag = f"{bl.name}/"
            logger.info(
                "=== Sequential: Level %d/%d -- %s ===",
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
                "=== Independent: Level %d/%d -- %s ===",
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

    if ppo.save_path is not None:
        ckpt = Path(ppo.save_path) / "final.pkl"
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
    add_ppo_args(p)
    p.add_argument("--lambda-cost", type=float, default=0.5)
    p.add_argument("--balance-threshold", type=float, default=3.0)
    p.add_argument(
        "--mode",
        type=str,
        default="sequential",
        choices=["independent", "sequential", "mixed"],
    )

    args = p.parse_args()
    ppo = ppo_config_from_args(
        args,
        wandb_project="factoriax-balanced",
        total_steps=5_000_000,
    )
    config = Config(
        ppo=ppo,
        lambda_cost=args.lambda_cost,
        balance_threshold=args.balance_threshold,
        mode=args.mode,
    )
    train(config)


if __name__ == "__main__":
    main()
