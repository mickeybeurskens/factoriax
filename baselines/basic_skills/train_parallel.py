"""Train a PPO agent on mining and crafting simultaneously.

Each iteration collects rollouts from both levels with their own
reward functions, concatenates the transitions, and runs a single
PPO update on the combined batch. This lets the network learn both
skills at once without curriculum sequencing.

Usage::

    python -m baselines.basic_skills.train_parallel
    python -m baselines.basic_skills.train_parallel --total-steps 10_000_000
    python -m baselines.basic_skills.train_parallel --use-wandb
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import time
from collections import deque
from collections.abc import Callable
from typing import Any, NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from factoriax.benchmarks.basic_skills.benchmark import REWARD_FNS
from factoriax.benchmarks.basic_skills.levels import BASIC_SKILLS_LEVELS
from factoriax.benchmarks.core import BenchmarkLevel
from factoriax.constants import NUM_ACTIONS
from factoriax.envs import FactoriaXEnv
from factoriax.levels import build_state
from factoriax.observations import local_array
from factoriax.state import EnvState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_LEVEL_MAP = {bl.name: bl for bl in BASIC_SKILLS_LEVELS}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Config:
    """Hyperparameters for parallel two-level training.

    Attributes:
        hidden_dims: MLP hidden layer sizes.
        num_envs: Parallel environments per level.
        rollout_steps: Steps per env per update.
        total_steps: Total env steps (across both levels).
        learning_rate: Adam learning rate.
        gamma: Discount factor.
        gae_lambda: GAE lambda.
        clip_eps: PPO clipping epsilon.
        value_coef: Value loss weight.
        entropy_coef: Entropy bonus weight.
        update_epochs: PPO epochs per batch.
        num_minibatches: Minibatches per epoch.
        max_grad_norm: Gradient clipping norm.
        obs_radius: Local observation window half-width.
        seed: Random seed.
        log_interval: Iterations between log lines.
        use_wandb: Whether to log to Weights and Biases.
        wandb_project: W&B project name.
        wandb_run_name: W&B run name.
    """

    hidden_dims: tuple[int, ...] = (256, 256)
    num_envs: int = 64
    rollout_steps: int = 128
    total_steps: int = 10_000_000
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    update_epochs: int = 4
    num_minibatches: int = 8
    max_grad_norm: float = 0.5
    obs_radius: int = 7
    seed: int = 0
    log_interval: int = 10
    use_wandb: bool = False
    wandb_project: str = "factoriax-basic-skills"
    wandb_run_name: str | None = None


# ---------------------------------------------------------------------------
# Network + Transition + GAE
# ---------------------------------------------------------------------------


class ActorCritic(nn.Module):
    """Shared-trunk MLP with policy and value heads.

    Attributes:
        hidden_dims: Hidden layer sizes.
        num_actions: Discrete action count.
    """

    hidden_dims: tuple[int, ...]
    num_actions: int

    @nn.compact
    def __call__(self, obs: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Forward pass.

        Args:
            obs: Observation array.

        Returns:
            Tuple of (logits, value).
        """
        x = obs.astype(jnp.float32)
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.LayerNorm()(x)
            x = nn.tanh(x)
        logits = nn.Dense(self.num_actions)(x)
        value = nn.Dense(1)(x).squeeze(-1)
        return logits, value


class Transition(NamedTuple):
    """One step of rollout data."""

    obs: jax.Array
    action: jax.Array
    log_prob: jax.Array
    value: jax.Array
    reward: jax.Array
    done: jax.Array


def compute_gae(
    rewards: jax.Array,
    values: jax.Array,
    dones: jax.Array,
    last_value: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    """Compute generalized advantage estimates and value targets.

    Args:
        rewards: Shape (T, num_envs).
        values: Shape (T, num_envs).
        dones: Shape (T, num_envs).
        last_value: Shape (num_envs,).
        gamma: Discount factor.
        gae_lambda: GAE lambda.

    Returns:
        Tuple of (advantages, returns), each shape (T, num_envs).
    """
    not_done = 1.0 - dones.astype(jnp.float32)
    next_values = jnp.concatenate([values[1:], last_value[None]], axis=0)

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
        (rewards[::-1], values[::-1], not_done[::-1], next_values[::-1]),
    )
    advantages = advantages[::-1]
    return advantages, advantages + values


# ---------------------------------------------------------------------------
# Per-level collector builder
# ---------------------------------------------------------------------------


def _make_collector(
    env: FactoriaXEnv,
    bench_level: BenchmarkLevel,
    reward_fn: Callable,
    network: ActorCritic,
    config: Config,
) -> tuple[Callable, Callable]:
    """Build a JIT-compiled collection function for one level.

    Args:
        env: FactoriaX environment instance.
        bench_level: Level to collect from.
        reward_fn: Reward function for this level.
        network: Actor-critic network.
        config: Training configuration.

    Returns:
        Tuple of ``(collect_fn, init_fn)``.
    """
    env_params = bench_level.env_params
    level_state = build_state(bench_level.level, env_params)

    vmap_step = jax.vmap(env.step_env, in_axes=(0, 0, 0, None))
    vmap_reward = jax.vmap(reward_fn, in_axes=(0, 0, None))

    def _obs(state: EnvState) -> jax.Array:
        return local_array(
            state, env_params, state.selected_player, config.obs_radius
        )

    vmap_obs = jax.vmap(_obs)

    def _broadcast(x: jax.Array) -> jax.Array:
        a = jnp.asarray(x)
        return jnp.broadcast_to(a[None], (config.num_envs,) + a.shape)

    fixed_states = jax.tree_util.tree_map(_broadcast, level_state)

    def init() -> tuple[EnvState, jax.Array]:
        """Return initial batched states and observations."""
        return fixed_states, vmap_obs(fixed_states)

    def _normalize(
        obs: jax.Array, mean: jax.Array, var: jax.Array
    ) -> jax.Array:
        return jnp.clip(
            (obs - mean) / jnp.sqrt(var + 1e-8), -10.0, 10.0
        )

    @jax.jit
    def collect(
        net_params: Any,
        env_states: EnvState,
        obs: jax.Array,
        step_rng: jax.Array,
        obs_mean: jax.Array,
        obs_var: jax.Array,
    ) -> tuple[Transition, EnvState, jax.Array, jax.Array, jax.Array]:
        """Collect rollout_steps transitions from num_envs environments.

        Args:
            net_params: Network parameters.
            env_states: Current batched env states.
            obs: Current batched observations.
            step_rng: PRNG key.
            obs_mean: Running observation mean.
            obs_var: Running observation variance.

        Returns:
            Tuple of (transitions, next_states, next_obs,
            last_values, next_rng).
        """

        def rollout_step(
            carry: tuple[EnvState, jax.Array, jax.Array],
            _: None,
        ) -> tuple[
            tuple[EnvState, jax.Array, jax.Array], Transition
        ]:
            states, cur_obs, rng = carry
            rng, key_act, key_step = jax.random.split(rng, 3)

            logits, values = network.apply(
                net_params, _normalize(cur_obs, obs_mean, obs_var)
            )
            actions = jax.random.categorical(key_act, logits)
            log_probs = jax.nn.log_softmax(logits)[
                jnp.arange(config.num_envs), actions
            ]

            keys_step = jax.random.split(key_step, config.num_envs)
            prev_states = states
            _, next_states, _, dones, _ = vmap_step(
                keys_step, states, actions, env_params
            )
            rewards = vmap_reward(prev_states, next_states, env_params)

            def _where(r: jax.Array, s: jax.Array) -> jax.Array:
                pad = dones.reshape((-1,) + (1,) * (s.ndim - 1))
                return jnp.where(pad, r, s)

            next_states = jax.tree_util.tree_map(
                _where, fixed_states, next_states
            )
            next_obs = vmap_obs(next_states)

            return (next_states, next_obs, rng), Transition(
                obs=cur_obs,
                action=actions,
                log_prob=log_probs,
                value=values,
                reward=rewards,
                done=dones,
            )

        (next_states, next_obs, rng), trajectories = jax.lax.scan(
            rollout_step,
            (env_states, obs, step_rng),
            None,
            length=config.rollout_steps,
        )
        _, last_values = network.apply(
            net_params, _normalize(next_obs, obs_mean, obs_var)
        )
        return trajectories, next_states, next_obs, last_values, rng

    return collect, init


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(config: Config) -> None:
    """Train one policy on mining and crafting simultaneously.

    Each iteration collects rollouts from both levels (each with its
    own reward function), concatenates the transitions, and runs a
    single PPO update on the combined batch.

    Args:
        config: Training configuration.
    """
    mine_level = _LEVEL_MAP["mine_resources"]
    craft_level = _LEVEL_MAP["craft_chests"]
    levels = [mine_level, craft_level]
    reward_fns = [REWARD_FNS["mine_resources"], REWARD_FNS["craft_chests"]]

    env = FactoriaXEnv()

    # Derive obs_dim.
    sample_state = build_state(mine_level.level, mine_level.env_params)
    obs_dim = int(
        local_array(
            sample_state, mine_level.env_params, 0, config.obs_radius
        ).shape[0]
    )
    logger.info(
        "obs_dim=%d  num_actions=%d  num_envs=%d/level  levels=%d",
        obs_dim, NUM_ACTIONS, config.num_envs, len(levels),
    )

    network = ActorCritic(
        hidden_dims=config.hidden_dims, num_actions=NUM_ACTIONS
    )
    rng = jax.random.PRNGKey(config.seed)
    rng, key_init = jax.random.split(rng)
    params = network.init(key_init, jnp.zeros(obs_dim))
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.learning_rate),
    )
    opt_state = optimizer.init(params)

    obs_mean = jnp.zeros(obs_dim, dtype=jnp.float32)
    obs_var = jnp.ones(obs_dim, dtype=jnp.float32)
    obs_count = jnp.array(0, dtype=jnp.int32)

    def _normalize(
        obs: jax.Array, mean: jax.Array, var: jax.Array
    ) -> jax.Array:
        return jnp.clip(
            (obs - mean) / jnp.sqrt(var + 1e-8), -10.0, 10.0
        )

    # Build per-level collectors.
    collectors: list[Callable] = []
    level_states: list[EnvState] = []
    level_obs: list[jax.Array] = []
    for bl, rfn in zip(levels, reward_fns):
        collect_fn, init_fn = _make_collector(
            env, bl, rfn, network, config
        )
        states, obs = init_fn()
        collectors.append(collect_fn)
        level_states.append(states)
        level_obs.append(obs)

    # PPO update (JIT-compiled).
    @jax.jit
    def update(
        net_params: Any,
        os: optax.OptState,
        flat_obs: jax.Array,
        flat_actions: jax.Array,
        flat_log_probs: jax.Array,
        flat_advantages: jax.Array,
        flat_returns: jax.Array,
        update_rng: jax.Array,
        cur_obs_mean: jax.Array,
        cur_obs_var: jax.Array,
    ) -> tuple[Any, optax.OptState, dict[str, jax.Array], jax.Array]:
        """PPO update on a combined batch from both levels."""
        batch_size = flat_obs.shape[0]
        mb_size = batch_size // config.num_minibatches

        def _loss(p, mb_obs, mb_act, mb_old_lp, mb_adv, mb_rets):
            logits, values = network.apply(
                p, _normalize(mb_obs, cur_obs_mean, cur_obs_var)
            )
            lp_all = jax.nn.log_softmax(logits)
            lp = lp_all[jnp.arange(mb_obs.shape[0]), mb_act]
            probs = jax.nn.softmax(logits)
            entropy = -(probs * lp_all).sum(axis=-1).mean()
            adv_n = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
            ratio = jnp.exp(lp - mb_old_lp)
            pg_loss = -jnp.minimum(
                ratio * adv_n,
                jnp.clip(
                    ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps
                ) * adv_n,
            ).mean()
            v_loss = 0.5 * ((values - mb_rets) ** 2).mean()
            total = (
                pg_loss
                + config.value_coef * v_loss
                - config.entropy_coef * entropy
            )
            return total, {
                "loss": total, "pg": pg_loss,
                "vf": v_loss, "ent": entropy,
            }

        def _mb_step(carry, mb):
            p, o = carry
            (_, m), grads = jax.value_and_grad(_loss, has_aux=True)(
                p, *mb
            )
            updates, new_o = optimizer.update(grads, o, p)
            return (optax.apply_updates(p, updates), new_o), m

        def _epoch(carry, _):
            p, o, epoch_rng = carry
            epoch_rng, key_perm = jax.random.split(epoch_rng)
            perm = jax.random.permutation(key_perm, batch_size)

            def _reshape(x):
                return x[perm].reshape(
                    (config.num_minibatches, mb_size) + x.shape[1:]
                )

            mbs = (
                _reshape(flat_obs), _reshape(flat_actions),
                _reshape(flat_log_probs), _reshape(flat_advantages),
                _reshape(flat_returns),
            )
            (p, o), metrics = jax.lax.scan(_mb_step, (p, o), mbs)
            return (p, o, epoch_rng), metrics

        (net_params, os, update_rng), metrics = jax.lax.scan(
            _epoch, (net_params, os, update_rng), None,
            length=config.update_epochs,
        )
        metrics = jax.tree_util.tree_map(lambda x: x.mean(), metrics)
        return net_params, os, metrics, update_rng

    # Wandb.
    wandb_run = None
    if config.use_wandb:
        try:
            import wandb  # type: ignore[import-untyped]

            wandb_run = wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name or "parallel_mine_craft",
                config=dataclasses.asdict(config),
                tags=["basic_skills", "parallel"],
            )
        except ImportError:
            logger.error("wandb not found. Install with: uv add wandb")

    # Training loop.
    steps_per_level = config.num_envs * config.rollout_steps
    steps_per_iter = steps_per_level * len(levels)
    num_iters = max(1, config.total_steps // steps_per_iter)

    logger.info(
        "%d steps/iter (%d/level x %d levels), %d iters",
        steps_per_iter, steps_per_level, len(levels), num_iters,
    )

    per_level_returns: dict[str, deque[float]] = {
        bl.name: deque(maxlen=500) for bl in levels
    }
    running_return = np.zeros(
        config.num_envs * len(levels), dtype=np.float32
    )
    current_step = 0
    t_start = time.time()

    for it in range(num_iters):
        rng, key_update = jax.random.split(rng)

        all_obs: list[jax.Array] = []
        all_actions: list[jax.Array] = []
        all_log_probs: list[jax.Array] = []
        all_adv: list[jax.Array] = []
        all_ret: list[jax.Array] = []

        for l_idx, (collect_fn, bl) in enumerate(
            zip(collectors, levels)
        ):
            rng, key_l = jax.random.split(rng)
            traj, level_states[l_idx], level_obs[l_idx], last_vals, _ = (
                collect_fn(
                    params, level_states[l_idx], level_obs[l_idx],
                    key_l, obs_mean, obs_var,
                )
            )

            adv, ret = compute_gae(
                traj.reward, traj.value, traj.done, last_vals,
                config.gamma, config.gae_lambda,
            )

            all_obs.append(traj.obs.reshape(-1, obs_dim))
            all_actions.append(traj.action.reshape(-1))
            all_log_probs.append(traj.log_prob.reshape(-1))
            all_adv.append(adv.reshape(-1))
            all_ret.append(ret.reshape(-1))

            # Track per-level returns.
            rewards_np = np.array(traj.reward)
            dones_np = np.array(traj.done)
            offset = l_idx * config.num_envs
            for t in range(rewards_np.shape[0]):
                running_return[offset:offset + config.num_envs] += (
                    rewards_np[t]
                )
                for n in np.where(dones_np[t])[0]:
                    per_level_returns[bl.name].append(
                        float(running_return[offset + n])
                    )
                    running_return[offset + n] = 0.0

        flat_obs = jnp.concatenate(all_obs)

        # Welford update.
        n = flat_obs.shape[0]
        batch_mean = flat_obs.mean(axis=0)
        batch_var = flat_obs.var(axis=0)
        total = obs_count + n
        delta = batch_mean - obs_mean
        obs_mean = obs_mean + delta * (n / total)
        obs_var = (
            obs_var * obs_count + batch_var * n
            + delta**2 * obs_count * n / total
        ) / total
        obs_count = total

        params, opt_state, metrics, _ = update(
            params, opt_state,
            flat_obs,
            jnp.concatenate(all_actions),
            jnp.concatenate(all_log_probs),
            jnp.concatenate(all_adv),
            jnp.concatenate(all_ret),
            key_update,
            obs_mean, obs_var,
        )
        current_step += steps_per_iter

        if (it + 1) % config.log_interval == 0 or it == num_iters - 1:
            elapsed = time.time() - t_start
            sps = current_step / elapsed
            ret_parts = []
            log_data: dict[str, float] = {
                "train/step": float(current_step),
                "train/sps": sps,
                **{
                    f"train/{k}": float(v)
                    for k, v in metrics.items()
                },
            }
            for bl in levels:
                rets = per_level_returns[bl.name]
                mean_r = (
                    float(np.mean(list(rets))) if rets else 0.0
                )
                log_data[f"train/{bl.name}/mean_ep_return"] = mean_r
                ret_parts.append(f"{bl.name}={mean_r:.2f}")

            logger.info(
                "iter=%d/%d  step=%dk  sps=%.0f  %s"
                "  loss=%.4f  ent=%.4f",
                it + 1, num_iters, current_step // 1000, sps,
                "  ".join(ret_parts),
                float(metrics["loss"]), float(metrics["ent"]),
            )
            if wandb_run is not None:
                wandb_run.log(log_data, step=current_step)

    elapsed = time.time() - t_start
    logger.info(
        "Done. %dk steps in %.1fs (%.0f sps).",
        current_step // 1000, elapsed, current_step / elapsed,
    )

    if wandb_run is not None:
        wandb_run.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments and run parallel training."""
    parser = argparse.ArgumentParser(
        description=(
            "Train PPO on mining and crafting simultaneously."
        ),
    )
    parser.add_argument(
        "--num-envs", type=int, default=64,
        help="Parallel envs per level (default: 64).",
    )
    parser.add_argument(
        "--total-steps", type=int, default=10_000_000,
        help="Total env steps across both levels (default: 10M).",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed (default: 0).",
    )
    parser.add_argument(
        "--log-interval", type=int, default=10,
        help="Iterations between log lines (default: 10).",
    )
    parser.add_argument(
        "--use-wandb", action="store_true",
        help="Log to Weights and Biases.",
    )
    parser.add_argument(
        "--wandb-project", type=str,
        default="factoriax-basic-skills",
    )
    parser.add_argument(
        "--wandb-run-name", type=str, default=None,
    )
    args = parser.parse_args()

    config = Config(
        num_envs=args.num_envs,
        total_steps=args.total_steps,
        seed=args.seed,
        log_interval=args.log_interval,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )
    train(config)


if __name__ == "__main__":
    main()
