"""Train a PPO agent on a single basic_skills level.

Focused training script for verifying that each task is individually
learnable. Strips away the multi-level mixing, evaluation, and video
rendering from ``train_ppo.py`` to give a fast feedback loop.

Usage::

    python -m baselines.basic_skills.train_single mine_resources
    python -m baselines.basic_skills.train_single craft_pallets --total-steps 5_000_000
    python -m baselines.basic_skills.train_single fill_pallet --num-envs 128
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import time
from collections import deque
from typing import Any, NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from baselines.ppo.network import VisionActorCritic
from factoriax.benchmarks.basic_skills.benchmark import REWARD_FNS
from factoriax.benchmarks.basic_skills.levels import BASIC_SKILLS_LEVELS
from factoriax.benchmarks.core import BenchmarkLevel
from factoriax.constants import NUM_ACTIONS, Action, ItemType
from factoriax.envs import FactoriaXEnv
from factoriax.jax_renderer import JaxRenderer
from factoriax.levels import build_state
from factoriax.observations import local_array
from factoriax.state import EnvState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

LEVEL_NAMES: list[str] = [bl.name for bl in BASIC_SKILLS_LEVELS]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Config:
    """Training hyperparameters.

    Attributes:
        level_name: Which basic_skills level to train on.
        hidden_dims: MLP hidden layer sizes.
        num_envs: Parallel environments.
        rollout_steps: Steps per env per update.
        total_steps: Total env steps to train for.
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
        obs_type: Observation type, "vector" or "vision".
        tile_px: Pixel size per tile for vision observations.
        seed: Random seed.
        log_interval: Iterations between log lines.
    """

    level_name: str = "mine_ores"
    hidden_dims: tuple[int, ...] = (256, 256)
    num_envs: int = 64
    rollout_steps: int = 128
    total_steps: int = 5_000_000
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
    obs_type: str = "vector"
    tile_px: int = 8
    action_mask: list[str] | None = None
    use_wandb: bool = False
    wandb_project: str = "factoriax-basic-skills"
    wandb_run_name: str | None = None


# ---------------------------------------------------------------------------
# Network
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
# Training
# ---------------------------------------------------------------------------


def train(config: Config) -> None:
    """Train a PPO agent on one level and log progress.

    Args:
        config: Training configuration.
    """
    # Resolve level.
    level_map = {bl.name: bl for bl in BASIC_SKILLS_LEVELS}
    if config.level_name not in level_map:
        raise ValueError(
            f"Unknown level {config.level_name!r}. Choose from: {LEVEL_NAMES}"
        )
    bench_level: BenchmarkLevel = level_map[config.level_name]
    reward_fn = REWARD_FNS[config.level_name]
    env_params = bench_level.env_params

    # Wandb.
    wandb_run = None
    if config.use_wandb:
        try:
            import wandb  # type: ignore[import-untyped]

            run_name = config.wandb_run_name or f"single_{config.level_name}"
            wandb_run = wandb.init(
                project=config.wandb_project,
                name=run_name,
                config=dataclasses.asdict(config),
                tags=[
                    "basic_skills",
                    "single",
                    config.level_name,
                    config.obs_type,
                    (
                        f"obs_local_r{config.obs_radius}"
                        if config.obs_type == "vector"
                        else f"tile_px{config.tile_px}"
                    ),
                ],
            )
        except ImportError:
            logger.error("wandb not found. Install with: uv add wandb")

    # Build env and network.
    env = FactoriaXEnv(
        achievement_fn=bench_level.achievement_fn
        if bench_level.achievement_fn is not None
        else None,
    )
    level_state = build_state(bench_level.level, env_params)
    use_vision = config.obs_type == "vision"

    # Create renderer for vision mode (None for vector).
    renderer: JaxRenderer | None = None
    if use_vision:
        renderer = JaxRenderer(tile_px=config.tile_px)
        dummy_obs = renderer.jit_render_map(level_state)
        obs_shape: tuple[int, ...] = dummy_obs.shape
        network: ActorCritic | VisionActorCritic = VisionActorCritic(
            num_actions=NUM_ACTIONS,
        )
    else:
        obs_dim = int(
            local_array(level_state, env_params, 0, config.obs_radius).shape[0]
        )
        obs_shape = (obs_dim,)
        network = ActorCritic(hidden_dims=config.hidden_dims, num_actions=NUM_ACTIONS)

    logger.info(
        "Level: %s  obs_type=%s  obs_shape=%s  num_actions=%d  num_envs=%d",
        config.level_name,
        config.obs_type,
        obs_shape,
        NUM_ACTIONS,
        config.num_envs,
    )

    # Build action mask: -1e9 on disallowed actions, 0 on allowed.
    if config.action_mask is not None:
        allowed = {Action[n] for n in config.action_mask}
        logit_mask = jnp.array(
            [0.0 if Action(i) in allowed else -1e9 for i in range(NUM_ACTIONS)],
            dtype=jnp.float32,
        )
        logger.info(
            "Action mask: %d/%d actions allowed (%s)",
            len(allowed),
            NUM_ACTIONS,
            ", ".join(config.action_mask),
        )
    else:
        logit_mask = jnp.zeros(NUM_ACTIONS, dtype=jnp.float32)

    rng = jax.random.PRNGKey(config.seed)
    rng, key_init = jax.random.split(rng)
    params = network.init(key_init, jnp.zeros(obs_shape))
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.learning_rate),
    )
    opt_state = optimizer.init(params)

    # Running observation stats for normalization (vector only).
    obs_mean = jnp.zeros(obs_shape, dtype=jnp.float32)
    obs_var = jnp.ones(obs_shape, dtype=jnp.float32)
    obs_count = jnp.array(0, dtype=jnp.int32)

    if use_vision:

        def _normalize(obs: jax.Array) -> jax.Array:
            """Identity for vision; the CNN normalizes internally."""
            return obs
    else:

        def _normalize(obs: jax.Array) -> jax.Array:
            """Welford running-stats normalization for vector obs."""
            return jnp.clip(
                (obs - obs_mean) / jnp.sqrt(obs_var + 1e-8),
                -10.0,
                10.0,
            )

    # Broadcast initial state to num_envs copies.
    def _broadcast(x: jax.Array) -> jax.Array:
        a = jnp.asarray(x)
        return jnp.broadcast_to(a[None], (config.num_envs,) + a.shape)

    fixed_states = jax.tree_util.tree_map(_broadcast, level_state)

    # Vmapped env operations.
    vmap_step = jax.vmap(env.step_env, in_axes=(0, 0, 0, None))
    vmap_reward = jax.vmap(reward_fn, in_axes=(0, 0, None))

    if use_vision:
        assert renderer is not None
        vmap_obs = renderer.vmap_render_map
    else:

        def _obs(state: EnvState) -> jax.Array:
            return local_array(
                state,
                env_params,
                state.selected_player,
                config.obs_radius,
            )

        vmap_obs = jax.vmap(_obs)

    # JIT-compiled collection.
    @jax.jit
    def collect(
        net_params: Any,
        env_states: EnvState,
        obs: jax.Array,
        step_rng: jax.Array,
    ) -> tuple[Transition, EnvState, jax.Array, jax.Array, jax.Array]:
        """Collect rollout_steps transitions from num_envs environments.

        Args:
            net_params: Network parameters.
            env_states: Current batched env states.
            obs: Current batched observations.
            step_rng: PRNG key.

        Returns:
            Tuple of (transitions, next_states, next_obs, last_values,
            next_rng).
        """

        def rollout_step(
            carry: tuple[EnvState, jax.Array, jax.Array],
            _: None,
        ) -> tuple[tuple[EnvState, jax.Array, jax.Array], Transition]:
            states, cur_obs, rng = carry
            rng, key_act, key_step = jax.random.split(rng, 3)

            logits, values = network.apply(net_params, _normalize(cur_obs))
            masked_logits = logits + logit_mask
            actions = jax.random.categorical(key_act, masked_logits)
            log_probs = jax.nn.log_softmax(masked_logits)[
                jnp.arange(config.num_envs), actions
            ]

            keys_step = jax.random.split(key_step, config.num_envs)
            prev_states = states
            _, next_states, _, dones, _ = vmap_step(
                keys_step, states, actions, env_params
            )
            rewards = vmap_reward(prev_states, next_states, env_params)

            # Reset done environments to the fixed initial state.
            def _where(r: jax.Array, s: jax.Array) -> jax.Array:
                pad = dones.reshape((-1,) + (1,) * (s.ndim - 1))
                return jnp.where(pad, r, s)

            next_states = jax.tree_util.tree_map(_where, fixed_states, next_states)
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
        _, last_values = network.apply(net_params, _normalize(next_obs))
        return trajectories, next_states, next_obs, last_values, rng

    # JIT-compiled PPO update.
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
    ) -> tuple[Any, optax.OptState, dict[str, jax.Array], jax.Array]:
        """Run PPO update epochs on a flattened trajectory batch.

        Args:
            net_params: Current network parameters.
            os: Current optimizer state.
            flat_obs: Flattened observations.
            flat_actions: Flattened actions.
            flat_log_probs: Flattened old log-probs.
            flat_advantages: Flattened advantages.
            flat_returns: Flattened value targets.
            update_rng: PRNG key.

        Returns:
            Tuple of (new_params, new_opt_state, metrics, next_rng).
        """
        batch_size = flat_obs.shape[0]
        mb_size = batch_size // config.num_minibatches

        def _loss(
            p: Any,
            mb_obs: jax.Array,
            mb_actions: jax.Array,
            mb_old_lp: jax.Array,
            mb_adv: jax.Array,
            mb_rets: jax.Array,
        ) -> tuple[jax.Array, dict[str, jax.Array]]:
            logits, values = network.apply(p, _normalize(mb_obs))
            masked_logits = logits + logit_mask
            lp_all = jax.nn.log_softmax(masked_logits)
            lp = lp_all[jnp.arange(mb_obs.shape[0]), mb_actions]
            probs = jax.nn.softmax(masked_logits)
            entropy = -(probs * lp_all).sum(axis=-1).mean()
            adv_n = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
            ratio = jnp.exp(lp - mb_old_lp)
            pg_loss = -jnp.minimum(
                ratio * adv_n,
                jnp.clip(
                    ratio,
                    1.0 - config.clip_eps,
                    1.0 + config.clip_eps,
                )
                * adv_n,
            ).mean()
            v_loss = 0.5 * ((values - mb_rets) ** 2).mean()
            total = pg_loss + config.value_coef * v_loss - config.entropy_coef * entropy
            return total, {
                "loss": total,
                "pg": pg_loss,
                "vf": v_loss,
                "ent": entropy,
            }

        def _mb_step(
            carry: tuple[Any, optax.OptState],
            mb: tuple,
        ) -> tuple[tuple[Any, optax.OptState], dict]:
            p, o = carry
            (_, m), grads = jax.value_and_grad(_loss, has_aux=True)(p, *mb)
            updates, new_o = optimizer.update(grads, o, p)
            return (optax.apply_updates(p, updates), new_o), m

        def _epoch(
            carry: tuple[Any, optax.OptState, jax.Array],
            _: None,
        ) -> tuple[tuple[Any, optax.OptState, jax.Array], dict]:
            p, o, epoch_rng = carry
            epoch_rng, key_perm = jax.random.split(epoch_rng)
            perm = jax.random.permutation(key_perm, batch_size)

            def _reshape(x: jax.Array) -> jax.Array:
                return x[perm].reshape((config.num_minibatches, mb_size) + x.shape[1:])

            mbs = (
                _reshape(flat_obs),
                _reshape(flat_actions),
                _reshape(flat_log_probs),
                _reshape(flat_advantages),
                _reshape(flat_returns),
            )
            (p, o), metrics = jax.lax.scan(_mb_step, (p, o), mbs)
            return (p, o, epoch_rng), metrics

        (net_params, os, update_rng), metrics = jax.lax.scan(
            _epoch,
            (net_params, os, update_rng),
            None,
            length=config.update_epochs,
        )
        metrics = jax.tree_util.tree_map(lambda x: x.mean(), metrics)
        return net_params, os, metrics, update_rng

    # Training loop.
    steps_per_iter = config.num_envs * config.rollout_steps
    num_iters = max(1, config.total_steps // steps_per_iter)
    logger.info(
        "%d steps/iter, %d iters, %d total steps",
        steps_per_iter,
        num_iters,
        num_iters * steps_per_iter,
    )

    env_states = fixed_states
    obs = vmap_obs(env_states)
    ep_returns: deque[float] = deque(maxlen=500)
    running_return = np.zeros(config.num_envs, dtype=np.float32)
    current_step = 0
    t_start = time.time()

    for it in range(num_iters):
        rng, key_collect, key_update = jax.random.split(rng, 3)

        traj, env_states, obs, last_vals, _ = collect(
            params, env_states, obs, key_collect
        )

        adv, ret = compute_gae(
            traj.reward,
            traj.value,
            traj.done,
            last_vals,
            config.gamma,
            config.gae_lambda,
        )

        flat_obs = traj.obs.reshape((-1,) + obs_shape)

        # Update running obs stats (Welford, vector only).
        if not use_vision:
            n = flat_obs.shape[0]
            batch_mean = flat_obs.mean(axis=0)
            batch_var = flat_obs.var(axis=0)
            total = obs_count + n
            delta = batch_mean - obs_mean
            obs_mean = obs_mean + delta * (n / total)
            obs_var = (
                obs_var * obs_count + batch_var * n + delta**2 * obs_count * n / total
            ) / total
            obs_count = total

        params, opt_state, metrics, _ = update(
            params,
            opt_state,
            flat_obs,
            traj.action.reshape(-1),
            traj.log_prob.reshape(-1),
            adv.reshape(-1),
            ret.reshape(-1),
            key_update,
        )
        current_step += steps_per_iter

        # Track episode returns.
        rewards_np = np.array(traj.reward)
        dones_np = np.array(traj.done)
        for t in range(rewards_np.shape[0]):
            running_return += rewards_np[t]
            for n in np.where(dones_np[t])[0]:
                ep_returns.append(float(running_return[n]))
                running_return[n] = 0.0

        if (it + 1) % config.log_interval == 0 or it == num_iters - 1:
            elapsed = time.time() - t_start
            sps = current_step / elapsed
            mean_ret = float(np.mean(list(ep_returns))) if ep_returns else 0.0

            # Diagnostics: action distribution and reward stats.
            actions_np = np.array(traj.action).ravel()
            act_counts = np.bincount(actions_np, minlength=NUM_ACTIONS)
            top3 = np.argsort(act_counts)[::-1][:3]
            act_str = " ".join(
                f"{Action(a).name}={act_counts[a] / len(actions_np) * 100:.0f}%"
                for a in top3
            )
            total_reward_batch = float(np.sum(rewards_np))
            nonzero_reward_steps = int(np.sum(rewards_np > 0))

            logger.info(
                "iter=%d/%d  step=%dk  sps=%.0f"
                "  ret=%.2f  loss=%.4f  ent=%.4f"
                "  | r_batch=%.0f r_steps=%d"
                "  | %s",
                it + 1,
                num_iters,
                current_step // 1000,
                sps,
                mean_ret,
                float(metrics["loss"]),
                float(metrics["ent"]),
                total_reward_batch,
                nonzero_reward_steps,
                act_str,
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

    elapsed = time.time() - t_start
    mean_ret = float(np.mean(list(ep_returns))) if ep_returns else 0.0
    logger.info(
        "Training done. %dk steps in %.1fs (%.0f sps). Final mean return: %.2f",
        current_step // 1000,
        elapsed,
        current_step / elapsed,
        mean_ret,
    )

    # ------------------------------------------------------------------
    # Post-training evaluation: trajectory, video, diagnostic plots
    # ------------------------------------------------------------------
    _evaluate(
        config,
        bench_level,
        reward_fn,
        network,
        params,
        obs_mean,
        obs_var,
        wandb_run,
        renderer,
    )

    if wandb_run is not None:
        wandb_run.finish()


# ---------------------------------------------------------------------------
# Post-training evaluation
# ---------------------------------------------------------------------------


def _evaluate(
    config: Config,
    bench_level: BenchmarkLevel,
    reward_fn: Any,
    network: ActorCritic | VisionActorCritic,
    params: Any,
    obs_mean: jax.Array,
    obs_var: jax.Array,
    wandb_run: Any | None,
    renderer: JaxRenderer | None = None,
) -> None:
    """Run the trained policy, save trajectory, render video, plot diagnostics.

    Args:
        config: Training configuration.
        bench_level: The level that was trained on.
        reward_fn: Reward function for this level.
        network: Actor-critic network.
        params: Trained network parameters.
        obs_mean: Running observation mean for normalization.
        obs_var: Running observation variance for normalization.
        wandb_run: Live wandb run or None.
        renderer: JaxRenderer instance for vision mode, None for vector.
    """
    from dataclasses import replace as dc_replace
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from factoriax.analysis.actions import action_raster, plot_ngram_sweep
    from factoriax.analysis.state import plot_episode_rewards
    from factoriax.analysis.trajectory import (
        Trajectory,
        states_to_trajectory,
    )

    env_params = bench_level.env_params
    use_vision = config.obs_type == "vision"
    out_dir = Path("runs") / f"single_{config.level_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    jit_apply = jax.jit(network.apply)
    _mean, _var = obs_mean, obs_var

    if use_vision:

        def _normalize(obs: jax.Array) -> jax.Array:
            """Identity for vision; the CNN normalizes internally."""
            return obs
    else:

        def _normalize(obs: jax.Array) -> jax.Array:
            """Welford running-stats normalization for vector obs."""
            return jnp.clip((obs - _mean) / jnp.sqrt(_var + 1e-8), -10.0, 10.0)

    def _make_policy(seed: int):
        """Build a stochastic policy with its own PRNG."""
        state_holder = {"rng": jax.random.PRNGKey(seed + 1000)}

        def policy(obs: jax.Array) -> jax.Array:
            state_holder["rng"], key = jax.random.split(state_holder["rng"])
            logits, _ = jit_apply(params, _normalize(obs))
            return jax.random.categorical(key, logits)

        return policy

    if use_vision:
        assert renderer is not None

        def _obs_fn(
            state: EnvState,
            ep: Any,
            player_idx: int,
        ) -> jax.Array:
            return renderer.jit_render_map(state)
    else:

        def _obs_fn(
            state: EnvState,
            ep: Any,
            player_idx: int,
        ) -> jax.Array:
            return local_array(state, ep, player_idx, config.obs_radius)

    # 1. Save trajectory.
    logger.info("Running evaluation rollout...")
    env = FactoriaXEnv(
        achievement_fn=bench_level.achievement_fn
        if bench_level.achievement_fn is not None
        else None,
    )
    jit_step = jax.jit(env.step_env)
    state = build_state(bench_level.level, env_params)
    rng_eval = jax.random.PRNGKey(config.seed)
    policy = _make_policy(0)

    states_log = [state]
    actions_log: list[int] = []
    rewards_log: list[float] = []

    for _ in range(env_params.max_timesteps):
        obs = _obs_fn(state, env_params, 0)
        action = policy(obs)
        rng_eval, subkey = jax.random.split(rng_eval)
        prev_state = state
        _, state, _, done, _ = jit_step(subkey, state, action, env_params)
        reward = float(reward_fn(prev_state, state, env_params))
        actions_log.append(int(action))
        rewards_log.append(reward)
        states_log.append(state)
        if bool(done):
            break

    act_arr = np.array(
        actions_log + [0] * (len(states_log) - len(actions_log)),
        dtype=np.int32,
    )
    rew_arr = np.array(
        rewards_log + [0.0] * (len(states_log) - len(rewards_log)),
        dtype=np.float32,
    )
    traj = states_to_trajectory(states_log, actions=act_arr, rewards=rew_arr)
    obs_scheme: dict[str, object] = (
        {"type": "vision", "tile_px": config.tile_px}
        if use_vision
        else {"type": 2, "radius": config.obs_radius}
    )
    traj = dc_replace(traj, observation_scheme=obs_scheme)
    traj_path = out_dir / f"{config.level_name}_trajectory.npz"
    traj.save(str(traj_path))
    logger.info(
        "Saved trajectory: %s (%d steps, reward=%.1f)",
        traj_path,
        len(actions_log),
        sum(rewards_log),
    )

    # 2. Render video from saved states.
    logger.info("Rendering video...")
    from factoriax.renderer import render_pixels

    frames = [render_pixels(s, block_pixel_size=16) for s in states_log]
    mp4_path = out_dir / f"{config.level_name}.mp4"
    try:
        import warnings

        import imageio.v3 as iio

        mp4_path.parent.mkdir(parents=True, exist_ok=True)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message="os.fork()",
            )
            iio.imwrite(
                str(mp4_path),
                np.stack([f.astype(np.uint8) for f in frames]),
                plugin="FFMPEG",
                fps=10,
                codec="libx264",
                pixelformat="yuv420p",
            )
        logger.info("Saved video: %s (%d frames)", mp4_path, len(frames))
    except ImportError:
        logger.info("imageio[ffmpeg] not available, skipping video.")

    # 3. Diagnostic plots.
    traj = Trajectory.load(str(traj_path))
    figs: dict[str, plt.Figure] = {}

    if traj.rewards is not None:
        fig_r, _ = plot_episode_rewards(
            traj,
            episode=0,
            title=f"{config.level_name} -- evaluation rewards",
        )
        figs["rewards"] = fig_r

    fig_ar, _ = action_raster(
        traj,
        title=f"{config.level_name} -- action raster",
    )
    figs["action_raster"] = fig_ar

    fig_ng, _ = plot_ngram_sweep(
        traj,
        n_range=(2, 5),
        top_k=5,
        title=f"{config.level_name} -- top 5 n-grams (n=2..5)",
    )
    figs["ngram_sweep"] = fig_ng

    # 3b. Inventory over time — total count per item type at each step.
    item_names = {
        int(it): it.name.replace("_", " ").title()
        for it in ItemType
        if it != ItemType.EMPTY
    }
    num_steps = len(states_log)
    totals: dict[int, np.ndarray] = {
        it: np.zeros(num_steps, dtype=np.int32) for it in item_names
    }
    for t, s in enumerate(states_log):
        items_arr = np.array(s.inventory_items[0])
        counts_arr = np.array(s.inventory_counts[0])
        for it_id in item_names:
            mask = items_arr == it_id
            totals[it_id][t] = int(np.sum(counts_arr[mask]))

    # Only plot items that appear at least once.
    active = {it: vals for it, vals in totals.items() if np.any(vals > 0)}
    if active:
        fig_inv, ax_inv = plt.subplots(figsize=(10, 4))
        for it_id, vals in active.items():
            ax_inv.plot(vals, label=item_names[it_id])
        ax_inv.set_xlabel("Step")
        ax_inv.set_ylabel("Count")
        ax_inv.set_title(f"{config.level_name} -- player inventory")
        ax_inv.legend(loc="upper left", fontsize=8)
        figs["inventory"] = fig_inv

    for plot_name, fig in figs.items():
        fig_path = out_dir / f"{config.level_name}_{plot_name}.png"
        fig.savefig(fig_path, dpi=150)
        logger.info("Saved %s: %s", plot_name, fig_path)

    # 4. Upload to wandb.
    if wandb_run is not None:
        try:
            import wandb  # type: ignore[import-untyped]

            log_data: dict[str, object] = {
                "eval/total_reward": sum(rewards_log),
                "eval/episode_length": len(actions_log),
            }
            log_data[f"videos/{config.level_name}"] = wandb.Video(
                str(mp4_path),
                fps=10,
                format="mp4",
            )
            for plot_name, fig in figs.items():
                log_data[f"plots/{plot_name}"] = wandb.Image(fig)

            artifact = wandb.Artifact(
                f"trajectory-{config.level_name}",
                type="trajectory",
            )
            artifact.add_file(str(traj_path))
            wandb_run.log_artifact(artifact)
            wandb_run.log(log_data)
        except ImportError:
            pass

    for fig in figs.values():
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(
        description="Train PPO on a single basic_skills level.",
    )
    parser.add_argument(
        "level",
        choices=LEVEL_NAMES,
        help="Level to train on.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=64,
        help="Parallel environments (default: 64).",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=5_000_000,
        help="Total env steps (default: 5M).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0).",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Iterations between log lines (default: 10).",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Log to Weights and Biases.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="factoriax-basic-skills",
        help="W&B project name.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name (default: single_<level>).",
    )
    parser.add_argument(
        "--action-mask",
        type=str,
        nargs="+",
        default=None,
        metavar="ACTION",
        help=(
            "Allow only these actions (by name, e.g. FORWARD MINE). "
            "All other actions are masked out."
        ),
    )
    parser.add_argument(
        "--obs-type",
        type=str,
        choices=["vector", "vision"],
        default="vector",
        help="Observation type: 'vector' (local_array) or 'vision' "
        "(pixel render). Default: vector.",
    )
    parser.add_argument(
        "--tile-px",
        type=int,
        default=8,
        help="Tile pixel size for vision observations (default: 8).",
    )
    args = parser.parse_args()

    config = Config(
        level_name=args.level,
        num_envs=args.num_envs,
        total_steps=args.total_steps,
        seed=args.seed,
        log_interval=args.log_interval,
        action_mask=args.action_mask,
        obs_type=args.obs_type,
        tile_px=args.tile_px,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )
    train(config)


if __name__ == "__main__":
    main()
