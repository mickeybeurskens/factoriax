"""Sequenced curriculum training across basic_skills levels.

Each phase trains on one level until the mean episode return exceeds
a convergence threshold, then carries the network weights forward to
the next level. This tests whether skills transfer across tasks.

Phases are specified as positional ``level:threshold`` pairs::

    python -m baselines.basic_skills.train_sequenced mine_resources:16 craft_pallets
    python -m baselines.basic_skills.train_sequenced \\
        mine_resources:16 craft_pallets:5 fill_pallet

A phase without a threshold (no colon) trains for ``--max-steps-per-phase``
steps without early stopping. The last phase typically has no threshold.
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
class Phase:
    """One phase in the training sequence.

    Attributes:
        level_name: Name of the basic_skills level to train on.
        threshold: Convergence threshold for mean episode return.
            ``None`` means train for the full step budget.
    """

    level_name: str
    threshold: float | None = None


@dataclasses.dataclass
class Config:
    """Hyperparameters for sequenced training.

    Attributes:
        phases: Ordered list of training phases.
        hidden_dims: MLP hidden layer sizes.
        num_envs: Parallel environments.
        rollout_steps: Steps per env per update.
        max_steps_per_phase: Max env steps per phase.
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
        convergence_window: Number of recent episodes to average for
            the convergence check.
        use_wandb: Whether to log to Weights and Biases.
        wandb_project: W&B project name.
        wandb_run_name: W&B run name.
    """

    phases: list[Phase] = dataclasses.field(default_factory=list)
    hidden_dims: tuple[int, ...] = (256, 256)
    num_envs: int = 64
    rollout_steps: int = 128
    max_steps_per_phase: int = 10_000_000
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
    convergence_window: int = 200
    use_wandb: bool = False
    wandb_project: str = "factoriax-basic-skills"
    wandb_run_name: str | None = None


# ---------------------------------------------------------------------------
# Network + Transition + GAE (same as train_single)
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
# Phase training
# ---------------------------------------------------------------------------


def _train_phase(
    phase_name: str,
    bench_level: BenchmarkLevel,
    reward_fn: Callable,
    network: ActorCritic,
    params: Any,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    obs_mean: jax.Array,
    obs_var: jax.Array,
    obs_count: jax.Array,
    obs_dim: int,
    config: Config,
    max_steps: int,
    convergence_threshold: float | None,
    rng: jax.Array,
    wandb_run: Any | None,
    global_step_offset: int,
) -> tuple[Any, optax.OptState, jax.Array, jax.Array, jax.Array, jax.Array, int]:
    """Run one training phase on a single level.

    Args:
        phase_name: Label for logging (e.g. "mine", "craft").
        bench_level: Level to train on.
        reward_fn: Reward function for this level.
        network: Actor-critic network.
        params: Initial network parameters.
        opt_state: Initial optimizer state.
        optimizer: Optax optimizer.
        obs_mean: Running observation mean.
        obs_var: Running observation variance.
        obs_count: Running observation count.
        obs_dim: Observation vector length.
        config: Training configuration.
        max_steps: Maximum env steps for this phase.
        convergence_threshold: If not None, stop early when mean episode
            return over the last ``config.convergence_window`` episodes
            exceeds this value.
        rng: PRNG key.
        wandb_run: Live wandb run or None.
        global_step_offset: Steps already completed in prior phases
            (for logging and wandb step counter).

    Returns:
        Tuple of (params, opt_state, obs_mean, obs_var, obs_count,
        rng, total_steps_this_phase).
    """
    env_params = bench_level.env_params
    env = FactoriaXEnv()
    level_state = build_state(bench_level.level, env_params)

    def _broadcast(x: jax.Array) -> jax.Array:
        a = jnp.asarray(x)
        return jnp.broadcast_to(a[None], (config.num_envs,) + a.shape)

    fixed_states = jax.tree_util.tree_map(_broadcast, level_state)

    vmap_step = jax.vmap(env.step_env, in_axes=(0, 0, 0, None))
    vmap_reward = jax.vmap(reward_fn, in_axes=(0, 0, None))

    def _obs(state: EnvState) -> jax.Array:
        return local_array(state, env_params, state.selected_player, config.obs_radius)

    vmap_obs = jax.vmap(_obs)

    def _normalize(obs: jax.Array, mean: jax.Array, var: jax.Array) -> jax.Array:
        return jnp.clip((obs - mean) / jnp.sqrt(var + 1e-8), -10.0, 10.0)

    @jax.jit
    def collect(
        net_params: Any,
        env_states: EnvState,
        obs: jax.Array,
        step_rng: jax.Array,
        cur_obs_mean: jax.Array,
        cur_obs_var: jax.Array,
    ) -> tuple[Transition, EnvState, jax.Array, jax.Array, jax.Array]:
        """Collect rollout_steps transitions."""

        def rollout_step(
            carry: tuple[EnvState, jax.Array, jax.Array],
            _: None,
        ) -> tuple[tuple[EnvState, jax.Array, jax.Array], Transition]:
            states, cur_obs, step_rng_ = carry
            step_rng_, key_act, key_step = jax.random.split(step_rng_, 3)

            logits, values = network.apply(
                net_params, _normalize(cur_obs, cur_obs_mean, cur_obs_var)
            )
            actions = jax.random.categorical(key_act, logits)
            log_probs = jax.nn.log_softmax(logits)[jnp.arange(config.num_envs), actions]

            keys_step = jax.random.split(key_step, config.num_envs)
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

            return (next_states, next_obs, step_rng_), Transition(
                obs=cur_obs,
                action=actions,
                log_prob=log_probs,
                value=values,
                reward=rewards,
                done=dones,
            )

        (next_states, next_obs, step_rng), trajectories = jax.lax.scan(
            rollout_step,
            (env_states, obs, step_rng),
            None,
            length=config.rollout_steps,
        )
        _, last_values = network.apply(
            net_params,
            _normalize(next_obs, cur_obs_mean, cur_obs_var),
        )
        return trajectories, next_states, next_obs, last_values, step_rng

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
        """PPO update."""
        batch_size = flat_obs.shape[0]
        mb_size = batch_size // config.num_minibatches

        def _loss(p, mb_obs, mb_actions, mb_old_lp, mb_adv, mb_rets):
            logits, values = network.apply(
                p, _normalize(mb_obs, cur_obs_mean, cur_obs_var)
            )
            lp_all = jax.nn.log_softmax(logits)
            lp = lp_all[jnp.arange(mb_obs.shape[0]), mb_actions]
            probs = jax.nn.softmax(logits)
            entropy = -(probs * lp_all).sum(axis=-1).mean()
            adv_n = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
            ratio = jnp.exp(lp - mb_old_lp)
            pg_loss = -jnp.minimum(
                ratio * adv_n,
                jnp.clip(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps) * adv_n,
            ).mean()
            v_loss = 0.5 * ((values - mb_rets) ** 2).mean()
            total = pg_loss + config.value_coef * v_loss - config.entropy_coef * entropy
            return total, {"loss": total, "pg": pg_loss, "vf": v_loss, "ent": entropy}

        def _mb_step(carry, mb):
            p, o = carry
            (_, m), grads = jax.value_and_grad(_loss, has_aux=True)(p, *mb)
            updates, new_o = optimizer.update(grads, o, p)
            return (optax.apply_updates(p, updates), new_o), m

        def _epoch(carry, _):
            p, o, epoch_rng = carry
            epoch_rng, key_perm = jax.random.split(epoch_rng)
            perm = jax.random.permutation(key_perm, batch_size)

            def _reshape(x):
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

    # Training loop for this phase.
    steps_per_iter = config.num_envs * config.rollout_steps
    num_iters = max(1, max_steps // steps_per_iter)

    env_states = fixed_states
    obs = vmap_obs(env_states)
    ep_returns: deque[float] = deque(maxlen=config.convergence_window)
    running_return = np.zeros(config.num_envs, dtype=np.float32)
    phase_steps = 0
    t_start = time.time()

    logger.info(
        "[%s] Starting phase: %s (%d max steps, threshold=%s)",
        phase_name,
        bench_level.name,
        max_steps,
        convergence_threshold,
    )

    for it in range(num_iters):
        rng, key_collect, key_update = jax.random.split(rng, 3)

        traj, env_states, obs, last_vals, _ = collect(
            params,
            env_states,
            obs,
            key_collect,
            obs_mean,
            obs_var,
        )

        adv, ret = compute_gae(
            traj.reward,
            traj.value,
            traj.done,
            last_vals,
            config.gamma,
            config.gae_lambda,
        )

        flat_obs = traj.obs.reshape(-1, obs_dim)

        # Welford update.
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
            obs_mean,
            obs_var,
        )
        phase_steps += steps_per_iter
        global_step = global_step_offset + phase_steps

        # Track episode returns.
        rewards_np = np.array(traj.reward)
        dones_np = np.array(traj.done)
        for t in range(rewards_np.shape[0]):
            running_return += rewards_np[t]
            for idx in np.where(dones_np[t])[0]:
                ep_returns.append(float(running_return[idx]))
                running_return[idx] = 0.0

        mean_ret = float(np.mean(list(ep_returns))) if ep_returns else 0.0

        if (it + 1) % config.log_interval == 0 or it == num_iters - 1:
            elapsed = time.time() - t_start
            sps = phase_steps / elapsed
            logger.info(
                "[%s] iter=%d/%d  step=%dk  sps=%.0f  ret=%.2f  loss=%.4f  ent=%.4f",
                phase_name,
                it + 1,
                num_iters,
                global_step // 1000,
                sps,
                mean_ret,
                float(metrics["loss"]),
                float(metrics["ent"]),
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/step": float(global_step),
                        "train/sps": sps,
                        f"train/{phase_name}/mean_ep_return": mean_ret,
                        **{f"train/{k}": float(v) for k, v in metrics.items()},
                    },
                    step=global_step,
                )

        # Check convergence.
        if (
            convergence_threshold is not None
            and len(ep_returns) >= config.convergence_window
            and mean_ret >= convergence_threshold
        ):
            logger.info(
                "[%s] Converged: mean return %.2f >= %.2f after %dk steps",
                phase_name,
                mean_ret,
                convergence_threshold,
                phase_steps // 1000,
            )
            break

    elapsed = time.time() - t_start
    logger.info(
        "[%s] Phase done. %dk steps in %.1fs (%.0f sps). Final mean return: %.2f",
        phase_name,
        phase_steps // 1000,
        elapsed,
        phase_steps / elapsed if elapsed > 0 else 0,
        mean_ret,
    )

    return (params, opt_state, obs_mean, obs_var, obs_count, rng, phase_steps)


# ---------------------------------------------------------------------------
# Evaluation (reuse from train_single)
# ---------------------------------------------------------------------------


def _evaluate(
    phase_name: str,
    bench_level: BenchmarkLevel,
    reward_fn: Callable,
    network: ActorCritic,
    params: Any,
    obs_mean: jax.Array,
    obs_var: jax.Array,
    obs_radius: int,
    seed: int,
    wandb_run: Any | None,
) -> None:
    """Save trajectory, render video, and generate diagnostic plots.

    Args:
        phase_name: Label for file naming.
        bench_level: Level to evaluate on.
        reward_fn: Reward function for this level.
        network: Actor-critic network.
        params: Trained parameters.
        obs_mean: Observation mean for normalization.
        obs_var: Observation variance for normalization.
        obs_radius: Local observation radius.
        seed: Random seed.
        wandb_run: Live wandb run or None.
    """
    from dataclasses import replace as dc_replace
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from factoriax.benchmarks.single_agent_mining.analysis import (
        render_level_video,
        save_mp4,
    )

    from factoriax.analysis.actions import action_raster, plot_ngram_sweep
    from factoriax.analysis.state import plot_episode_rewards
    from factoriax.analysis.trajectory import (
        Trajectory,
        states_to_trajectory,
    )

    env_params = bench_level.env_params
    out_dir = Path("runs") / "sequenced"
    out_dir.mkdir(parents=True, exist_ok=True)

    jit_apply = jax.jit(network.apply)
    _mean, _var = obs_mean, obs_var

    def _norm(obs: jax.Array) -> jax.Array:
        return jnp.clip((obs - _mean) / jnp.sqrt(_var + 1e-8), -10.0, 10.0)

    def _make_policy(s: int) -> Callable:
        holder = {"rng": jax.random.PRNGKey(s + 1000)}

        def policy(obs: jax.Array) -> jax.Array:
            holder["rng"], key = jax.random.split(holder["rng"])
            logits, _ = jit_apply(params, _norm(obs))
            return jax.random.categorical(key, logits)

        return policy

    def _obs_fn(state, ep, player_idx):
        return local_array(state, ep, player_idx, obs_radius)

    logger.info("[%s] Running evaluation rollout...", phase_name)
    env = FactoriaXEnv()
    jit_step = jax.jit(env.step_env)
    state = build_state(bench_level.level, env_params)
    rng_eval = jax.random.PRNGKey(seed)
    policy = _make_policy(0)

    states_log = [state]
    actions_log: list[int] = []
    rewards_log: list[float] = []

    for _ in range(env_params.max_timesteps):
        obs = _obs_fn(state, env_params, 0)
        action = policy(obs)
        rng_eval, subkey = jax.random.split(rng_eval)
        prev = state
        _, state, _, done, _ = jit_step(subkey, state, action, env_params)
        rewards_log.append(float(reward_fn(prev, state, env_params)))
        actions_log.append(int(action))
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
    traj = dc_replace(
        traj,
        observation_scheme={"type": 2, "radius": obs_radius},
    )
    traj_path = out_dir / f"{phase_name}_trajectory.npz"
    traj.save(str(traj_path))
    logger.info(
        "[%s] Saved trajectory: %s (%d steps, reward=%.1f)",
        phase_name,
        traj_path,
        len(actions_log),
        sum(rewards_log),
    )

    frames = render_level_video(
        bench_level,
        _make_policy(0),
        seed=seed,
        obs_fn=_obs_fn,
    )
    mp4_path = out_dir / f"{phase_name}.mp4"
    save_mp4(frames, mp4_path)
    logger.info("[%s] Saved video: %s (%d frames)", phase_name, mp4_path, len(frames))

    traj = Trajectory.load(str(traj_path))
    figs: dict[str, plt.Figure] = {}

    if traj.rewards is not None:
        fig_r, _ = plot_episode_rewards(
            traj,
            episode=0,
            title=f"{phase_name} -- rewards",
        )
        figs["rewards"] = fig_r

    fig_ar, _ = action_raster(
        traj,
        title=f"{phase_name} -- action raster",
    )
    figs["action_raster"] = fig_ar

    fig_ng, _ = plot_ngram_sweep(
        traj,
        n_range=(2, 5),
        top_k=5,
        title=f"{phase_name} -- n-grams",
    )
    figs["ngram_sweep"] = fig_ng

    for plot_name, fig in figs.items():
        fig_path = out_dir / f"{phase_name}_{plot_name}.png"
        fig.savefig(fig_path, dpi=150)
        logger.info("[%s] Saved %s: %s", phase_name, plot_name, fig_path)

    if wandb_run is not None:
        try:
            import wandb  # type: ignore[import-untyped]

            log_data: dict[str, object] = {
                f"eval/{phase_name}/total_reward": sum(rewards_log),
                f"eval/{phase_name}/episode_length": len(actions_log),
                f"videos/{phase_name}": wandb.Video(
                    str(mp4_path),
                    fps=10,
                    format="mp4",
                ),
            }
            for plot_name, fig in figs.items():
                log_data[f"plots/{phase_name}/{plot_name}"] = wandb.Image(fig)
            wandb_run.log(log_data)
        except ImportError:
            pass

    for fig in figs.values():
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_phase(spec: str) -> Phase:
    """Parse a ``level_name:threshold`` or ``level_name`` string.

    Args:
        spec: Phase specification.

    Returns:
        Parsed ``Phase``.

    Raises:
        ValueError: If the level name is not recognized.
    """
    if ":" in spec:
        name, thresh_str = spec.split(":", 1)
        threshold: float | None = float(thresh_str)
    else:
        name = spec
        threshold = None

    if name not in _LEVEL_MAP:
        available = ", ".join(_LEVEL_MAP.keys())
        raise ValueError(f"Unknown level {name!r}. Choose from: {available}")
    return Phase(level_name=name, threshold=threshold)


def train(config: Config) -> None:
    """Run sequenced curriculum training across all configured phases.

    Each phase trains on one level until convergence (or the step
    budget), then the network weights carry forward to the next phase.

    Args:
        config: Training configuration with ordered phases.
    """
    if not config.phases:
        raise ValueError("No phases specified.")

    rng = jax.random.PRNGKey(config.seed)

    # Derive obs_dim from the first level.
    first_level = _LEVEL_MAP[config.phases[0].level_name]
    sample_state = build_state(first_level.level, first_level.env_params)
    obs_dim = int(
        local_array(sample_state, first_level.env_params, 0, config.obs_radius).shape[0]
    )

    phase_desc = " -> ".join(
        f"{p.level_name}(>={p.threshold})" if p.threshold is not None else p.level_name
        for p in config.phases
    )
    logger.info(
        "Sequenced training: %s  obs_dim=%d  num_actions=%d",
        phase_desc,
        obs_dim,
        NUM_ACTIONS,
    )

    network = ActorCritic(hidden_dims=config.hidden_dims, num_actions=NUM_ACTIONS)
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

    # Wandb.
    wandb_run = None
    if config.use_wandb:
        try:
            import wandb  # type: ignore[import-untyped]

            phase_names = "_".join(p.level_name for p in config.phases)
            run_name = config.wandb_run_name or f"seq_{phase_names}"
            wandb_run = wandb.init(
                project=config.wandb_project,
                name=run_name,
                config=dataclasses.asdict(config),
                tags=["basic_skills", "sequenced"],
            )
        except ImportError:
            logger.error("wandb not found. Install with: uv add wandb")

    global_steps = 0
    phase_steps_log: list[tuple[str, int]] = []

    for i, phase in enumerate(config.phases):
        bench_level = _LEVEL_MAP[phase.level_name]
        reward_fn = REWARD_FNS[phase.level_name]

        rng, key_phase = jax.random.split(rng)
        (
            params,
            opt_state,
            obs_mean,
            obs_var,
            obs_count,
            _,
            phase_steps,
        ) = _train_phase(
            phase_name=f"{i + 1}_{phase.level_name}",
            bench_level=bench_level,
            reward_fn=reward_fn,
            network=network,
            params=params,
            opt_state=opt_state,
            optimizer=optimizer,
            obs_mean=obs_mean,
            obs_var=obs_var,
            obs_count=obs_count,
            obs_dim=obs_dim,
            config=config,
            max_steps=config.max_steps_per_phase,
            convergence_threshold=phase.threshold,
            rng=key_phase,
            wandb_run=wandb_run,
            global_step_offset=global_steps,
        )

        _evaluate(
            f"{i + 1}_{phase.level_name}",
            bench_level,
            reward_fn,
            network,
            params,
            obs_mean,
            obs_var,
            config.obs_radius,
            config.seed,
            wandb_run,
        )

        global_steps += phase_steps
        phase_steps_log.append((phase.level_name, phase_steps))

    summary = "  ".join(f"{name}: {steps // 1000}k" for name, steps in phase_steps_log)
    logger.info("Sequenced training complete. %s", summary)

    if wandb_run is not None:
        wandb_run.finish()


def main() -> None:
    """Parse arguments and run sequenced training."""
    parser = argparse.ArgumentParser(
        description=(
            "Sequenced PPO curriculum. Specify phases as level_name:threshold pairs."
        ),
    )
    parser.add_argument(
        "phases",
        nargs="+",
        help=(
            "Training phases as level:threshold pairs. "
            "Example: mine_resources:16 craft_pallets"
        ),
    )
    parser.add_argument(
        "--max-steps-per-phase",
        type=int,
        default=10_000_000,
        help="Max steps per phase (default: 10M).",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=64,
        help="Parallel environments (default: 64).",
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
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    phases = [_parse_phase(spec) for spec in args.phases]
    config = Config(
        phases=phases,
        max_steps_per_phase=args.max_steps_per_phase,
        num_envs=args.num_envs,
        seed=args.seed,
        log_interval=args.log_interval,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )
    train(config)


if __name__ == "__main__":
    main()
