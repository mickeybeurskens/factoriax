"""Single-agent PPO training for FactoriaX.

Trains a PPO agent to maximize achievement score using JAX-native
vectorization and JIT compilation throughout. Environment rollouts
run across ``num_envs`` parallel instances via ``jax.vmap``;
trajectory collection and gradient updates are compiled with
``jax.lax.scan`` so the entire inner loop runs on-device with no
Python overhead.

Optionally integrates with Weights & Biases for metric logging and
GIF upload. Checkpoints are saved with Python pickle so they can be
resumed with ``--load-path``.

Usage::

    python baselines/single_agent_ppo.py --num-envs 64 --total-steps 10000000
    python baselines/single_agent_ppo.py --use-wandb --save-path checkpoints/run1
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
from typing import Any, NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax
from flax import struct

from factoriax.analysis import actions as ana_actions
from factoriax.analysis.recorder import RolloutRecorder
from factoriax.analysis.trajectory import Trajectory as AnalysisTrajectory
from factoriax.constants import Action
from factoriax.levels import LEVELS, Level, build_state, get_level
from factoriax.observations import global_array, local_array
from factoriax.rewards import achievement_reward, mining_reward

# Must be called before any pyplot import (all pyplot usage is inside functions).
matplotlib.use("Agg")
from factoriax.envs import FactoriaXEnv
from factoriax.renderer import render_pixels
from factoriax.state import EnvParams, EnvState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Config:
    """Hyperparameters and runtime settings for PPO training.

    Attributes:
        hidden_dims: Sizes of the shared MLP hidden layers.
        num_envs: Number of parallel training environments.
        rollout_steps: Steps collected per environment per iteration.
        total_timesteps: Total environment steps to train for.
        learning_rate: Adam learning rate.
        gamma: Discount factor.
        gae_lambda: GAE lambda smoothing parameter.
        clip_eps: PPO surrogate clipping epsilon.
        value_coef: Weight of the value loss term.
        entropy_coef: Weight of the entropy bonus.
        update_epochs: PPO update epochs per data collection batch.
        num_minibatches: Minibatches per epoch.
        max_grad_norm: Global gradient clipping norm.
        normalize_obs: Whether to apply online observation normalization.
        restrict_actions: Limit the agent to NOOP, movement, and MINE only.
        obs_type: Observation type: "global" (full flattened map) or "local"
            (windowed patch centered on the agent).
        obs_radius: Half-width of the local observation window in tiles.
            Only used when obs_type is "local".
        level_name: Name of a built-in level (key in ``factoriax.levels.LEVELS``)
            to use for all resets instead of procedural generation.  Defaults
            to ``"15x15_resources"``.  Pass ``None`` for random procedural
            generation.
        reward_type: Reward function to use during training.  ``"mining"``
            (default) gives a dense proximity-plus-bonus reward.
            ``"achievement"`` gives sparse +1 per newly unlocked achievement.
        resource_density: Per-type resource spawn probability (coal/iron/copper).
            Only applies when ``level_name`` is ``None``.
        seed: Random seed.
        save_path: Directory for checkpoints (None = disabled).
        load_path: Path to a checkpoint file to resume from.
        log_interval: Iterations between console/W&B log lines.
        use_wandb: Whether to log to Weights & Biases.
        wandb_project: W&B project name.
        wandb_run_name: W&B run name (None = auto-generated).
    """

    hidden_dims: tuple[int, ...] = (256, 256)
    num_envs: int = 64
    rollout_steps: int = 128
    total_timesteps: int = 10_000_000
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
    restrict_actions: bool = False
    obs_type: str = "global"
    obs_radius: int = 10
    level_name: str | None = "15x15_resources"
    reward_type: str = "mining"
    resource_density: float = 0.02
    seed: int = 0
    save_path: str | None = None
    load_path: str | None = None
    log_interval: int = 10
    use_wandb: bool = False
    wandb_project: str = "factoriax-ppo"
    wandb_run_name: str | None = None


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


class ActorCritic(nn.Module):
    """Shared-trunk MLP with separate policy and value heads.

    A single ``LayerNorm -> tanh`` block is applied after each hidden
    layer, which stabilizes training on the flat 1048-dim FactoriaX
    observation without requiring input normalization to warm up first.

    Attributes:
        hidden_dims: Sizes of shared hidden layers.
        num_actions: Number of discrete actions.
    """

    hidden_dims: tuple[int, ...]
    num_actions: int

    @nn.compact
    def __call__(self, obs: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Compute action logits and scalar state value.

        Args:
            obs: Observation array of shape ``(..., obs_dim)``.

        Returns:
            Tuple ``(logits, value)`` with shapes
            ``(..., num_actions)`` and ``(...,)``.
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
# Rollout storage
# ---------------------------------------------------------------------------


class Transition(NamedTuple):
    """One (s, a, log_π, V, r, done) tuple per environment step.

    Attributes:
        obs: Raw (un-normalized) observation of shape ``(obs_dim,)``.
        action: Chosen action index.
        log_prob: Log-probability under the behavior policy.
        value: Value estimate from the value head.
        reward: Received reward.
        done: Whether the episode ended at this step.
    """

    obs: jax.Array
    action: jax.Array
    log_prob: jax.Array
    value: jax.Array
    reward: jax.Array
    done: jax.Array


# ---------------------------------------------------------------------------
# Observation normalizer
# ---------------------------------------------------------------------------


@struct.dataclass
class RunningStats:
    """Welford online mean/variance tracker for observation normalization.

    Attributes:
        mean: Running mean of shape ``(obs_dim,)``.
        var: Running variance of shape ``(obs_dim,)``.
        count: Total number of samples seen.
    """

    mean: jax.Array
    var: jax.Array
    count: jax.Array


def init_running_stats(obs_dim: int) -> RunningStats:
    """Create zero-initialized running statistics.

    Args:
        obs_dim: Dimensionality of the observation vector.

    Returns:
        RunningStats with zero mean, unit variance, and zero count.
    """
    return RunningStats(
        mean=jnp.zeros(obs_dim, dtype=jnp.float32),
        var=jnp.ones(obs_dim, dtype=jnp.float32),
        count=jnp.array(0, dtype=jnp.int32),
    )


def update_running_stats(stats: RunningStats, batch: jax.Array) -> RunningStats:
    """Welford parallel batch update of running mean and variance.

    Args:
        stats: Current running statistics.
        batch: New observations of shape ``(N, obs_dim)``.

    Returns:
        Updated RunningStats.
    """
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
    """Normalize observations with running statistics, clipped to [-10, 10].

    Args:
        stats: Running observation statistics.
        obs: Raw observation array of any batch shape.

    Returns:
        Normalized observation array of the same shape.
    """
    return jnp.clip((obs - stats.mean) / jnp.sqrt(stats.var + 1e-8), -10.0, 10.0)


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
    """Compute generalized advantage estimates (GAE) and value targets.

    Scans backwards over the time dimension so the entire computation
    is a single ``jax.lax.scan`` call — no Python loops.

    Args:
        rewards: Shape ``(T, N)``.
        values: Shape ``(T, N)``.
        dones: Terminal flags, shape ``(T, N)``.
        last_value: Bootstrap value after the rollout, shape ``(N,)``.
        gamma: Discount factor.
        gae_lambda: GAE smoothing parameter.

    Returns:
        Tuple of ``(advantages, returns)`` each with shape ``(T, N)``.
    """
    not_done = 1.0 - dones.astype(jnp.float32)
    # next_values[t] = values[t+1] for t < T-1, last_value for t = T-1
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
# JIT-compiled training functions (built via closures)
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
    """Build JIT-compiled collect and update functions for the training loop.

    Both returned functions are compiled once on the first call and reused
    thereafter. All hyperparameters and architecture choices are captured
    in the closure so the public signatures stay small.

    Args:
        env: FactoriaX environment instance.
        env_params: Default environment parameters.
        network: Actor-critic Flax module.
        optimizer: Optax gradient transformation.
        config: Training configuration.
        obs_fn: Observation function ``(state, params) -> obs_array``.
            Applied to each environment's state after every step and reset
            to produce the agent's observation.  Must be JAX-native and
            JIT-compatible.
        reward_fn: Reward function ``(prev_state, new_state, params) -> scalar``.
            Applied after each step to compute the training reward signal.
            Must be JAX-native and JIT-compatible.

    Returns:
        Tuple ``(collect_fn, update_fn)``.
        ``collect_fn(params, obs_stats, env_states, obs, rng)`` collects
        one batch of rollouts and returns
        ``(trajectories, env_states, obs, last_values, rng)``.
        ``update_fn(params, opt_state, obs_stats, flat_traj, advantages,
        returns, rng)`` runs PPO epochs and returns
        ``(new_params, new_opt_state, metrics, rng)``.
        ``reset_fn(keys, params)`` returns a batched :class:`EnvState`
        of shape ``(N, ...)`` used to reinitialise terminated environments.
        The key array has shape ``(N, 2)`` and may be ignored (e.g. for
        deterministic level resets).
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
        """Replace terminated environment states with fresh reset states."""

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
        """Collect ``rollout_steps`` transitions from ``num_envs`` environments.

        Args:
            params: Current network parameters.
            obs_stats: Running observation statistics for normalization.
            env_states: Current vectorized environment state, shape ``(N, ...)``.
            obs: Current observations, shape ``(N, obs_dim)``.
            rng: PRNG key.

        Returns:
            Tuple of (trajectories, next_env_states, next_obs,
            last_values, next_rng).  trajectories fields have
            shape ``(T, N, ...)``.
        """

        def rollout_step(
            carry: tuple[EnvState, jax.Array, jax.Array],
            _: None,
        ) -> tuple[tuple[EnvState, jax.Array, jax.Array], Transition]:
            states, cur_obs, step_rng = carry
            step_rng, key_act, key_step, key_reset = jax.random.split(step_rng, 4)

            norm = (
                normalize_obs(obs_stats, cur_obs) if config.normalize_obs else cur_obs
            )
            logits, values = network.apply(params, norm)

            actions = jax.random.categorical(key_act, logits)  # (N,)
            log_probs = jax.nn.log_softmax(logits)[jnp.arange(num_envs), actions]

            keys_step = jax.random.split(key_step, num_envs)
            prev_states = states
            _, next_states, _, dones, _ = vmap_step(
                keys_step, states, actions, env_params
            )
            rewards = vmap_reward_fn(prev_states, next_states, env_params)

            keys_reset = jax.random.split(key_reset, num_envs)
            reset_states = reset_fn(keys_reset, env_params)

            next_states = _select_states(dones, next_states, reset_states)
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

        # Bootstrap value for the last observation.
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
        """Run PPO epochs on a flattened trajectory batch.

        Args:
            params: Current network parameters.
            opt_state: Current optimizer state.
            obs_stats: Running observation statistics.
            flat_obs: Observations, shape ``(T*N, obs_dim)``.
            flat_actions: Actions, shape ``(T*N,)``.
            flat_log_probs: Behavior log-probabilities, shape ``(T*N,)``.
            flat_advantages: GAE advantages, shape ``(T*N,)``.
            flat_returns: Value targets, shape ``(T*N,)``.
            rng: PRNG key.

        Returns:
            Tuple ``(new_params, new_opt_state, metrics_dict, new_rng)``.
        """
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
                jnp.clip(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps) * adv,
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
                    (jnp.abs(ratio - 1.0) > config.clip_eps).astype(jnp.float32)
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
        ) -> tuple[tuple[Any, optax.OptState], dict[str, jax.Array]]:
            p, os = carry
            (loss, m), grads = jax.value_and_grad(_loss, has_aux=True)(p, *mb)
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
                return x[perm].reshape((config.num_minibatches, mb_size) + x.shape[1:])

            mbs = (
                _reshape(flat_obs),
                _reshape(flat_actions),
                _reshape(flat_log_probs),
                _reshape(flat_advantages),
                _reshape(flat_returns),
            )
            (p, os), metrics = jax.lax.scan(_minibatch_step, (p, os), mbs)
            return (p, os, epoch_rng), metrics

        (params, opt_state, rng), metrics = jax.lax.scan(
            _epoch,
            (params, opt_state, rng),
            None,
            length=config.update_epochs,
        )
        # Average metrics over epochs and minibatches.
        metrics = jax.tree_util.tree_map(lambda x: x.mean(), metrics)
        return params, opt_state, metrics, rng

    return collect, update


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: Path,
    params: Any,
    obs_stats: RunningStats,
    step: int,
) -> None:
    """Serialize training state to disk with pickle.

    Args:
        path: Destination file path (created if absent).
        params: Network parameters pytree.
        obs_stats: Running observation statistics.
        step: Current training step count.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "params": jax.device_get(jax.tree_util.tree_map(np.array, params)),
        "obs_mean": np.array(obs_stats.mean),
        "obs_var": np.array(obs_stats.var),
        "obs_count": int(obs_stats.count),
        "step": step,
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
    logger.info("Checkpoint saved -> %s  (step %d)", path, step)


def load_checkpoint(
    path: Path,
) -> tuple[Any, RunningStats, int]:
    """Restore training state from a pickle checkpoint.

    Args:
        path: Path to the checkpoint file.

    Returns:
        Tuple of ``(params, obs_stats, step)``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    params = jax.tree_util.tree_map(jnp.array, data["params"])
    obs_stats = RunningStats(
        mean=jnp.array(data["obs_mean"]),
        var=jnp.array(data["obs_var"]),
        count=jnp.array(data["obs_count"], dtype=jnp.int32),
    )
    return params, obs_stats, int(data["step"])


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def _draw_hud(frame: np.ndarray, action_name: str, cum_return: float) -> np.ndarray:
    """Overlay a semi-transparent HUD box on a frame.

    Draws a dark translucent box in the top-left corner showing the current
    action and cumulative return.

    Args:
        frame: RGB uint8 array of shape (H, W, 3).
        action_name: Human-readable action label.
        cum_return: Cumulative episode return so far.

    Returns:
        RGB uint8 array with the HUD composited in.
    """
    from PIL import Image, ImageDraw, ImageFont

    img = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default(size=14)

    lines = [f"act   {action_name}", f"ret   {cum_return:.2f}"]
    pad = 6
    line_h = 16
    box_w = 130
    box_h = pad * 2 + line_h * len(lines)

    draw.rectangle([4, 4, 4 + box_w, 4 + box_h], fill=(0, 0, 0, 150))
    for i, line in enumerate(lines):
        draw.text(
            (4 + pad, 4 + pad + i * line_h), line, font=font, fill=(255, 255, 255, 255)
        )

    return np.array(Image.alpha_composite(img, overlay).convert("RGB"))


def render_episode(
    params: Any,
    network: ActorCritic,
    obs_stats: RunningStats,
    env: FactoriaXEnv,
    env_params: EnvParams,
    obs_fn: Callable[[EnvState, EnvParams], jax.Array],
    rng: jax.Array,
    normalize: bool = True,
    initial_state: EnvState | None = None,
) -> list[np.ndarray]:
    """Run one greedy episode and collect rendered RGB frames.

    Args:
        params: Network parameters.
        network: Actor-critic module.
        obs_stats: Running observation statistics.
        env: Environment instance.
        env_params: Environment parameters.
        obs_fn: Observation function ``(state, params) -> obs_array``.
        rng: PRNG key.
        normalize: Whether to normalize observations before the forward pass.
        initial_state: Fixed initial state to use instead of a random reset.
            Pass the level state here when training on a fixed level so the
            rendered episode matches the training distribution.

    Returns:
        Tuple of (frames, actions) where frames is a list of ``(H, W, 3)``
        uint8 numpy arrays and actions is the integer action taken each step.
    """
    jit_step = jax.jit(env.step_env, static_argnums=(3,))
    jit_apply = jax.jit(network.apply)

    logger.info("  Compiling step/apply kernels (first call)...")
    if initial_state is not None:
        state = initial_state
        obs = obs_fn(state, env_params)
    else:
        jit_reset = jax.jit(env.reset_env, static_argnums=(1,))
        rng, key_reset = jax.random.split(rng)
        obs, state = jit_reset(key_reset, env_params)
    # Warm up jit_apply and jit_step so that compilation happens now rather
    # than silently blocking the first render-loop iteration.  The warmup
    # results are thrown away; obs/state are NOT advanced so the episode
    # starts from the freshly-reset state.
    _w_norm = normalize_obs(obs_stats, obs) if normalize else obs
    _w_logits, _ = jit_apply(params, _w_norm)
    _w_action = int(jnp.argmax(_w_logits))
    rng, _key_warmup = jax.random.split(rng)
    jit_step(_key_warmup, state, _w_action, env_params)  # compile only; discard
    logger.info(
        "  Kernels ready; collecting up to %d frames...", env_params.max_timesteps
    )

    frames: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    cum_return = 0.0
    for _ in range(env_params.max_timesteps):
        norm = normalize_obs(obs_stats, obs) if normalize else obs
        logits, _ = jit_apply(params, norm)
        action = int(jnp.argmax(logits))
        actions.append(action)
        rng, key_step = jax.random.split(rng)
        obs, next_state, reward, done, _ = jit_step(key_step, state, action, env_params)
        cum_return += float(reward)
        rewards.append(float(reward))
        frame = render_pixels(state)
        frames.append(_draw_hud(frame, Action(action).name, cum_return))
        state = next_state
        if bool(done):
            frames.append(
                _draw_hud(render_pixels(state), Action(action).name, cum_return)
            )
            break

    logger.info("  Episode finished: %d frames collected.", len(frames))
    return frames, actions, rewards


def save_mp4(frames: list[np.ndarray], path: Path, fps: int = 10) -> None:
    """Write a list of frames to an MP4 video.

    Args:
        frames: List of ``(H, W, 3)`` uint8 arrays.
        path: Destination file path.
        fps: Frames per second.

    Raises:
        ValueError: If ``frames`` is empty.
    """
    if not frames:
        raise ValueError("frames list is empty; cannot write MP4")
    path.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v3 as iio

    iio.imwrite(
        str(path),
        np.stack([f.astype(np.uint8) for f in frames]),
        plugin="FFMPEG",
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
    )
    logger.info("Visualization saved -> %s", path)


def save_episode_return_plot(rewards: list[float], path: Path) -> None:
    """Plot cumulative reward over the final rendered episode and save to disk.

    Args:
        rewards: Per-step rewards from the rendered episode.
        path: Destination file path for the PNG.
    """
    import matplotlib.pyplot as plt

    cumulative = np.cumsum(rewards)
    steps = np.arange(1, len(cumulative) + 1)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(steps, cumulative, color="steelblue", linewidth=1.5)
    ax.fill_between(steps, cumulative, alpha=0.15, color="steelblue")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative reward")
    ax.set_title("Final episode — cumulative return")
    ax.set_xlim(1, max(len(cumulative), 1))
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Episode return plot saved -> %s", path)


def _save_analysis_plots(
    traj: AnalysisTrajectory | None,
    single_ep_actions: list[int],
    out_dir: Path,
    num_actions: int,
) -> None:
    """Generate and save all available analysis plots to *out_dir*.

    Multi-episode plots draw from *traj*, which is built from the last 5% of
    training iterations (up to 1 000 episodes) via
    :class:`~factoriax.analysis.recorder.RolloutRecorder`.  If no episodes
    were recorded (e.g. very short runs), multi-episode plots are skipped and
    a warning is logged.  The single-episode action raster always runs and
    comes from the greedy rendered episode.

    Args:
        traj: Multi-episode trajectory from the recorder, or *None*.
        single_ep_actions: Action sequence from the final rendered episode.
        out_dir: Directory for PNG output.
        num_actions: Number of discrete actions in the action space.
    """
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    action_labels = ana_actions.DEFAULT_ACTION_LABELS[:num_actions]

    # --- Multi-episode plots (last 5% of training, up to 1 000 episodes) -----
    if traj is not None:
        n_eps, ep_len = traj.num_episodes, traj.episode_length
        logger.info(
            "Generating multi-episode analysis plots (%d episodes, %d steps)...",
            n_eps,
            ep_len,
        )
        third = ep_len // 3

        def _save(fig: Any, name: str) -> None:
            fig.savefig(out_dir / name, dpi=120, bbox_inches="tight")
            plt.close(fig)
            logger.info("  Saved %s", name)

        # Action raster — one row per episode, color = action
        fig, _ = ana_actions.action_raster(
            traj,
            num_actions=num_actions,
            action_labels=action_labels,
            title=f"Action raster — final {n_eps} episodes",
        )
        _save(fig, "analysis_action_raster.png")

        # Transition matrix — action-to-action probabilities
        fig, _ = ana_actions.plot_transition_matrix(
            traj,
            num_actions=num_actions,
            action_labels=action_labels,
            title=f"Transition matrix — final {n_eps} episodes",
        )
        _save(fig, "analysis_transition_matrix.png")

        # Phase transitions — early / mid / late thirds side-by-side
        phases = [(0, third), (third, 2 * third), (2 * third, ep_len)]
        phase_labels = [
            f"Early (0-{third})",
            f"Mid ({third}-{2 * third})",
            f"Late ({2 * third}-{ep_len})",
        ]
        fig, _ = ana_actions.plot_phase_transitions(
            traj,
            phases,
            num_actions=num_actions,
            action_labels=action_labels,
            phase_labels=phase_labels,
        )
        _save(fig, "analysis_phase_transitions.png")

        # Top bigrams
        fig, _ = ana_actions.plot_ngrams(
            traj,
            n=2,
            action_labels=action_labels,
            title=f"Top bigrams — final {n_eps} episodes",
        )
        _save(fig, "analysis_bigrams.png")

        # Action entropy over the episode
        fig, _ = ana_actions.plot_entropy(
            traj,
            num_actions=num_actions,
            title=f"Action entropy — final {n_eps} episodes",
        )
        _save(fig, "analysis_entropy.png")

        # Run-length distributions per action
        fig, _ = ana_actions.plot_run_lengths(
            traj,
            num_actions=num_actions,
            action_labels=action_labels,
            title=f"Run-length distributions — final {n_eps} episodes",
        )
        _save(fig, "analysis_run_lengths.png")

        # Stacked area — how action proportions evolve across the episode
        fig, _ = ana_actions.plot_action_distribution(
            traj,
            num_actions=num_actions,
            action_labels=action_labels,
            title=f"Action distribution — final {n_eps} episodes",
        )
        _save(fig, "analysis_action_distribution.png")
    else:
        logger.warning(
            "No episodes were recorded; skipping multi-episode analysis plots. "
            "Run for more iterations or reduce --log-interval to populate the recorder."
        )

    # --- Single-episode plots (rendered greedy episode) ----------------------
    ep_traj = AnalysisTrajectory(actions=np.array(single_ep_actions)[np.newaxis, :])
    fig, _ = ana_actions.action_raster(
        ep_traj,
        num_actions=num_actions,
        action_labels=action_labels,
        title="Action raster — final rendered episode",
    )
    fig.savefig(out_dir / "final_episode_actions.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved final_episode_actions.png")

    logger.info("Analysis plots saved to %s", out_dir)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train(config: Config) -> None:
    """Run PPO training with the given configuration.

    Creates the environment, initializes network and optimizer, runs
    the training loop with optional W&B logging and checkpointing,
    and finishes by saving a GIF of the final policy's behaviour.

    Args:
        config: Training configuration.
    """
    rng = jax.random.PRNGKey(config.seed)

    # --- W&B setup -----------------------------------------------------------
    wandb_run = None
    if config.use_wandb:
        try:
            import wandb  # type: ignore[import-untyped]

            tags = [
                "restrict_actions" if config.restrict_actions else "full_actions",
                f"obs_{config.obs_type}",
                f"level_{config.level_name}" if config.level_name else "procedural",
                f"reward_{config.reward_type}",
            ]
            wandb_run = wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=dataclasses.asdict(config),
                tags=tags,
            )
        except ImportError:
            logger.error(
                "wandb not found. Install with: uv add wandb  (continuing without W&B)"
            )

    # --- Environment and network setup ---------------------------------------
    env = FactoriaXEnv()
    env_params = env.default_params.replace(
        num_players=1,
        max_timesteps=200,
        coal_probability=config.resource_density,
        iron_probability=config.resource_density,
        copper_probability=config.resource_density,
    )

    # --- Level or procedural reset -------------------------------------------
    _level: Level | None = None
    if config.level_name is not None:
        _level = get_level(config.level_name)
        env_params = env_params.replace(
            map_width=_level.map_width,
            map_height=_level.map_height,
        )

    # --- Build observation function ------------------------------------------
    _obs_radius = config.obs_radius
    _obs_type = config.obs_type

    def _obs_fn(state: EnvState, params: EnvParams) -> jax.Array:
        """Return observation for the selected player under the configured type."""
        if _obs_type == "local":
            return local_array(state, params, state.selected_player, _obs_radius)
        return global_array(state, params, state.selected_player)

    if _level is not None:
        _level_state = build_state(_level, env_params)
        dummy_state = _level_state

        def _reset_fn(keys: jax.Array, params: EnvParams) -> EnvState:
            """Return the fixed level state broadcast across all environments."""

            def _broadcast(x: jax.Array) -> jax.Array:
                a = jnp.asarray(x)
                return jnp.broadcast_to(a[None], (config.num_envs,) + a.shape)

            return jax.tree_util.tree_map(_broadcast, _level_state)
    else:
        rng, key_dummy = jax.random.split(rng)
        _, dummy_state = env.reset_env(key_dummy, env_params)
        _vmap_reset = jax.vmap(env.reset_env, in_axes=(0, None))

        def _reset_fn(keys: jax.Array, params: EnvParams) -> EnvState:
            """Return fresh procedurally generated states for all environments."""
            _, states = _vmap_reset(keys, params)
            return states

    # --- Build reward function ------------------------------------------------
    _reward_fn = mining_reward if config.reward_type == "mining" else achievement_reward

    obs_dim = int(_obs_fn(dummy_state, env_params).shape[0])
    # NOOP=0, LEFT=1, RIGHT=2, UP=3, DOWN=4, MINE=5 are the first 6 actions.
    num_actions = 6 if config.restrict_actions else int(env.action_space(env_params).n)

    network = ActorCritic(hidden_dims=config.hidden_dims, num_actions=num_actions)
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.learning_rate),
    )

    rng, key_init = jax.random.split(rng)
    params = network.init(key_init, jnp.zeros(obs_dim))
    opt_state = optimizer.init(params)
    obs_stats = init_running_stats(obs_dim)
    start_step = 0

    # --- Optional checkpoint restore -----------------------------------------
    if config.load_path is not None:
        load_path = Path(config.load_path)
        params, obs_stats, start_step = load_checkpoint(load_path)
        opt_state = optimizer.init(params)
        logger.info("Resumed from %s at step %d", load_path, start_step)

    # --- Build JIT'd training functions --------------------------------------
    collect_fn, update_fn = make_train_fns(
        env, env_params, network, optimizer, config, _obs_fn, _reset_fn, _reward_fn
    )

    # --- Compute loop counts -------------------------------------------------
    steps_per_iter = config.num_envs * config.rollout_steps
    remaining = config.total_timesteps - start_step
    total_iters = max(1, remaining // steps_per_iter)
    current_step = start_step

    # --- Analysis recorder (last 5% of training, up to 1 000 episodes) -------
    # Multi-episode analysis plots (transition matrices, entropy, etc.) are
    # generated from this window so they reflect the converged policy rather
    # than early exploration.
    record_start_iter = total_iters - max(1, total_iters // 20)
    recorder = RolloutRecorder(max_episodes=1000)

    logger.info(
        "Starting training: %d iters x %d steps/iter = %s total steps",
        total_iters,
        steps_per_iter,
        f"{total_iters * steps_per_iter:,}",
    )
    obs_label = (
        f"local(r={config.obs_radius})" if config.obs_type == "local" else "global"
    )
    level_label = config.level_name if config.level_name else "procedural"
    logger.info(
        "  level=%s  obs=%s  obs_dim=%d  reward=%s  num_actions=%d"
        "  num_envs=%d  rollout_steps=%d  max_timesteps=%d",
        level_label,
        obs_label,
        obs_dim,
        config.reward_type,
        num_actions,
        config.num_envs,
        config.rollout_steps,
        env_params.max_timesteps,
    )
    if config.save_path is not None:
        logger.info("Checkpoints will be saved to %s", Path(config.save_path))

    # --- Initialize vectorized environments ----------------------------------
    logger.info(
        "Compiling and initializing %d environments (first JIT — may take a minute)...",
        config.num_envs,
    )
    rng, key_envs = jax.random.split(rng)
    keys_envs = jax.random.split(key_envs, config.num_envs)
    env_states = _reset_fn(keys_envs, env_params)
    obs = jax.vmap(_obs_fn, in_axes=(0, None))(env_states, env_params)
    logger.info("Environments ready.")

    t_start = time.time()
    running_ep_return = np.zeros(config.num_envs, dtype=np.float32)
    running_ep_length = np.zeros(config.num_envs, dtype=np.int32)
    completed_ep_returns: deque[float] = deque(maxlen=1000)
    completed_ep_lengths: deque[int] = deque(maxlen=1000)

    for it in range(total_iters):
        rng, key_collect, key_update = jax.random.split(rng, 3)

        # Collect rollout ---------------------------------------------------
        trajectories, env_states, obs, last_values, _ = collect_fn(
            params, obs_stats, env_states, obs, key_collect
        )

        # Feed recorder during the last 5% of training iterations ----------
        if it >= record_start_iter:
            recorder.record(trajectories)

        # Update obs normalizer from raw collected observations -------------
        flat_obs = trajectories.obs.reshape(-1, obs_dim)
        obs_stats = update_running_stats(obs_stats, flat_obs)

        # GAE ---------------------------------------------------------------
        advantages, returns = compute_gae(
            trajectories.reward,
            trajectories.value,
            trajectories.done,
            last_values,
            config.gamma,
            config.gae_lambda,
        )

        # PPO update --------------------------------------------------------
        flat_actions = trajectories.action.reshape(-1)
        flat_log_probs = trajectories.log_prob.reshape(-1)
        flat_adv = advantages.reshape(-1)
        flat_ret = returns.reshape(-1)

        params, opt_state, metrics, _ = update_fn(
            params,
            opt_state,
            obs_stats,
            flat_obs,
            flat_actions,
            flat_log_probs,
            flat_adv,
            flat_ret,
            key_update,
        )

        current_step += steps_per_iter

        # Track completed episodes via done signals -------------------------
        rewards_np = np.array(trajectories.reward)  # (T, N)
        dones_np = np.array(trajectories.done)  # (T, N)
        for t in range(rewards_np.shape[0]):
            running_ep_return += rewards_np[t]
            running_ep_length += 1
            done_envs = np.where(dones_np[t])[0]
            for n in done_envs:
                completed_ep_returns.append(float(running_ep_return[n]))
                completed_ep_lengths.append(int(running_ep_length[n]))
                running_ep_return[n] = 0.0
                running_ep_length[n] = 0

        # Logging -----------------------------------------------------------
        if (it + 1) % config.log_interval == 0 or it == total_iters - 1:
            elapsed = time.time() - t_start
            sps = current_step / elapsed
            if completed_ep_returns:
                mean_ep_reward = float(
                    np.mean(
                        [
                            r / l
                            for r, l in zip(completed_ep_returns, completed_ep_lengths)
                        ]
                    )
                )
                max_ep_return = float(np.max(completed_ep_returns))
            else:
                mean_ep_reward = 0.0
                max_ep_return = 0.0
            log_data: dict[str, float] = {
                "train/step": float(current_step),
                "train/sps": sps,
                "train/mean_ep_reward": mean_ep_reward,
                "train/max_ep_return": max_ep_return,
                **{k: float(v) for k, v in metrics.items()},
            }
            logger.info(
                "step=%d  sps=%.0f  mean_rew=%.4f  max_ret=%.3f  "
                "loss=%.4f  entropy=%.4f  kl=%.5f",
                current_step,
                sps,
                mean_ep_reward,
                max_ep_return,
                float(metrics["loss/total"]),
                float(metrics["loss/entropy"]),
                float(metrics["misc/approx_kl"]),
            )
            if wandb_run is not None:
                wandb_run.log(log_data, step=current_step)

        # Checkpointing -----------------------------------------------------
        if config.save_path is not None and (it + 1) % config.log_interval == 0:
            ckpt_path = Path(config.save_path) / f"step_{current_step}.pkl"
            save_checkpoint(ckpt_path, params, obs_stats, current_step)

    # --- Final checkpoint ----------------------------------------------------
    if config.save_path is not None:
        final_path = Path(config.save_path) / "final.pkl"
        save_checkpoint(final_path, params, obs_stats, current_step)
        logger.info("Final checkpoint saved. To resume: --load-path %s", final_path)

    # --- Visualization of final policy ---------------------------------------
    logger.info("Rendering final-policy episode...")
    rng, key_vis = jax.random.split(rng)
    frames, actions, rewards = render_episode(
        params,
        network,
        obs_stats,
        env,
        env_params,
        _obs_fn,
        key_vis,
        normalize=config.normalize_obs,
        initial_state=_level_state if _level is not None else None,
    )
    out_dir = Path(config.save_path) if config.save_path is not None else Path(".")
    mp4_path = out_dir / "final_episode.mp4"
    save_mp4(frames, mp4_path)
    save_episode_return_plot(rewards, out_dir / "final_episode_return.png")

    analysis_traj: AnalysisTrajectory | None = None
    if recorder.num_recorded_steps > 0:
        analysis_traj = recorder.finish()
        logger.info(
            "Recorder captured %d complete episodes from the last 5%% of training.",
            analysis_traj.num_episodes,
        )
    _save_analysis_plots(analysis_traj, actions, out_dir, num_actions)

    if wandb_run is not None:
        try:
            import wandb  # type: ignore[import-untyped]

            def _img(name: str) -> wandb.Image | None:
                p = out_dir / name
                return wandb.Image(str(p)) if p.exists() else None

            upload: dict[str, Any] = {
                "eval/final_episode": wandb.Video(str(mp4_path), fps=10),
                "eval/episode_return": _img("final_episode_return.png"),
                "eval/action_timeline": _img("final_episode_actions.png"),
                "eval/action_raster": _img("analysis_action_raster.png"),
                "eval/transition_matrix": _img("analysis_transition_matrix.png"),
                "eval/phase_transitions": _img("analysis_phase_transitions.png"),
                "eval/bigrams": _img("analysis_bigrams.png"),
                "eval/entropy": _img("analysis_entropy.png"),
                "eval/run_lengths": _img("analysis_run_lengths.png"),
                "eval/action_distribution": _img("analysis_action_distribution.png"),
            }
            wandb_run.log({k: v for k, v in upload.items() if v is not None})
        except Exception as exc:  # noqa: BLE001
            logger.error("W&B upload failed: %s", exc)
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
        description="Single-agent PPO for FactoriaX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--rollout-steps", type=int, default=128)
    p.add_argument("--total-steps", type=int, default=10_000_000)
    p.add_argument("--lr", type=float, default=2.5e-4, dest="learning_rate")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--num-minibatches", type=int, default=8)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument(
        "--no-normalize-obs",
        action="store_false",
        dest="normalize_obs",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-path", type=str, default=None)
    p.add_argument("--load-path", type=str, default=None)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--restrict-actions", action="store_true")
    p.add_argument(
        "--obs-type",
        type=str,
        default="global",
        choices=["global", "local"],
        help="Observation type: 'global' (full flattened map) or 'local' (windowed patch).",
    )
    p.add_argument(
        "--obs-radius",
        type=int,
        default=10,
        help="Half-width of the local observation window in tiles (local obs only).",
    )
    p.add_argument(
        "--level",
        type=str,
        default="15x15_resources",
        dest="level_name",
        choices=list(LEVELS),
        help="Built-in level name for fixed resets. Omit for procedural generation.",
    )
    p.add_argument(
        "--reward-type",
        type=str,
        default="mining",
        choices=["achievement", "mining"],
        help=(
            "Reward function: 'achievement' (sparse +1 per unlocked achievement) "
            "or 'mining' (dense proximity + ore-extraction bonus)."
        ),
    )
    p.add_argument("--resource-density", type=float, default=0.02)
    p.add_argument("--use-wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="factoriax-ppo")
    p.add_argument("--wandb-run-name", type=str, default=None)

    args = p.parse_args()
    return Config(
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        total_timesteps=args.total_steps,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        update_epochs=args.update_epochs,
        num_minibatches=args.num_minibatches,
        max_grad_norm=args.max_grad_norm,
        normalize_obs=args.normalize_obs,
        restrict_actions=args.restrict_actions,
        obs_type=args.obs_type,
        obs_radius=args.obs_radius,
        level_name=args.level_name,
        reward_type=args.reward_type,
        resource_density=args.resource_density,
        seed=args.seed,
        save_path=args.save_path,
        load_path=args.load_path,
        log_interval=args.log_interval,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )


if __name__ == "__main__":
    train(_parse_args())
