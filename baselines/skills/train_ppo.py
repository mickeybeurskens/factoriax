"""Train PPO on factoriax skill benchmarks.

Drives :class:`SkillsBenchmark` levels with sparse, time-discounted
:func:`skills_reward`. Each level uses its per-level
``blocked_actions`` mask via :class:`ActionMaskWrapper` so the policy
sees only the skill-relevant action subset. The entire
collect-GAE-update pipeline is fused into a single JIT-compiled
``train_step`` to maximise GPU throughput.

Reuses the shared PPO infrastructure from ``baselines.ppo`` and logs
to wandb under the ``factoriax_skills_benchmark`` project with tags
``skills``, ``train``, ``<level_name>``, ``<git_sha_short>``, ``ppo``.

Usage::

    python -m baselines.skills.train_ppo navigate
    python -m baselines.skills.train_ppo navigate --total-steps 100_000
    python -m baselines.skills.train_ppo all --use-wandb
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import subprocess
import time
from collections import deque
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from gymnax.environments import environment, spaces  # type: ignore[import-untyped]

from baselines.ppo.gae import Transition, compute_gae
from baselines.ppo.network import ActorCritic
from baselines.ppo.normalization import (
    RunningStats,
    init_running_stats,
    normalize_obs,
    update_running_stats,
)
from factoriax.benchmarks.core import BenchmarkLevel
from factoriax.benchmarks.skills import (
    SkillsBenchmark,
    skills_conditions,
    skills_reward,
)
from factoriax.constants import NUM_ACTIONS, Action
from factoriax.envs.action_mask_wrapper import ActionMaskWrapper
from factoriax.envs.factoriax_env import FactoriaXEnv
from factoriax.observations import NUM_PLAYER_SCALARS, NUM_SPATIAL_CHANNELS
from factoriax.state import EnvParams, EnvState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


SKILL_NAMES: tuple[str, ...] = (
    "navigate",
    "mine",
    "craft_miner",
    "place_miner",
)


class SkillsRewardEnv(environment.Environment[EnvState, EnvParams]):  # type: ignore[misc]
    """Inject :func:`skills_reward` and early-terminate on the target bit.

    Wraps :class:`FactoriaXEnv` (with ``skills_conditions`` already
    bound as the achievement function), runs the inner step, replaces
    the returned reward with the time-discounted
    :func:`skills_reward`, and sets ``done=True`` the moment the
    level's target achievement bit is unlocked. Without the early
    terminate, the policy keeps stepping a solved level for the full
    ``max_timesteps`` budget — which both wastes rollout space and
    blunts the learning signal (the agent never sees "solving ends
    the episode" and so can't optimise time-to-solve).

    Args:
        target_bit: Index into ``state.achievements_unlocked`` for
            the achievement that should terminate the episode. For
            the skills curriculum this matches the level's index in
            ``SkillsBenchmark().levels()``.
        inner: Optional pre-built inner env. A fresh
            ``FactoriaXEnv(achievement_fn=skills_conditions)`` is used
            when ``None``.
    """

    def __init__(
        self,
        target_bit: int,
        inner: FactoriaXEnv | None = None,
    ) -> None:
        super().__init__()
        self._inner = inner or FactoriaXEnv(achievement_fn=skills_conditions)
        self._target_bit = int(target_bit)

    @property
    def default_params(self) -> EnvParams:
        params: EnvParams = self._inner.default_params
        return params

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[str, Any]]:
        prev_state = state
        obs, new_state, _r, inner_done, info = self._inner.step_env(
            key, state, action, params
        )
        reward = skills_reward(prev_state, new_state, params)
        target_unlocked = new_state.achievements_unlocked[self._target_bit]
        done = inner_done | target_unlocked
        return obs, new_state, reward, done, info

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        return self._inner.reset_env(key, params)

    def get_obs(self, state: EnvState, params: EnvParams) -> jax.Array:
        obs: jax.Array = self._inner.get_obs(state, params)
        return obs

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        terminal: jax.Array = self._inner.is_terminal(state, params)
        return terminal

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        return spaces.Discrete(NUM_ACTIONS)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        obs_size = (
            NUM_SPATIAL_CHANNELS * params.map_width * params.map_height
            + NUM_PLAYER_SCALARS
        )
        return spaces.Box(0.0, 1.0, shape=(obs_size,), dtype=jnp.float32)


def _benchmark_level(skill_name: str) -> tuple[int, BenchmarkLevel]:
    """Look up the BenchmarkLevel by skill name and return ``(index, level)``.

    The index is the level's position in ``SkillsBenchmark().levels()``,
    which by the curriculum's bit-indexing contract equals the
    achievement bit the level targets.
    """
    bench = SkillsBenchmark()
    for i, bench_level in enumerate(bench.levels()):
        if bench_level.name == skill_name:
            return i, bench_level
    available = [bl.name for bl in bench.levels()]
    raise ValueError(f"Unknown skill: {skill_name!r}. Available: {available}")


def _build_env(
    target_bit: int, blocked_actions: frozenset[int]
) -> SkillsRewardEnv | Any:
    """Build the wrapped env: SkillsRewardEnv → ActionMaskWrapper if mask set."""
    inner_env = SkillsRewardEnv(target_bit=target_bit)
    if not blocked_actions:
        return inner_env
    return ActionMaskWrapper(inner_env, tuple(blocked_actions))


def _git_sha_short() -> str:
    """Best-effort git SHA short. Falls back to 'nogit' if unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


@dataclasses.dataclass
class Config:
    """Training configuration for PPO on skill benchmarks.

    Attributes:
        skill_name: Which skill to train on.
        map_size: Square map side length.
        max_timesteps: Episode length.
        hidden_dims: MLP hidden layer sizes.
        num_envs: Parallel environments.
        rollout_steps: Steps per env per update iteration.
        total_steps: Total environment steps to train for.
        learning_rate: Peak Adam learning rate.
        anneal_lr: Linearly anneal LR to zero over training.
        gamma: Discount factor.
        gae_lambda: GAE smoothing parameter.
        clip_eps: PPO clipping epsilon.
        value_coef: Value loss weight.
        entropy_coef: Entropy bonus weight.
        update_epochs: PPO epochs per collected batch.
        num_minibatches: Minibatches per epoch.
        max_grad_norm: Gradient clipping norm.
        normalize_obs: Online Welford observation normalization.
        seed: Random seed.
        log_interval: Iterations between log lines.
        use_wandb: Log to Weights and Biases.
        wandb_project: W&B project name.
        wandb_run_name: W&B run name (None = auto).
    """

    skill_name: str = "navigate"
    hidden_dims: tuple[int, ...] = (256, 256)
    num_envs: int = 1024
    rollout_steps: int = 128
    total_steps: int = 100_000
    learning_rate: float = 2.5e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    update_epochs: int = 4
    num_minibatches: int = 8
    max_grad_norm: float = 0.5
    normalize_obs: bool = True
    seed: int = 0
    log_interval: int = 1
    use_wandb: bool = False
    wandb_project: str = "factoriax_skills_benchmark"
    wandb_run_name: str | None = None


def _ppo_loss(
    network: ActorCritic,
    config: Config,
    obs_stats: RunningStats,
    params: Any,
    obs: jax.Array,
    actions: jax.Array,
    old_lp: jax.Array,
    adv: jax.Array,
    rets: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Compute the PPO clipped surrogate loss.

    Args:
        network: Actor-critic network module.
        config: Training config with PPO hyperparameters.
        obs_stats: Running observation statistics.
        params: Network parameters.
        obs: Raw observations, shape ``(B, obs_dim)``.
        actions: Chosen actions, shape ``(B,)``.
        old_lp: Behavior log-probs, shape ``(B,)``.
        adv: Advantages, shape ``(B,)``.
        rets: Returns (value targets), shape ``(B,)``.

    Returns:
        Tuple of (total_loss, metrics_dict).
    """
    norm = normalize_obs(obs_stats, obs) if config.normalize_obs else obs
    logits, values = network.apply(params, norm)
    lp_all = jax.nn.log_softmax(logits)
    lp = lp_all[jnp.arange(obs.shape[0]), actions]

    probs = jax.nn.softmax(logits)
    entropy = -(probs * lp_all).sum(axis=-1).mean()

    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    ratio = jnp.exp(lp - old_lp)
    pg_loss = -jnp.minimum(
        ratio * adv,
        jnp.clip(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps) * adv,
    ).mean()
    v_loss = 0.5 * ((values - rets) ** 2).mean()
    total = pg_loss + config.value_coef * v_loss - config.entropy_coef * entropy
    return total, {
        "loss/total": total,
        "loss/policy": pg_loss,
        "loss/value": v_loss,
        "loss/entropy": entropy,
        "misc/approx_kl": ((ratio - 1.0) - jnp.log(ratio)).mean(),
        "misc/clip_frac": (jnp.abs(ratio - 1.0) > config.clip_eps)
        .astype(jnp.float32)
        .mean(),
    }


def train(config: Config) -> dict[str, float]:
    """Train a PPO agent on one skill benchmark.

    Args:
        config: Training configuration.

    Returns:
        Dict with final metrics (mean_ep_return, sps).
    """
    level_idx, bench_level = _benchmark_level(config.skill_name)
    env_params = bench_level.env_params
    level = bench_level.level
    blocked = bench_level.blocked_actions or frozenset()
    env = _build_env(target_bit=level_idx, blocked_actions=blocked)

    # Build initial state from the level (build_state pulls in
    # pre-placed pallets/miners/etc. that ``reset_env`` would skip).
    from factoriax.levels import build_state

    initial_state = build_state(level, env_params)
    initial_obs = env.get_obs(initial_state, env_params)
    obs_dim = int(initial_obs.shape[0])

    logger.info(
        "Skill: %s  obs_dim=%d  num_actions=%d  num_envs=%d  rollout=%d  "
        "total=%dk  blocked=%d",
        config.skill_name,
        obs_dim,
        NUM_ACTIONS,
        config.num_envs,
        config.rollout_steps,
        config.total_steps // 1000,
        len(blocked),
    )

    sha_short = _git_sha_short()

    # W&B.
    wandb_run = None
    if config.use_wandb:
        try:
            import wandb

            run_name = (
                config.wandb_run_name
                or f"ppo_{config.skill_name}_{sha_short}_{config.seed}"
            )
            wandb_run = wandb.init(
                project=config.wandb_project,
                name=run_name,
                config={
                    **dataclasses.asdict(config),
                    "git_sha": sha_short,
                },
                tags=[
                    "skills",
                    "train",
                    config.skill_name,
                    sha_short,
                    "ppo",
                ],
            )
        except ImportError:
            logger.error("wandb not installed. Run: uv add wandb")
    else:
        logger.warning("wandb disabled — results saved locally only.")

    # Network.
    network = ActorCritic(
        hidden_dims=config.hidden_dims,
        num_actions=NUM_ACTIONS,
    )
    rng = jax.random.PRNGKey(config.seed)
    rng, key_init = jax.random.split(rng)
    params = network.init(key_init, jnp.zeros(obs_dim))

    # Optimizer with optional LR annealing.
    steps_per_iter = config.num_envs * config.rollout_steps
    num_iters = max(1, config.total_steps // steps_per_iter)
    if config.anneal_lr:
        total_opt_steps = num_iters * config.update_epochs * config.num_minibatches
        lr_schedule = optax.linear_schedule(
            init_value=config.learning_rate,
            end_value=0.0,
            transition_steps=total_opt_steps,
        )
    else:
        lr_schedule = config.learning_rate

    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(lr_schedule, eps=1e-5),
    )
    opt_state = optimizer.init(params)

    # Observation normalization.
    obs_stats = init_running_stats(obs_dim)

    # Broadcast initial state to num_envs copies.
    def _broadcast(x: jax.Array) -> jax.Array:
        a = jnp.asarray(x)
        return jnp.broadcast_to(a[None], (config.num_envs,) + a.shape)

    fixed_states: EnvState = jax.tree_util.tree_map(
        _broadcast,
        initial_state,
    )

    # Vmapped env operations.
    vmap_step = jax.vmap(env.step_env, in_axes=(0, 0, 0, None))
    vmap_obs = jax.vmap(env.get_obs, in_axes=(0, None))

    # ---------------------------------------------------------------
    # Fused train_step: collect + GAE + PPO update in one JIT call.
    # ---------------------------------------------------------------
    mb_size = steps_per_iter // config.num_minibatches

    @jax.jit
    def train_step(
        params: Any,
        opt_state: optax.OptState,
        obs_stats: RunningStats,
        env_states: EnvState,
        obs: jax.Array,
        rng: jax.Array,
    ) -> tuple[
        Any,
        optax.OptState,
        RunningStats,
        EnvState,
        jax.Array,
        jax.Array,
        Transition,
        dict[str, jax.Array],
    ]:
        """One training iteration: collect rollout, compute GAE, PPO update.

        Args:
            params: Network parameters.
            opt_state: Optimizer state.
            obs_stats: Running observation stats.
            env_states: Batched env states.
            obs: Current batched observations.
            rng: PRNG key.

        Returns:
            Tuple of (new_params, new_opt_state, new_obs_stats,
            new_env_states, new_obs, new_rng, trajectories, metrics).
        """
        rng, key_collect, key_update = jax.random.split(rng, 3)

        # -- Rollout collection --
        def _rollout_step(
            carry: tuple[EnvState, jax.Array, jax.Array],
            _: None,
        ) -> tuple[tuple[EnvState, jax.Array, jax.Array], Transition]:
            states, cur_obs, key = carry
            key, key_act, key_step = jax.random.split(key, 3)

            norm = (
                normalize_obs(obs_stats, cur_obs) if config.normalize_obs else cur_obs
            )
            logits, values = network.apply(params, norm)
            actions = jax.random.categorical(key_act, logits)
            log_probs = jax.nn.log_softmax(logits)[jnp.arange(config.num_envs), actions]

            keys = jax.random.split(key_step, config.num_envs)
            _, next_states, rewards, dones, _ = vmap_step(
                keys, states, actions, env_params
            )

            def _where(r: jax.Array, s: jax.Array) -> jax.Array:
                mask = dones.reshape((-1,) + (1,) * (s.ndim - 1))
                return jnp.where(mask, r, s)

            next_states = jax.tree_util.tree_map(_where, fixed_states, next_states)
            next_obs = vmap_obs(next_states, env_params)

            return (next_states, next_obs, key), Transition(
                obs=cur_obs,
                action=actions,
                log_prob=log_probs,
                value=values,
                reward=rewards,
                done=dones,
            )

        (env_states, obs, _), traj = jax.lax.scan(
            _rollout_step,
            (env_states, obs, key_collect),
            None,
            length=config.rollout_steps,
        )
        norm_last = normalize_obs(obs_stats, obs) if config.normalize_obs else obs
        _, last_vals = network.apply(params, norm_last)

        # -- GAE --
        adv, ret = compute_gae(
            traj.reward,
            traj.value,
            traj.done,
            last_vals,
            config.gamma,
            config.gae_lambda,
        )

        # -- Obs stats update --
        flat_obs = traj.obs.reshape((-1, obs_dim))
        obs_stats = (
            update_running_stats(obs_stats, flat_obs)
            if config.normalize_obs
            else obs_stats
        )

        # -- PPO update (inlined to stay in the same JIT) --
        flat_actions = traj.action.reshape(-1)
        flat_lp = traj.log_prob.reshape(-1)
        flat_adv = adv.reshape(-1)
        flat_ret = ret.reshape(-1)

        def _mb_step(
            carry: tuple[Any, optax.OptState],
            mb: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
        ) -> tuple[tuple[Any, optax.OptState], dict[str, jax.Array]]:
            p, os = carry
            (_, m), grads = jax.value_and_grad(_ppo_loss, argnums=3, has_aux=True)(
                network, config, obs_stats, p, *mb
            )
            updates, new_os = optimizer.update(grads, os, p)
            return (optax.apply_updates(p, updates), new_os), m

        def _epoch(
            carry: tuple[Any, optax.OptState, jax.Array],
            _: None,
        ) -> tuple[tuple[Any, optax.OptState, jax.Array], dict[str, jax.Array]]:
            p, os, epoch_rng = carry
            epoch_rng, key_perm = jax.random.split(epoch_rng)
            perm = jax.random.permutation(key_perm, steps_per_iter)

            def _reshape(x: jax.Array) -> jax.Array:
                return x[perm].reshape((config.num_minibatches, mb_size) + x.shape[1:])

            mbs = (
                _reshape(flat_obs),
                _reshape(flat_actions),
                _reshape(flat_lp),
                _reshape(flat_adv),
                _reshape(flat_ret),
            )
            (p, os), metrics = jax.lax.scan(_mb_step, (p, os), mbs)
            return (p, os, epoch_rng), metrics

        (params, opt_state, _), metrics = jax.lax.scan(
            _epoch,
            (params, opt_state, key_update),
            None,
            length=config.update_epochs,
        )
        metrics = jax.tree_util.tree_map(lambda x: x.mean(), metrics)

        return (
            params,
            opt_state,
            obs_stats,
            env_states,
            obs,
            rng,
            traj,
            metrics,
        )

    # ---------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------
    logger.info(
        "%d steps/iter, %d iters, %dk total steps",
        steps_per_iter,
        num_iters,
        num_iters * steps_per_iter // 1000,
    )

    env_states = fixed_states
    obs = vmap_obs(fixed_states, env_params)
    ep_returns: deque[float] = deque(maxlen=500)
    running_return = np.zeros(config.num_envs, dtype=np.float32)
    current_step = 0
    t_start = time.time()

    for it in range(num_iters):
        (params, opt_state, obs_stats, env_states, obs, rng, traj, metrics) = (
            train_step(params, opt_state, obs_stats, env_states, obs, rng)
        )
        current_step += steps_per_iter

        # Track episode returns (in numpy after the single JIT call).
        rewards_np = np.asarray(traj.reward)
        dones_np = np.asarray(traj.done)
        for t in range(rewards_np.shape[0]):
            running_return += rewards_np[t]
            for n in np.where(dones_np[t])[0]:
                ep_returns.append(float(running_return[n]))
                running_return[n] = 0.0

        if (it + 1) % config.log_interval == 0 or it == num_iters - 1:
            elapsed = time.time() - t_start
            sps = current_step / elapsed
            mean_ret = float(np.mean(list(ep_returns))) if ep_returns else 0.0

            # Top-3 action distribution.
            actions_np = np.asarray(traj.action).ravel()
            counts = np.bincount(actions_np, minlength=NUM_ACTIONS)
            top3 = np.argsort(counts)[::-1][:3]
            act_str = " ".join(
                f"{Action(a).name}={counts[a] / len(actions_np) * 100:.0f}%"
                for a in top3
            )

            logger.info(
                "iter=%d/%d  step=%dk  sps=%.0f  ret=%.2f  loss=%.4f  ent=%.4f  | %s",
                it + 1,
                num_iters,
                current_step // 1000,
                sps,
                mean_ret,
                float(metrics["loss/total"]),
                float(metrics["loss/entropy"]),
                act_str,
            )

            if wandb_run is not None:
                log_data: dict[str, float] = {
                    "train/step": float(current_step),
                    "train/sps": sps,
                    "train/mean_ep_return": mean_ret,
                }
                for k, v in metrics.items():
                    log_data[f"train/{k}"] = float(v)
                wandb_run.log(log_data, step=current_step)

    elapsed = time.time() - t_start
    mean_ret = float(np.mean(list(ep_returns))) if ep_returns else 0.0
    logger.info(
        "Done. %dk steps in %.1fs (%.0f sps). Final mean return: %.2f",
        current_step // 1000,
        elapsed,
        current_step / elapsed,
        mean_ret,
    )

    # Post-training evaluation: run the trained policy and render video.
    _evaluate(config, env, level, env_params, network, params, obs_stats, wandb_run)

    if wandb_run is not None:
        wandb_run.finish()

    return {
        "mean_ep_return": mean_ret,
        "sps": current_step / elapsed,
    }


# ---------------------------------------------------------------
# Post-training evaluation
# ---------------------------------------------------------------


def _evaluate(
    config: Config,
    env: Any,
    level: Any,
    env_params: Any,
    network: ActorCritic,
    params: Any,
    obs_stats: RunningStats,
    wandb_run: Any | None,
) -> None:
    """Run the trained policy for one episode and upload video to wandb.

    The video uses :func:`compose_frame_with_inventory` so each frame
    shows the map plus a sprite-based inventory side-panel — matches
    the agent debugger and the scripted-baseline runner output.

    Args:
        config: Training config.
        env: Skill environment instance.
        level: Level used for training.
        env_params: Environment parameters.
        network: Actor-critic network.
        params: Trained network parameters.
        obs_stats: Final observation normalization statistics.
        wandb_run: Live wandb run, or None.
    """
    from pathlib import Path

    from factoriax.analysis.video import compose_frame_with_inventory, write_video
    from factoriax.levels import build_state

    logger.info("Running evaluation rollout...")
    state = build_state(level, env_params)
    jit_step = jax.jit(env.step_env)
    jit_apply = jax.jit(network.apply)
    rng_eval = jax.random.PRNGKey(config.seed + 9999)

    states: list[EnvState] = [state]
    actions_log: list[int] = []
    rewards_log: list[float] = []
    achievement_unlocked = False

    for _ in range(env_params.max_timesteps):
        obs_raw = env.get_obs(state, env_params)
        norm = normalize_obs(obs_stats, obs_raw) if config.normalize_obs else obs_raw
        logits, _ = jit_apply(params, norm)
        rng_eval, key = jax.random.split(rng_eval)
        action = jax.random.categorical(key, logits)

        rng_eval, subkey = jax.random.split(rng_eval)
        _, state, reward, done, _ = jit_step(subkey, state, action, env_params)

        actions_log.append(int(action))
        rewards_log.append(float(reward))
        states.append(state)
        if bool(done):
            break

    # Inspect the level's target bit on the final state.
    level_idx, _ = _benchmark_level(config.skill_name)
    achievement_unlocked = bool(jnp.asarray(state.achievements_unlocked)[level_idx])
    total_reward = sum(rewards_log)
    logger.info(
        "Eval: %d steps, total reward=%.3f, solved=%s",
        len(actions_log),
        total_reward,
        achievement_unlocked,
    )

    # Render video with sprite-based inventory side-panel.
    out_dir = Path("runs") / f"skills_ppo_{config.skill_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = [compose_frame_with_inventory(s, block_pixel_size=32) for s in states]
    mp4_path = out_dir / f"{config.skill_name}.mp4"
    try:
        write_video(mp4_path, frames, fps=2)
    except ImportError:
        logger.error("imageio[ffmpeg] not available, skipping video.")
        return
    logger.info("Saved video: %s (%d frames)", mp4_path, len(frames))

    # Upload to wandb.
    if wandb_run is not None:
        try:
            import wandb

            wandb_run.log(
                {
                    "eval/total_reward": total_reward,
                    "eval/episode_length": len(actions_log),
                    "eval/solved": int(achievement_unlocked),
                    f"videos/{config.skill_name}": wandb.Video(
                        str(mp4_path), format="mp4"
                    ),
                }
            )
            logger.info("Uploaded video to wandb.")
        except ImportError:
            pass


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------


def main() -> None:
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(
        description="Train PPO on factoriax skill benchmarks.",
    )
    parser.add_argument(
        "skill",
        choices=[*SKILL_NAMES, "all"],
        help="Skill to train on, or 'all' to run each in sequence.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1024,
        help="Parallel environments (default: 1024).",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=100_000,
        help="Total env steps (default: 100k).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2.5e-4,
        help="Peak learning rate (default: 2.5e-4).",
    )
    parser.add_argument(
        "--no-anneal-lr",
        action="store_true",
        help="Disable linear LR annealing.",
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.01,
        help="Entropy bonus weight (default: 0.01).",
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
        default=1,
        help="Iterations between log lines (default: 1).",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        default=True,
        help="Log to Weights and Biases (default: enabled).",
    )
    parser.add_argument(
        "--no-wandb",
        dest="use_wandb",
        action="store_false",
        help="Disable wandb logging — useful for local-only smoke runs.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="factoriax_skills_benchmark",
        help="W&B project name.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name (default: ppo_<skill>).",
    )
    args = parser.parse_args()

    skills = list(SKILL_NAMES) if args.skill == "all" else [args.skill]

    for skill_name in skills:
        config = Config(
            skill_name=skill_name,
            num_envs=args.num_envs,
            total_steps=args.total_steps,
            learning_rate=args.learning_rate,
            anneal_lr=not args.no_anneal_lr,
            entropy_coef=args.entropy_coef,
            seed=args.seed,
            log_interval=args.log_interval,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
        )
        train(config)


if __name__ == "__main__":
    main()
