"""Train PPO on the rocket achievement benchmark.

Builds :class:`~factoriax.envs.FactoriaXEnv` with the rocket benchmark's
:func:`rocket_conditions` bound as ``achievement_fn``, then uses
:func:`rocket_reward` as the training signal — Craftax-style sparse
reward of +weight on each newly-unlocked achievement.

Reuses the shared PPO infrastructure from ``baselines.ppo``. The
collect/GAE/update pipeline is fused into one JIT-compiled step to keep
GPU throughput high.

Usage::

    python -m baselines.rocket.train_ppo
    python -m baselines.rocket.train_ppo --total-steps 10_000_000 --use-wandb
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import time
from collections import deque
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from baselines.ppo.gae import Transition, compute_gae
from baselines.ppo.network import ActorCritic
from baselines.ppo.normalization import (
    RunningStats,
    init_running_stats,
    normalize_obs,
    update_running_stats,
)
from factoriax.analysis.eval import EvalRollout, generate_eval_plots
from factoriax.analysis.video import compose_frame_with_inventory, write_video
from factoriax.benchmarks.rocket import (
    MAX_ROCKET_SCORE,
    NUM_ROCKET_ACHIEVEMENTS,
    ROCKET_ACHIEVEMENT_INFO,
    ROCKET_ACHIEVEMENT_WEIGHTS,
    ROCKET_BLOCKED_ACTIONS,
    build_rocket_level,
    rocket_conditions,
    rocket_reward,
)
from factoriax.constants import MAX_ACHIEVEMENTS, NUM_ACTIONS, Action
from factoriax.envs import FactoriaXEnv
from factoriax.envs.action_mask_wrapper import ActionMaskWrapper
from factoriax.envs.local_observation_wrapper import LocalObservationWrapper
from factoriax.levels import build_state
from factoriax.state import EnvParams, EnvState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Config:
    """Training configuration for PPO on the rocket benchmark.

    Defaults target a pilot run that finishes in a few minutes on a
    single GPU. Scale ``total_steps`` up for a real baseline.
    """

    map_size: int = 32
    # 8000 matches the scripted-agent benchmark. Naive needs 5907 ticks
    # to reach 38/38; 2000 (the Apr-20 default) caps the policy well
    # short of the rocket chain.
    max_timesteps: int = 8000
    # Half-width of the local obs window. ``radius=7`` → 15x15 tiles.
    # The full-map global obs scales quadratically with map size and
    # dominates the first FC layer; a 15x15 window covers the agent's
    # immediate factory footprint (furnace + assembler + pallet strips)
    # without paying the 32x32 cost.
    obs_radius: int = 7
    hidden_dims: tuple[int, ...] = (256, 256)
    num_envs: int = 512
    rollout_steps: int = 128
    total_steps: int = 3_000_000
    learning_rate: float = 2.5e-4
    anneal_lr: bool = True
    gamma: float = 0.995
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
    wandb_project: str = "factoriax_rocket"
    wandb_run_name: str | None = None
    # Post-training artifacts. out_dir collects final_model.msgpack and
    # final_rollout.mp4; when use_wandb is on both are also uploaded as
    # wandb artifacts and the video is embedded in the run page.
    out_dir: str | None = None
    save_final_model: bool = True
    save_final_video: bool = True
    video_fps: int = 30


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
    """Clipped-surrogate PPO loss with entropy bonus."""
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


def _make_env_and_state(
    config: Config,
) -> tuple[ActionMaskWrapper, EnvState, EnvParams]:
    """Build the wrapped env, initial EnvState, and EnvParams.

    Action mask matches the scripted benchmark: ``CRAFT_*`` actions are
    blocked so the policy has to produce intermediates through machines
    (furnaces/assemblers) rather than handcrafting. Without the mask the
    PPO task would be strictly easier than what the scripted agents
    solve, making the numbers incomparable.

    Observations are a local ``(2r+1) x (2r+1)`` window centered on the
    player (``config.obs_radius``). The global full-map obs scales
    quadratically with ``map_size`` and dominates the first FC layer of
    the policy; switching to a local window keeps throughput flat as the
    map grows.
    """
    base_env = FactoriaXEnv(achievement_fn=rocket_conditions)
    env = ActionMaskWrapper(
        LocalObservationWrapper(base_env, radius=config.obs_radius),
        ROCKET_BLOCKED_ACTIONS,
    )
    env_params = EnvParams(
        map_width=config.map_size,
        map_height=config.map_size,
        num_players=1,
        max_timesteps=config.max_timesteps,
    )
    level = build_rocket_level()
    state0 = build_state(level, env_params)
    return env, state0, env_params


# ---------------------------------------------------------------------------
# Post-training artifacts (final model + rollout video + wandb upload)
# ---------------------------------------------------------------------------


def _resolve_out_dir(config: Config) -> Any:
    """Return the directory where final artifacts are written.

    Uses ``config.out_dir`` when set; otherwise derives
    ``runs/rocket_ppo/{wandb_run_name or "default"}``.
    """
    from pathlib import Path  # noqa: PLC0415

    if config.out_dir is not None:
        return Path(config.out_dir)
    name = config.wandb_run_name or "default"
    return Path("runs") / "rocket_ppo" / name


def _save_final_model(
    path: Any,
    params: Any,
    obs_stats: RunningStats,
    config: Config,
) -> None:
    """Serialize params + obs_stats to msgpack and config to sibling JSON.

    The msgpack path stays JAX-friendly (no tuples etc.); the config is
    written alongside so an external script can reload both without
    reconstructing a ``Config`` object by hand.
    """
    from pathlib import Path  # noqa: PLC0415

    import orjson  # noqa: PLC0415
    from flax import serialization  # type: ignore[import-untyped]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "params": params,
        "obs_stats": {
            "count": np.asarray(obs_stats.count),
            "mean": np.asarray(obs_stats.mean),
            "var": np.asarray(obs_stats.var),
        },
    }
    path.write_bytes(serialization.msgpack_serialize(payload))
    config_path = path.with_suffix(".config.json")
    config_path.write_bytes(
        orjson.dumps(
            dataclasses.asdict(config),
            option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY,
        ),
    )


def _render_eval_episode(
    config: Config,
    env: ActionMaskWrapper,
    env_params: EnvParams,
    initial_state: EnvState,
    network: ActorCritic,
    params: Any,
    obs_stats: RunningStats,
) -> EvalRollout:
    """Run one deterministic eval episode and return collected artifacts.

    Uses a Python-side loop (not a JIT-compiled scan) so we can snapshot
    the underlying ``EnvState`` at every tick for rendering and for
    trajectory-level analysis. This is fine because the eval is a
    one-shot ~2000-step rollout, not a hot path.

    Each frame is composed by
    :func:`factoriax.analysis.video.compose_frame_with_inventory` —
    the map render horizontally concatenated with an inventory
    side-panel rendered from the same ``EnvState``.
    """
    jit_step = jax.jit(env.step_env)
    jit_apply = jax.jit(network.apply)
    rng = jax.random.PRNGKey(config.seed + 4242)
    state = initial_state

    frames: list[np.ndarray] = [compose_frame_with_inventory(state)]
    env_states: list[Any] = [state]
    ach_per_step: list[np.ndarray] = [np.asarray(state.achievements_unlocked)]
    actions_log: list[int] = []

    for _ in range(env_params.max_timesteps):
        obs = env.get_obs(state, env_params)
        norm = normalize_obs(obs_stats, obs) if config.normalize_obs else obs
        logits, _ = jit_apply(params, norm)
        rng, k_act = jax.random.split(rng)
        action = jax.random.categorical(k_act, logits)
        rng, k_step = jax.random.split(rng)
        _, state, _, done, _ = jit_step(k_step, state, action, env_params)
        actions_log.append(int(action))
        frames.append(compose_frame_with_inventory(state))
        env_states.append(state)
        ach_per_step.append(np.asarray(state.achievements_unlocked))
        if bool(done):
            break

    return EvalRollout(
        frames=frames,
        actions=np.asarray(actions_log, dtype=np.int32),
        env_states=env_states,
        ach_per_step=np.stack(ach_per_step, axis=0),
    )


def _finalize_artifacts(
    config: Config,
    env: ActionMaskWrapper,
    env_params: EnvParams,
    initial_state: EnvState,
    network: ActorCritic,
    params: Any,
    obs_stats: RunningStats,
    wandb_run: Any | None,
    current_step: int,
) -> None:
    """Write final model + rollout video + analysis plots.

    Runs strictly after training completes; never touches the hot loop.
    Uploads the model, video, and PNGs as wandb Artifacts and inline
    images when wandb is active.
    """
    out_dir = _resolve_out_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = None
    if config.save_final_model:
        model_path = out_dir / "final_model.msgpack"
        _save_final_model(model_path, params, obs_stats, config)
        logger.info("Saved final model: %s", model_path)

    video_path = None
    plot_paths: dict[str, Any] = {}
    ach_mask = None

    if config.save_final_video:
        logger.info("Rendering final eval rollout...")
        rollout = _render_eval_episode(
            config,
            env,
            env_params,
            initial_state,
            network,
            params,
            obs_stats,
        )
        ach_mask = rollout.final_ach_mask

        video_path = out_dir / "final_rollout.mp4"
        try:
            assert rollout.frames is not None  # PPO eval always buffers
            write_video(video_path, rollout.frames, config.video_fps)
            logger.info(
                "Saved final rollout video: %s (%d frames, %d unique actions)",
                video_path,
                len(rollout.frames),
                int(np.unique(rollout.actions).size),
            )
        except ImportError:
            logger.error("imageio[ffmpeg] missing; final video skipped.")
            video_path = None

        plot_paths = generate_eval_plots(
            rollout,
            out_dir,
            achievement_labels=[info.id for info in ROCKET_ACHIEVEMENT_INFO],
            num_achievements=NUM_ROCKET_ACHIEVEMENTS,
            title_prefix="Final rollout",
        )
        for name, path in plot_paths.items():
            logger.info("Saved %s plot: %s", name, path)

    if wandb_run is None:
        return

    import wandb  # noqa: PLC0415  # type: ignore[import-untyped]

    run_id = getattr(wandb_run, "id", "run")

    if model_path is not None:
        artifact = wandb.Artifact(f"rocket-ppo-model-{run_id}", type="model")
        artifact.add_file(str(model_path))
        wandb_run.log_artifact(artifact)
        logger.info("Uploaded model artifact to wandb.")

    if video_path is not None:
        vid_artifact = wandb.Artifact(f"rocket-ppo-video-{run_id}", type="video")
        vid_artifact.add_file(str(video_path))
        wandb_run.log_artifact(vid_artifact)
        wandb_run.log(
            {
                "final/rollout": wandb.Video(
                    str(video_path),
                    fps=config.video_fps,
                    format="mp4",
                ),
            },
            step=current_step,
        )
        logger.info("Uploaded video artifact + inline video to wandb.")

    if plot_paths:
        plots_artifact = wandb.Artifact(
            f"rocket-ppo-plots-{run_id}",
            type="analysis",
        )
        inline: dict[str, Any] = {}
        for name, path in plot_paths.items():
            plots_artifact.add_file(str(path))
            inline[f"final/plots/{name}"] = wandb.Image(str(path))
        wandb_run.log_artifact(plots_artifact)
        wandb_run.log(inline, step=current_step)
        logger.info("Uploaded %d analysis plot(s) to wandb.", len(plot_paths))

    if ach_mask is not None:
        unlocked_ids = [
            ROCKET_ACHIEVEMENT_INFO[i].id
            for i in range(NUM_ROCKET_ACHIEVEMENTS)
            if bool(ach_mask[i])
        ]
        score = float(
            np.sum(
                np.asarray(ROCKET_ACHIEVEMENT_WEIGHTS)[:NUM_ROCKET_ACHIEVEMENTS]
                * ach_mask[:NUM_ROCKET_ACHIEVEMENTS]
            )
        )
        unlocked_str = ", ".join(unlocked_ids) if unlocked_ids else "(none)"
        ach_count = int(ach_mask[:NUM_ROCKET_ACHIEVEMENTS].sum())
        wandb_run.log(
            {
                "final/achievements_count": ach_count,
                "final/score": score,
                "final/unlocked": unlocked_str,
            },
            step=current_step,
        )


def train(config: Config) -> dict[str, float]:
    """Train PPO against the rocket benchmark and return final metrics."""
    env, initial_state, env_params = _make_env_and_state(config)
    initial_obs = env.get_obs(initial_state, env_params)
    obs_dim = int(initial_obs.shape[0])

    logger.info(
        "Rocket PPO — obs_dim=%d  actions=%d  envs=%d  rollout=%d  total=%dk",
        obs_dim,
        NUM_ACTIONS,
        config.num_envs,
        config.rollout_steps,
        config.total_steps // 1000,
    )

    wandb_run = None
    if config.use_wandb:
        try:
            import wandb  # type: ignore[import-untyped]

            run_name = config.wandb_run_name or "ppo_rocket"
            wandb_run = wandb.init(
                project=config.wandb_project,
                name=run_name,
                config=dataclasses.asdict(config),
                tags=[
                    "rocket",
                    "ppo",
                    "achievement",
                    "local_observation",
                    f"obs_radius_{config.obs_radius}",
                ],
            )
        except ImportError:
            logger.error("wandb not installed. Run: uv add wandb")

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

    obs_stats = init_running_stats(obs_dim)

    # Broadcast the initial EnvState to (num_envs, ...).
    def _broadcast(x: jax.Array) -> jax.Array:
        a = jnp.asarray(x)
        return jnp.broadcast_to(a[None], (config.num_envs,) + a.shape)

    fixed_states: EnvState = jax.tree_util.tree_map(_broadcast, initial_state)

    vmap_step = jax.vmap(env.step_env, in_axes=(0, 0, 0, None))
    vmap_obs = jax.vmap(env.get_obs, in_axes=(0, None))
    vmap_reward = jax.vmap(rocket_reward, in_axes=(0, 0, None))

    mb_size = steps_per_iter // config.num_minibatches

    @jax.jit
    def train_step(
        params: Any,
        opt_state: optax.OptState,
        obs_stats: RunningStats,
        env_states: EnvState,
        obs: jax.Array,
        rng: jax.Array,
    ):
        """One fused training iteration: rollout + GAE + PPO update."""
        rng, key_collect, key_update = jax.random.split(rng, 3)

        def _rollout_step(carry, _):
            states, cur_obs, key = carry
            key, key_act, key_step = jax.random.split(key, 3)

            norm = (
                normalize_obs(obs_stats, cur_obs) if config.normalize_obs else cur_obs
            )
            logits, values = network.apply(params, norm)
            actions = jax.random.categorical(key_act, logits)
            log_probs = jax.nn.log_softmax(logits)[jnp.arange(config.num_envs), actions]

            keys = jax.random.split(key_step, config.num_envs)
            _, next_states, _env_rewards, dones, _ = vmap_step(
                keys, states, actions, env_params
            )

            # Reward = newly-unlocked weighted achievements.
            rewards = vmap_reward(states, next_states, env_params)

            # Reset on done: swap back the fixed initial EnvState.
            def _where(r, s):
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

        adv, ret = compute_gae(
            traj.reward,
            traj.value,
            traj.done,
            last_vals,
            config.gamma,
            config.gae_lambda,
        )

        flat_obs = traj.obs.reshape((-1, obs_dim))
        obs_stats = (
            update_running_stats(obs_stats, flat_obs)
            if config.normalize_obs
            else obs_stats
        )

        flat_actions = traj.action.reshape(-1)
        flat_lp = traj.log_prob.reshape(-1)
        flat_adv = adv.reshape(-1)
        flat_ret = ret.reshape(-1)

        def _mb_step(carry, mb):
            p, os = carry
            (_, m), grads = jax.value_and_grad(_ppo_loss, argnums=3, has_aux=True)(
                network, config, obs_stats, p, *mb
            )
            updates, new_os = optimizer.update(grads, os, p)
            return (optax.apply_updates(p, updates), new_os), m

        def _epoch(carry, _):
            p, os, epoch_rng = carry
            epoch_rng, key_perm = jax.random.split(epoch_rng)
            perm = jax.random.permutation(key_perm, steps_per_iter)

            def _reshape(x):
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

        # Pre-reset achievement masks (before _where reset zeroes them) for logging.
        # traj is already stacked (T, N, ...). Rebuild by tracking per-step masks.
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

    # Training loop.
    logger.info(
        "%d steps/iter  %d iters  %dk total",
        steps_per_iter,
        num_iters,
        num_iters * steps_per_iter // 1000,
    )

    env_states = fixed_states
    obs = vmap_obs(fixed_states, env_params)
    ep_returns: deque[float] = deque(maxlen=500)
    ep_ach_counts: deque[int] = deque(maxlen=500)
    running_return = np.zeros(config.num_envs, dtype=np.float32)
    running_peak_mask = np.zeros((config.num_envs, MAX_ACHIEVEMENTS), dtype=bool)
    best_ach_count_ever = 0
    per_ach_unlock_total = np.zeros(NUM_ROCKET_ACHIEVEMENTS, dtype=np.int64)
    total_episodes = 0
    current_step = 0
    t_start = time.time()

    for it in range(num_iters):
        (params, opt_state, obs_stats, env_states, obs, rng, traj, metrics) = (
            train_step(params, opt_state, obs_stats, env_states, obs, rng)
        )
        current_step += steps_per_iter

        rewards_np = np.asarray(traj.reward)
        dones_np = np.asarray(traj.done)
        # We also need the live achievement mask at each step. Re-running
        # rocket_conditions in numpy is expensive; instead we peek at the
        # achievement state accumulated on env_states (after the scan).
        # For logging, we rely on the running_peak_mask of per-env
        # achievements-before-reset: inferred by accumulating newly-unlocked
        # reward > 0 events — but we already have raw reward so that's lossy.
        # Simplest: sample the wrapped state's achievements_unlocked at the
        # END of every iter's final scan step (current env_states).
        live_mask = np.asarray(env_states.achievements_unlocked)[
            :, :NUM_ROCKET_ACHIEVEMENTS
        ]
        running_peak_mask[:, :NUM_ROCKET_ACHIEVEMENTS] = np.maximum(
            running_peak_mask[:, :NUM_ROCKET_ACHIEVEMENTS], live_mask
        )

        for t in range(rewards_np.shape[0]):
            running_return += rewards_np[t]
            done_idx = np.where(dones_np[t])[0]
            for n in done_idx:
                ep_returns.append(float(running_return[n]))
                ach_n = int(running_peak_mask[n, :NUM_ROCKET_ACHIEVEMENTS].sum())
                ep_ach_counts.append(ach_n)
                per_ach_unlock_total += running_peak_mask[
                    n, :NUM_ROCKET_ACHIEVEMENTS
                ].astype(np.int64)
                best_ach_count_ever = max(best_ach_count_ever, ach_n)
                total_episodes += 1
                running_return[n] = 0.0
                running_peak_mask[n] = False

        if (it + 1) % config.log_interval == 0 or it == num_iters - 1:
            elapsed = time.time() - t_start
            sps = current_step / elapsed
            mean_ret = float(np.mean(list(ep_returns))) if ep_returns else 0.0
            mean_ach = float(np.mean(list(ep_ach_counts))) if ep_ach_counts else 0.0

            actions_np = np.asarray(traj.action).ravel()
            counts = np.bincount(actions_np, minlength=NUM_ACTIONS)
            top3 = np.argsort(counts)[::-1][:3]
            act_str = " ".join(
                f"{Action(a).name}={counts[a] / len(actions_np) * 100:.0f}%"
                for a in top3
            )

            logger.info(
                (
                    "iter=%d/%d  step=%dk  sps=%.0f  ret=%.2f  "
                    "ach=%.2f/%d  best=%d  ent=%.3f  | %s"
                ),
                it + 1,
                num_iters,
                current_step // 1000,
                sps,
                mean_ret,
                mean_ach,
                NUM_ROCKET_ACHIEVEMENTS,
                best_ach_count_ever,
                float(metrics["loss/entropy"]),
                act_str,
            )

            if wandb_run is not None:
                log_data: dict[str, float] = {
                    "train/step": float(current_step),
                    "train/sps": sps,
                    "train/mean_ep_return": mean_ret,
                    "train/mean_ep_achievements": mean_ach,
                    "train/best_achievements_ever": float(best_ach_count_ever),
                }
                for k, v in metrics.items():
                    log_data[f"train/{k}"] = float(v)
                wandb_run.log(log_data, step=current_step)

    elapsed = time.time() - t_start
    mean_ret = float(np.mean(list(ep_returns))) if ep_returns else 0.0
    mean_ach = float(np.mean(list(ep_ach_counts))) if ep_ach_counts else 0.0
    logger.info(
        "Done. %dk steps in %.1fs (%.0f sps). Final ret=%.2f  ach=%.2f/%d  best=%d",
        current_step // 1000,
        elapsed,
        current_step / elapsed,
        mean_ret,
        mean_ach,
        NUM_ROCKET_ACHIEVEMENTS,
        best_ach_count_ever,
    )

    # Per-achievement unlock rate over all finished episodes.
    denom = max(1, total_episodes)
    logger.info("Per-achievement unlock rate (%d episodes):", total_episodes)
    for i, info in enumerate(ROCKET_ACHIEVEMENT_INFO):
        weight = float(ROCKET_ACHIEVEMENT_WEIGHTS[i])
        count = int(per_ach_unlock_total[i])
        if count > 0:
            logger.info(
                "  [%2d] %-20s  weight=%d  unlocked=%d/%d (%.2f%%)",
                i,
                info.id,
                int(weight),
                count,
                total_episodes,
                100.0 * count / denom,
            )

    # Final artifacts — never in the hot path, runs once after training.
    try:
        _finalize_artifacts(
            config=config,
            env=env,
            env_params=env_params,
            initial_state=initial_state,
            network=network,
            params=params,
            obs_stats=obs_stats,
            wandb_run=wandb_run,
            current_step=current_step,
        )
    except Exception:  # noqa: BLE001
        logger.exception("Final artifact generation failed (training itself OK).")

    if wandb_run is not None:
        wandb_run.finish()

    return {
        "mean_ep_return": mean_ret,
        "mean_ep_achievements": mean_ach,
        "best_achievements_ever": float(best_ach_count_ever),
        "max_possible_score": float(MAX_ROCKET_SCORE),
        "sps": current_step / elapsed,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--total-steps", type=int, default=3_000_000)
    parser.add_argument("--max-timesteps", type=int, default=8000)
    parser.add_argument(
        "--obs-radius",
        type=int,
        default=7,
        help="Half-width of the local obs window (default 7 → 15x15 tiles).",
    )
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--no-anneal-lr", action="store_true")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="factoriax_rocket")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help=(
            "Directory for final artifacts (model + video). "
            "Defaults to runs/rocket_ppo/<wandb_run_name or 'default'>."
        ),
    )
    parser.add_argument(
        "--no-save-model",
        action="store_true",
        help="Skip saving the final model msgpack.",
    )
    parser.add_argument(
        "--no-save-video",
        action="store_true",
        help="Skip rendering and saving the final rollout MP4.",
    )
    parser.add_argument("--video-fps", type=int, default=30)
    args = parser.parse_args()

    config = Config(
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        total_steps=args.total_steps,
        max_timesteps=args.max_timesteps,
        obs_radius=args.obs_radius,
        learning_rate=args.learning_rate,
        entropy_coef=args.entropy_coef,
        seed=args.seed,
        log_interval=args.log_interval,
        anneal_lr=not args.no_anneal_lr,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        out_dir=args.out_dir,
        save_final_model=not args.no_save_model,
        save_final_video=not args.no_save_video,
        video_fps=args.video_fps,
    )
    train(config)


if __name__ == "__main__":
    main()
