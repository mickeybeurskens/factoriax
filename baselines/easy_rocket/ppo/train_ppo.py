"""Train PPO on the easy rocket achievement scenario.

Loads the scenario via ``factoriax.make("EasyRocket-v1")`` — the env carries
the achievement conditions (a step hook) and the achievement reward (returned by
``step_env``), Craftax-style sparse +weight on each newly-unlocked achievement.

Each parallel env draws its own keyed layout from the scenario's reset_fn and
restores it on episode end (cheap cached reset). Per-episode regeneration is
available via the scenario's keyed ``AutoResetWrapper`` if wanted later.

Reuses the PPO infrastructure in this package (network, GAE, update,
normalization). The collect/GAE/update pipeline is fused into one
JIT-compiled step to keep GPU throughput high.

Usage::

    python -m baselines.easy_rocket.ppo.train_ppo
    python -m baselines.easy_rocket.ppo.train_ppo --num-envs 4 \\
        --rollout-steps 4 --total-steps 16
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import time
from collections import deque
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from baselines.easy_rocket.ppo.cli import add_ppo_args, ppo_config_from_args
from baselines.easy_rocket.ppo.config import PPOConfig
from baselines.easy_rocket.ppo.gae import Transition, compute_gae
from baselines.easy_rocket.ppo.network import ActorCritic
from baselines.easy_rocket.ppo.normalization import (
    RunningStats,
    init_running_stats,
    normalize_obs,
    update_running_stats,
)
from factoriax.analysis.eval import EvalRollout, generate_eval_plots
from factoriax.analysis.video import compose_frame_with_inventory, write_video
from factoriax.engine.constants import MAX_ACHIEVEMENTS, NUM_ACTIONS, Action
from factoriax.make import env_from_name
from factoriax.engine.envs.easy_rocket import (
    EASY_ROCKET_ACHIEVEMENT_NAMES,
    EASY_ROCKET_ACHIEVEMENT_WEIGHTS,
    MAX_EASY_ROCKET_SCORE,
    NUM_EASY_ROCKET_ACHIEVEMENTS,
)
from factoriax.engine.state import EnvParams, EnvState

if TYPE_CHECKING:
    from pathlib import Path

_SCENARIO_ID = "EasyRocket-v1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Every easy_rocket achievement is reachable through automated production —
# the bits read machine buffers, no belt-network stubs remain — so the soft
# ceiling equals the full count. Surfaced in logs alongside the live count.
REACHABLE_ACHIEVEMENTS: int = NUM_EASY_ROCKET_ACHIEVEMENTS

# Display labels come from the scenario, the single source of truth for the
# bit set and its order.
EASY_ROCKET_ACHIEVEMENT_LABELS: tuple[str, ...] = EASY_ROCKET_ACHIEVEMENT_NAMES


@dataclasses.dataclass
class Config:
    """Training configuration for PPO on the easy rocket scenario.

    Embeds a :class:`PPOConfig` for the shared PPO hyperparameters
    and keeps easy-rocket-specific fields (episode horizon, LR
    annealing, artifact toggles) flat at the top level. Defaults
    target a pilot run that finishes in a few minutes on a single
    GPU.

    The ``out_dir`` / ``save_final_*`` / ``video_fps`` fields are
    declared now so the Phase 2 artifact diff is small; v1 does not
    write artifacts.
    """

    ppo: PPOConfig = dataclasses.field(default_factory=PPOConfig)
    max_timesteps: int = 2000
    anneal_lr: bool = True
    #: When True, every parallel env resets from the same PRNG key, so all
    #: ``num_envs`` workers draw the same procgen layout. Useful for
    #: layout-invariant ablations. Default keeps the per-env keyed reset
    #: (procgen variation across parallel envs).
    fixed_env_seed: bool = False
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
    norm = normalize_obs(obs_stats, obs) if config.ppo.normalize_obs else obs
    logits, values = network.apply(params, norm)
    lp_all = jax.nn.log_softmax(logits)
    lp = lp_all[jnp.arange(obs.shape[0]), actions]

    probs = jax.nn.softmax(logits)
    entropy = -(probs * lp_all).sum(axis=-1).mean()

    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    ratio = jnp.exp(lp - old_lp)
    clip_eps = config.ppo.clip_eps
    pg_loss = -jnp.minimum(
        ratio * adv,
        jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv,
    ).mean()
    v_loss = 0.5 * ((values - rets) ** 2).mean()
    total = pg_loss + config.ppo.value_coef * v_loss - config.ppo.entropy_coef * entropy
    return total, {
        "loss/total": total,
        "loss/policy": pg_loss,
        "loss/value": v_loss,
        "loss/entropy": entropy,
        "misc/approx_kl": ((ratio - 1.0) - jnp.log(ratio)).mean(),
        "misc/clip_frac": (jnp.abs(ratio - 1.0) > clip_eps).astype(jnp.float32).mean(),
    }


def _resolve_out_dir(config: Config) -> Any:
    """Return the directory where final artifacts are written.

    Uses ``config.out_dir`` when set; otherwise derives
    ``runs/easy_rocket_ppo/{wandb_run_name or "default"}``.
    """
    from pathlib import Path  # noqa: PLC0415

    if config.out_dir is not None:
        return Path(config.out_dir)
    name = config.ppo.wandb_run_name or "default"
    return Path("runs") / "easy_rocket_ppo" / name


def _save_final_model(
    path: Any,
    params: Any,
    obs_stats: RunningStats,
    config: Config,
) -> None:
    """Serialize params + obs_stats to msgpack and config to sibling JSON."""
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
    env: Any,
    env_params: EnvParams,
    initial_state: EnvState,
    network: ActorCritic,
    params: Any,
    obs_stats: RunningStats,
) -> EvalRollout:
    """Run one eval episode and return collected artifacts.

    Python-side per-step loop so the underlying ``EnvState`` can be
    snapshotted at every tick for rendering and trajectory analysis.
    Fine here because the eval is a one-shot ~2000-step rollout, not
    a hot path.
    """
    jit_step = jax.jit(env.step_env)
    jit_apply = jax.jit(network.apply)
    rng = jax.random.PRNGKey(config.ppo.seed + 4242)
    state = initial_state

    frames: list[np.ndarray] = [compose_frame_with_inventory(state)]
    env_states: list[Any] = [state]
    ach_per_step: list[np.ndarray] = [np.asarray(state.achievements_unlocked)]
    actions_log: list[int] = []

    for _ in range(env_params.max_timesteps):
        obs = env.get_obs(state, env_params)
        norm = normalize_obs(obs_stats, obs) if config.ppo.normalize_obs else obs
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
    env: Any,
    env_params: EnvParams,
    initial_state: EnvState,
    network: ActorCritic,
    params: Any,
    obs_stats: RunningStats,
    wandb_run: Any | None,
    current_step: int,
) -> None:
    """Write final model + rollout video + analysis plots.

    Runs strictly after training completes; never touches the hot
    loop. Uploads the model, video, and PNGs as W&B Artifacts and
    inline images when ``wandb_run is not None``.
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
            assert rollout.frames is not None
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
            achievement_labels=list(EASY_ROCKET_ACHIEVEMENT_LABELS),
            num_achievements=NUM_EASY_ROCKET_ACHIEVEMENTS,
            title_prefix="Final rollout",
        )
        for name, path in plot_paths.items():
            logger.info("Saved %s plot: %s", name, path)

    if wandb_run is None:
        return

    import wandb  # noqa: PLC0415  # type: ignore[import-untyped]

    run_id = getattr(wandb_run, "id", "run")

    if model_path is not None:
        artifact = wandb.Artifact(f"easy-rocket-ppo-model-{run_id}", type="model")
        artifact.add_file(str(model_path))
        # Ship the sibling config JSON too, so the artifact is self-describing
        # (seed, num_envs, hidden_dims) without consulting the run record.
        config_path = model_path.with_suffix(".config.json")
        if config_path.exists():
            artifact.add_file(str(config_path))
        wandb_run.log_artifact(artifact)
        logger.info("Uploaded model artifact to wandb.")

    if video_path is not None:
        vid_artifact = wandb.Artifact(f"easy-rocket-ppo-video-{run_id}", type="video")
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
            f"easy-rocket-ppo-plots-{run_id}",
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
        unlocked_labels = [
            EASY_ROCKET_ACHIEVEMENT_LABELS[i]
            for i in range(NUM_EASY_ROCKET_ACHIEVEMENTS)
            if bool(ach_mask[i])
        ]
        score = float(
            np.sum(
                np.asarray(EASY_ROCKET_ACHIEVEMENT_WEIGHTS)[
                    :NUM_EASY_ROCKET_ACHIEVEMENTS
                ]
                * ach_mask[:NUM_EASY_ROCKET_ACHIEVEMENTS]
            )
        )
        unlocked_str = ", ".join(unlocked_labels) if unlocked_labels else "(none)"
        ach_count = int(ach_mask[:NUM_EASY_ROCKET_ACHIEVEMENTS].sum())
        wandb_run.log(
            {
                "final/achievements_count": ach_count,
                "final/score": score,
                "final/unlocked": unlocked_str,
            },
            step=current_step,
        )


def _query_gpu_name() -> str:
    """One-shot nvidia-smi query for the GPU model. ``"unknown"`` on miss."""
    import subprocess  # noqa: PLC0415

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            timeout=5,
        ).strip()
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        return "unknown"
    first = out.split("\n", 1)[0].strip()
    return first or "unknown"


def _write_throughput_json(
    output_path: str,
    scenario: str,
    config: Config,
    steady_sps: float,
    warmup_wall_samples: list[float],
    iter_wall_samples: list[float],
    steps_per_iter: int,
    achievement_names: tuple[str, ...] | None = None,
    first_unlock_step: np.ndarray | None = None,
    final_unlock_rate: np.ndarray | None = None,
) -> Path:
    """Write the throughput JSON consumed by paper/figures/throughput.py.

    The GPU model name (via ``nvidia-smi``) is suffixed onto the
    basename so a caller can pass the same ``--throughput-json`` path
    across devices without overwriting prior runs, and so the
    resulting filename matches the paper-side consolidator's
    ``ppo_throughput_*.json`` glob. The schema mirrors
    :mod:`scripts.ppo_throughput_bench` so the paper-side consolidator
    and figure builder consume both producers interchangeably.
    """
    import re  # noqa: PLC0415
    import statistics  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    import orjson  # noqa: PLC0415

    gpu_name = _query_gpu_name()
    safe_name = re.sub(r"[^A-Za-z0-9]+", "-", gpu_name).strip("-") or "unknown"

    ppo = config.ppo
    p = Path(output_path)
    out_path = p.with_name(f"{p.stem}_{safe_name}_envs{ppo.num_envs}{p.suffix}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    measured_steps = len(iter_wall_samples) * steps_per_iter
    warmup_seconds = sum(warmup_wall_samples) if warmup_wall_samples else 0.0
    if iter_wall_samples:
        mean_wall = statistics.mean(iter_wall_samples)
        sps_mean = steps_per_iter / mean_wall if mean_wall > 0 else 0.0
    else:
        sps_mean = 0.0
    wallclock_seconds = sum(iter_wall_samples)

    payload = {
        "scenario": scenario,
        "gpu_name": gpu_name,
        "num_envs": ppo.num_envs,
        "rollout_steps": ppo.rollout_steps,
        "update_epochs": ppo.update_epochs,
        "num_minibatches": ppo.num_minibatches,
        "hidden_dims": list(ppo.hidden_dims),
        "total_steps": ppo.total_steps,
        "jax_version": jax.__version__,
        "runs": [
            {
                "device": gpu_name,
                "device_full": gpu_name,
                "num_envs": ppo.num_envs,
                "rollout_steps": ppo.rollout_steps,
                "measured_steps": measured_steps,
                "steady_state_sps": steady_sps,
                "steady_state_sps_mean": sps_mean,
                "wallclock_seconds": wallclock_seconds,
                "startup_seconds": warmup_seconds,
                "warmup_samples": warmup_wall_samples,
                "iter_samples": iter_wall_samples,
            }
        ],
    }
    # Per-bit empirical-order data, if the caller collected it.
    if achievement_names is not None and (
        first_unlock_step is not None or final_unlock_rate is not None
    ):
        achievements_block: dict[str, object] = {"names": list(achievement_names)}
        if first_unlock_step is not None:
            achievements_block["first_unlock_step"] = [
                int(x) for x in first_unlock_step.tolist()
            ]
        if final_unlock_rate is not None:
            achievements_block["final_unlock_rate"] = [
                float(x) for x in final_unlock_rate.tolist()
            ]
        payload["achievements"] = achievements_block
    out_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    logger.info(
        "wrote throughput JSON to %s (gpu=%s, %d iters, %.0f sps)",
        out_path,
        gpu_name,
        len(iter_wall_samples),
        steady_sps,
    )
    return out_path


def train(config: Config) -> dict[str, float]:
    """Train PPO against the easy_rocket scenario and return final metrics."""
    ppo = config.ppo
    env, env_params = env_from_name(_SCENARIO_ID, auto_reset=False)
    env_params = env_params.replace(max_timesteps=config.max_timesteps)
    obs_dim = int(env.observation_space(env_params).shape[0])
    logger.info(
        "Easy rocket PPO  obs_dim=%d  actions=%d  envs=%d  rollout=%d  total=%dk",
        obs_dim,
        NUM_ACTIONS,
        ppo.num_envs,
        ppo.rollout_steps,
        ppo.total_steps // 1000,
    )

    wandb_run = None
    if ppo.use_wandb:
        try:
            import wandb  # type: ignore[import-untyped]

            run_name = ppo.wandb_run_name or "ppo_easy_rocket"
            wandb_run = wandb.init(
                project=ppo.wandb_project,
                name=run_name,
                config=dataclasses.asdict(config),
                tags=["easy_rocket", "ppo", "achievement", "global_observation"],
            )
        except ImportError:
            logger.error("wandb not installed. Run: uv add wandb")

    network = ActorCritic(
        hidden_dims=ppo.hidden_dims,
        num_actions=NUM_ACTIONS,
    )
    rng = jax.random.PRNGKey(ppo.seed)
    rng, key_init = jax.random.split(rng)
    params = network.init(key_init, jnp.zeros(obs_dim))

    steps_per_iter = ppo.num_envs * ppo.rollout_steps
    num_iters = max(1, ppo.total_steps // steps_per_iter)
    if config.anneal_lr:
        total_opt_steps = num_iters * ppo.update_epochs * ppo.num_minibatches
        lr_schedule: optax.ScalarOrSchedule = optax.linear_schedule(
            init_value=ppo.learning_rate,
            end_value=0.0,
            transition_steps=total_opt_steps,
        )
    else:
        lr_schedule = ppo.learning_rate

    optimizer = optax.chain(
        optax.clip_by_global_norm(ppo.max_grad_norm),
        optax.adam(lr_schedule, eps=1e-5),
    )
    opt_state = optimizer.init(params)

    obs_stats = init_running_stats(obs_dim)

    vmap_step = jax.vmap(env.step_env, in_axes=(0, 0, 0, None))
    vmap_obs = jax.vmap(env.get_obs, in_axes=(0, None))
    vmap_reset = jax.vmap(env.reset_env, in_axes=(0, None))

    # Each parallel env draws its own keyed layout from the scenario's reset_fn
    # and restores it on episode end (cheap cached reset). Reward comes from the
    # env's bound reward_fn via step_env. ``fixed_env_seed`` broadcasts a single
    # key to all workers so every env starts on the same layout (procgen off).
    rng, reset_rng = jax.random.split(rng)
    if config.fixed_env_seed:
        env_reset_keys = jnp.broadcast_to(reset_rng, (ppo.num_envs, *reset_rng.shape))
    else:
        env_reset_keys = jax.random.split(reset_rng, ppo.num_envs)
    _reset_obs, reset_states = vmap_reset(env_reset_keys, env_params)

    mb_size = steps_per_iter // ppo.num_minibatches

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
        """One fused training iteration: rollout + GAE + PPO update."""
        rng, key_collect, key_update = jax.random.split(rng, 3)

        def _rollout_step(
            carry: tuple[EnvState, jax.Array, jax.Array],
            _: None,
        ) -> tuple[tuple[EnvState, jax.Array, jax.Array], Transition]:
            states, cur_obs, key = carry
            key, key_act, key_step = jax.random.split(key, 3)

            norm = normalize_obs(obs_stats, cur_obs) if ppo.normalize_obs else cur_obs
            logits, values = network.apply(params, norm)
            actions = jax.random.categorical(key_act, logits)
            log_probs = jax.nn.log_softmax(logits)[jnp.arange(ppo.num_envs), actions]

            keys = jax.random.split(key_step, ppo.num_envs)
            _, next_states, rewards, dones, _ = vmap_step(
                keys, states, actions, env_params
            )

            def _where(r: jax.Array, s: jax.Array) -> jax.Array:
                mask = dones.reshape((-1,) + (1,) * (s.ndim - 1))
                return jnp.where(mask, r, s)

            next_states = jax.tree_util.tree_map(_where, reset_states, next_states)
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
        obs_stats = (
            update_running_stats(obs_stats, flat_obs)
            if ppo.normalize_obs
            else obs_stats
        )

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
                return x[perm].reshape((ppo.num_minibatches, mb_size) + x.shape[1:])

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
            length=ppo.update_epochs,
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

    logger.info(
        "%d steps/iter  %d iters  %dk total",
        steps_per_iter,
        num_iters,
        num_iters * steps_per_iter // 1000,
    )

    env_states = reset_states
    obs = vmap_obs(reset_states, env_params)
    ep_returns: deque[float] = deque(maxlen=500)
    ep_ach_counts: deque[int] = deque(maxlen=500)
    # Per-episode bit vector deque for computing rolling per-bit
    # unlock rates; sibling of ``ep_ach_counts`` but keeps the vector
    # instead of collapsing to a sum.
    ep_ach_masks: deque[np.ndarray] = deque(maxlen=500)
    running_return = np.zeros(ppo.num_envs, dtype=np.float32)
    running_peak_mask = np.zeros((ppo.num_envs, MAX_ACHIEVEMENTS), dtype=bool)
    best_ach_count_ever = 0
    # First env-step at which each bit unlocked in any parallel env (-1
    # while still locked). Updated each iter from the live mask.
    first_unlock_step = np.full(NUM_EASY_ROCKET_ACHIEVEMENTS, -1, dtype=np.int64)
    total_episodes = 0
    current_step = 0
    # Throughput is reported warmup-free. Iter 0 pays the one-time JIT
    # compile, so its wall time is recorded separately as the compile cost
    # and excluded from the steady-state steps/sec.
    warmup_seconds = 0.0
    steady_steps = 0
    steady_start = 0.0
    marginal_sps = 0.0
    t_prev = time.time()
    # Per-iteration wall time samples for the throughput-json dump.
    # ``warmup_wall_samples`` captures iter 0 (compile + first launch);
    # ``iter_wall_samples`` captures every steady-state iter that
    # follows. Kept as separate lists so a downstream consumer never
    # mixes the one-time compile cost into the throughput aggregates.
    warmup_wall_samples: list[float] = []
    iter_wall_samples: list[float] = []

    for it in range(num_iters):
        (params, opt_state, obs_stats, env_states, obs, rng, traj, metrics) = (
            train_step(params, opt_state, obs_stats, env_states, obs, rng)
        )
        jax.block_until_ready(metrics)
        t_now = time.time()
        iter_seconds = t_now - t_prev
        t_prev = t_now
        current_step += steps_per_iter
        if it == 0:
            warmup_seconds = iter_seconds
            warmup_wall_samples.append(iter_seconds)
            steady_start = t_now
        else:
            steady_steps += steps_per_iter
            marginal_sps = steps_per_iter / iter_seconds
            iter_wall_samples.append(iter_seconds)

        rewards_np = np.asarray(traj.reward)
        dones_np = np.asarray(traj.done)
        live_mask = np.asarray(env_states.achievements_unlocked)[
            :, :NUM_EASY_ROCKET_ACHIEVEMENTS
        ]
        running_peak_mask[:, :NUM_EASY_ROCKET_ACHIEVEMENTS] = np.maximum(
            running_peak_mask[:, :NUM_EASY_ROCKET_ACHIEVEMENTS], live_mask
        )
        # Stamp first_unlock_step for any bit that just appeared
        # anywhere in the batch and has not been stamped yet.
        any_unlocked = live_mask.any(axis=0)
        newly_unlocked = any_unlocked & (first_unlock_step == -1)
        if newly_unlocked.any():
            first_unlock_step[newly_unlocked] = current_step

        for t in range(rewards_np.shape[0]):
            running_return += rewards_np[t]
            done_idx = np.where(dones_np[t])[0]
            for n in done_idx:
                ep_returns.append(float(running_return[n]))
                ep_mask = running_peak_mask[n, :NUM_EASY_ROCKET_ACHIEVEMENTS].copy()
                ep_ach_masks.append(ep_mask)
                ach_n = int(ep_mask.sum())
                ep_ach_counts.append(ach_n)
                best_ach_count_ever = max(best_ach_count_ever, ach_n)
                total_episodes += 1
                running_return[n] = 0.0
                running_peak_mask[n] = False

        if (it + 1) % ppo.log_interval == 0 or it == num_iters - 1:
            mean_ret = float(np.mean(list(ep_returns))) if ep_returns else 0.0
            mean_ach = float(np.mean(list(ep_ach_counts))) if ep_ach_counts else 0.0

            actions_np = np.asarray(traj.action).ravel()
            counts = np.bincount(actions_np, minlength=NUM_ACTIONS)
            top3 = np.argsort(counts)[::-1][:3]
            act_str = " ".join(
                f"{Action(a).name}={counts[a] / len(actions_np) * 100:.0f}%"
                for a in top3
            )

            if it == 0:
                logger.info(
                    "iter=1/%d  compile+warmup=%.1fs  step=%dk  | %s",
                    num_iters,
                    warmup_seconds,
                    current_step // 1000,
                    act_str,
                )
            else:
                logger.info(
                    (
                        "iter=%d/%d  step=%dk  sps=%.0f  ret=%.2f  "
                        "ach=%.2f/%d (ceil=%d)  best=%d  ent=%.3f  | %s"
                    ),
                    it + 1,
                    num_iters,
                    current_step // 1000,
                    marginal_sps,
                    mean_ret,
                    mean_ach,
                    NUM_EASY_ROCKET_ACHIEVEMENTS,
                    REACHABLE_ACHIEVEMENTS,
                    best_ach_count_ever,
                    float(metrics["loss/entropy"]),
                    act_str,
                )

            if wandb_run is not None:
                log_data: dict[str, float] = {
                    "train/step": float(current_step),
                    "train/mean_ep_return": mean_ret,
                    "train/mean_ep_achievements": mean_ach,
                    "train/best_achievements_ever": float(best_ach_count_ever),
                }
                # Per-bit rolling unlock rate + first-unlock-step so the
                # paper-side empirical-order analysis can read off
                # which bits fired when. Rates are means over the
                # ep_ach_masks deque (last 500 episodes).
                if ep_ach_masks:
                    rates = np.mean(np.stack(list(ep_ach_masks)), axis=0)
                    for idx, name in enumerate(EASY_ROCKET_ACHIEVEMENT_NAMES):
                        log_data[f"bit/{name}/recent_unlock_rate"] = float(rates[idx])
                for idx, name in enumerate(EASY_ROCKET_ACHIEVEMENT_NAMES):
                    fu = int(first_unlock_step[idx])
                    if fu >= 0:
                        log_data[f"bit/{name}/first_unlock_step"] = float(fu)
                # Warmup-free throughput: per-iteration steps/sec, skipping
                # iter 0 (which pays the one-time JIT compile).
                if it > 0:
                    log_data["perf/sps"] = marginal_sps
                for k, v in metrics.items():
                    log_data[f"train/{k}"] = float(v)
                wandb_run.log(log_data, step=current_step)

    steady_elapsed = t_prev - steady_start
    steady_sps = steady_steps / steady_elapsed if steady_elapsed > 0 else float("nan")
    mean_ret = float(np.mean(list(ep_returns))) if ep_returns else 0.0
    mean_ach = float(np.mean(list(ep_ach_counts))) if ep_ach_counts else 0.0
    logger.info(
        (
            "Done. %dk steps  compile=%.1fs  steady-state=%.0f sps "
            "(%d iters, warmup-free).  Final ret=%.2f  "
            "ach=%.2f/%d (ceil=%d)  best=%d  episodes=%d"
        ),
        current_step // 1000,
        warmup_seconds,
        steady_sps,
        max(num_iters - 1, 0),
        mean_ret,
        mean_ach,
        NUM_EASY_ROCKET_ACHIEVEMENTS,
        REACHABLE_ACHIEVEMENTS,
        best_ach_count_ever,
        total_episodes,
    )

    if wandb_run is not None:
        import wandb  # noqa: PLC0415

        mean_steady_iter = steady_elapsed / max(num_iters - 1, 1)
        iter_time_table = wandb.Table(
            data=[
                ["compile (iter 0)", warmup_seconds],
                ["steady iter (mean)", mean_steady_iter],
            ],
            columns=["phase", "seconds"],
        )
        wandb_run.log(
            {
                "perf/iter_time": wandb.plot.bar(
                    iter_time_table,
                    "phase",
                    "seconds",
                    title="Compile (warmup) vs steady iteration time",
                ),
                "perf/compile_seconds": warmup_seconds,
                "perf/steady_sps": steady_sps,
            },
            step=current_step,
        )

    # Throughput JSON: default into the run's output directory (per-run);
    # an explicit --throughput-json path overrides the location; "" disables.
    if ppo.throughput_json != "":
        out_dir = _resolve_out_dir(config)
        throughput_target = ppo.throughput_json or str(out_dir / "ppo_throughput.json")
        final_unlock_rate = (
            np.mean(np.stack(list(ep_ach_masks)), axis=0)
            if ep_ach_masks
            else np.zeros(NUM_EASY_ROCKET_ACHIEVEMENTS, dtype=np.float32)
        )
        throughput_path = _write_throughput_json(
            output_path=throughput_target,
            scenario=_SCENARIO_ID,
            config=config,
            steady_sps=steady_sps,
            warmup_wall_samples=warmup_wall_samples,
            iter_wall_samples=iter_wall_samples,
            steps_per_iter=steps_per_iter,
            achievement_names=EASY_ROCKET_ACHIEVEMENT_NAMES,
            first_unlock_step=first_unlock_step,
            final_unlock_rate=final_unlock_rate,
        )
        if wandb_run is not None:
            try:
                import wandb  # noqa: PLC0415

                # Files tab (visible on the run page) + a versioned artifact
                # the paper-side consolidator can pull.
                wandb_run.save(str(throughput_path), policy="now")
                artifact = wandb.Artifact(
                    f"ppo-throughput-{wandb_run.id}", type="throughput"
                )
                artifact.add_file(str(throughput_path))
                wandb_run.log_artifact(artifact)
            except Exception:  # noqa: BLE001
                logger.exception("Throughput JSON upload to W&B failed.")

    # Render the eval episode on the layout the agent actually trained on:
    # env 0 of the training reset. A fresh PRNGKey(seed) reset would draw a
    # different procgen layout, filming the policy on a map it never saw.
    eval_initial_state = jax.tree_util.tree_map(lambda leaf: leaf[0], reset_states)
    try:
        _finalize_artifacts(
            config=config,
            env=env,
            env_params=env_params,
            initial_state=eval_initial_state,
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
        "max_possible_score": float(MAX_EASY_ROCKET_SCORE),
        "reachable_achievements": float(REACHABLE_ACHIEVEMENTS),
        "sps": steady_sps,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_ppo_args(parser)
    parser.set_defaults(
        num_envs=256,
        total_steps=2_000_000,
        log_interval=1,
        wandb_project="factoriax_easy_rocket",
    )
    parser.add_argument("--max-timesteps", type=int, default=2000)
    parser.add_argument("--no-anneal-lr", action="store_true")
    parser.add_argument(
        "--fixed-env-seed",
        action="store_true",
        help=(
            "Broadcast a single reset key to all parallel envs so every "
            "worker draws the same procgen layout. Default is per-env keyed."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help=(
            "Directory for final artifacts (model + video). "
            "Defaults to runs/easy_rocket_ppo/<wandb_run_name or 'default'>."
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

    ppo = ppo_config_from_args(args)
    config = Config(
        ppo=ppo,
        max_timesteps=args.max_timesteps,
        anneal_lr=not args.no_anneal_lr,
        fixed_env_seed=args.fixed_env_seed,
        out_dir=args.out_dir,
        save_final_model=not args.no_save_model,
        save_final_video=not args.no_save_video,
        video_fps=args.video_fps,
    )
    train(config)


if __name__ == "__main__":
    main()
