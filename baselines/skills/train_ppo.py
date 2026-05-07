"""Train PPO on factoriax skill benchmarks.

Each skill is a small 5x5 grid task with dense rewards and 200-step
episodes. The two skills form an increasing difficulty ladder:
mining (walk + mine) and place_miner (navigate + place from inventory).

The entire collect-GAE-update pipeline is fused into a single
JIT-compiled ``train_step`` to maximize GPU throughput.

Reuses the shared PPO infrastructure from ``baselines.ppo`` and logs
to wandb under the ``fast_basic_skills_ppo`` project.

Usage::

    python -m baselines.skills.train_ppo mining
    python -m baselines.skills.train_ppo mining --total-steps 50_000_000
    python -m baselines.skills.train_ppo all --use-wandb
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
from factoriax.benchmarks.skills.mining import MiningSkill, mining_level
from factoriax.benchmarks.skills.place_miner import (
    PlaceMinerSkill,
    place_miner_level,
)
from factoriax.constants import NUM_ACTIONS, Action
from factoriax.envs.factoriax_env import FactoriaXEnv
from factoriax.state import EnvState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Maps skill name to environment class.
SKILL_ENVS: dict[str, type] = {
    "mining": MiningSkill,
    "place_miner": PlaceMinerSkill,
}


def _make_level(
    skill_name: str,
    map_size: int,
    max_timesteps: int,
) -> tuple[Any, Any]:
    """Build a level and env_params for the given skill at the specified scale.

    Args:
        skill_name: One of "mining", "place_miner".
        map_size: Square map side length.
        max_timesteps: Episode length.

    Returns:
        Tuple of (Level, EnvParams).
    """
    if skill_name == "mining":
        return mining_level(
            map_size=map_size,
            ore_fraction=0.5,
            max_timesteps=max_timesteps,
        )
    if skill_name == "place_miner":
        n = max(5, map_size * map_size // 8)
        return place_miner_level(
            map_size=map_size,
            num_patches=n,
            num_miners=n,
            max_timesteps=max_timesteps,
        )
    raise ValueError(f"Unknown skill: {skill_name!r}")


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

    skill_name: str = "mining"
    map_size: int = 12
    max_timesteps: int = 500
    hidden_dims: tuple[int, ...] = (256, 256)
    num_envs: int = 1024
    rollout_steps: int = 128
    total_steps: int = 5_000_000
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
    wandb_project: str = "fast_basic_skills_ppo"
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
    if config.skill_name not in SKILL_ENVS:
        raise ValueError(
            f"Unknown skill {config.skill_name!r}. Choose from: {list(SKILL_ENVS)}"
        )
    level, env_params = _make_level(
        config.skill_name,
        config.map_size,
        config.max_timesteps,
    )
    inner = FactoriaXEnv(level=level)
    env = SKILL_ENVS[config.skill_name](inner=inner)

    # Build initial state and determine observation shape.
    initial_obs, initial_state = env.reset_env(jax.random.PRNGKey(0), env_params)
    obs_dim = int(initial_obs.shape[0])

    logger.info(
        "Skill: %s  obs_dim=%d  num_actions=%d  num_envs=%d  rollout=%d  total=%dk",
        config.skill_name,
        obs_dim,
        NUM_ACTIONS,
        config.num_envs,
        config.rollout_steps,
        config.total_steps // 1000,
    )

    # W&B.
    wandb_run = None
    if config.use_wandb:
        try:
            import wandb  # type: ignore[import-untyped]

            run_name = config.wandb_run_name or f"ppo_{config.skill_name}"
            wandb_run = wandb.init(
                project=config.wandb_project,
                name=run_name,
                config=dataclasses.asdict(config),
                tags=["skills", "ppo", config.skill_name],
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

    from factoriax.jax_renderer import JaxRenderer
    from factoriax.levels import build_state

    logger.info("Running evaluation rollout...")
    state = build_state(level, env_params)
    jit_step = jax.jit(env.step_env)
    jit_apply = jax.jit(network.apply)
    rng_eval = jax.random.PRNGKey(config.seed + 9999)

    states: list[EnvState] = [state]
    actions_log: list[int] = []
    rewards_log: list[float] = []

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

    total_reward = sum(rewards_log)
    logger.info(
        "Eval: %d steps, total reward=%.1f",
        len(actions_log),
        total_reward,
    )

    # Render video.
    out_dir = Path("runs") / f"skills_ppo_{config.skill_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    renderer = JaxRenderer(tile_px=16)
    frames = [np.asarray(renderer.jit_render_map(s)) for s in states]
    mp4_path = out_dir / f"{config.skill_name}.mp4"
    try:
        import warnings

        import imageio.v3 as iio

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=RuntimeWarning, message="os.fork()"
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
        logger.error("imageio[ffmpeg] not available, skipping video.")
        return

    # Build trajectory for analysis charts.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from factoriax.analysis.actions import plot_action_proportions
    from factoriax.analysis.state import plot_inventory_proportions
    from factoriax.analysis.trajectory import states_to_trajectory

    act_arr = np.array(
        actions_log + [0] * (len(states) - len(actions_log)),
        dtype=np.int32,
    )
    rew_arr = np.array(
        rewards_log + [0.0] * (len(states) - len(rewards_log)),
        dtype=np.float32,
    )
    eval_traj = states_to_trajectory(states, actions=act_arr, rewards=rew_arr)

    figs: dict[str, plt.Figure] = {}

    fig_ap, _ = plot_action_proportions(
        eval_traj,
        episode=0,
        title=f"{config.skill_name} -- action proportions",
    )
    figs["action_proportions"] = fig_ap

    if eval_traj.player_inventory is not None:
        fig_ip, _ = plot_inventory_proportions(
            eval_traj,
            episode=0,
            title=f"{config.skill_name} -- inventory composition",
        )
        figs["inventory_proportions"] = fig_ip

    for plot_name, fig in figs.items():
        fig_path = out_dir / f"{config.skill_name}_{plot_name}.png"
        fig.savefig(fig_path, dpi=150)
        logger.info("Saved %s: %s", plot_name, fig_path)

    # Upload to wandb.
    if wandb_run is not None:
        try:
            import wandb  # type: ignore[import-untyped]

            log_data_eval: dict[str, object] = {
                "eval/total_reward": total_reward,
                "eval/episode_length": len(actions_log),
                f"videos/{config.skill_name}": wandb.Video(
                    str(mp4_path), fps=10, format="mp4"
                ),
            }
            for plot_name, fig in figs.items():
                log_data_eval[f"plots/{plot_name}"] = wandb.Image(fig)

            wandb_run.log(log_data_eval)
            logger.info("Uploaded video and charts to wandb.")
        except ImportError:
            pass

    for fig in figs.values():
        plt.close(fig)


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
        choices=list(SKILL_ENVS) + ["all"],
        help="Skill to train on, or 'all' to run each.",
    )
    parser.add_argument(
        "--map-size",
        type=int,
        default=12,
        help="Square map side length (default: 12).",
    )
    parser.add_argument(
        "--max-timesteps",
        type=int,
        default=500,
        help="Episode length (default: 500).",
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
        default=5_000_000,
        help="Total env steps (default: 5M).",
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
        help="Log to Weights and Biases.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="fast_basic_skills_ppo",
        help="W&B project name.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name (default: ppo_<skill>).",
    )
    args = parser.parse_args()

    skills = list(SKILL_ENVS) if args.skill == "all" else [args.skill]

    for skill_name in skills:
        config = Config(
            skill_name=skill_name,
            map_size=args.map_size,
            max_timesteps=args.max_timesteps,
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
