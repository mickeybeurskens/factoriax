"""PPO training on the basic_skills benchmark.

Trains a shared policy across the three basic_skills levels (mine_resources,
craft_pallets, fill_pallet) using mixed sampling. Each level has its own reward
function: sparse mining for mining, sparse crafting for crafting, and pallet
filling for the deposit level.

All three levels have different map sizes, so local observations with a fixed
radius keep the input dimension constant. The full 15-action space is used
since crafting and depositing require recipe cycling, slot selection, and the
DEPOSIT action.

Usage::

    python -m baselines.basic_skills.train_ppo --mode mixed --use-wandb
    python -m baselines.basic_skills.train_ppo --mode single --use-wandb
    python -m baselines.basic_skills.train_ppo --total-steps 100000  # smoke test
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import pickle
import time
from collections import deque
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
from factoriax.benchmarks.basic_skills.benchmark import (
    REWARD_FNS,
    BasicSkillsBenchmark,
)
from factoriax.benchmarks.core import BenchmarkLevel
from factoriax.benchmarks.runner import BenchmarkRunner
from factoriax.constants import NUM_ACTIONS
from factoriax.envs import FactoriaXEnv
from factoriax.levels import build_state
from factoriax.observations import local_array
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
    """Hyperparameters for PPO training on the basic_skills benchmark.

    Embeds a ``PPOConfig`` for shared PPO hyperparameters and adds
    domain-specific fields for the basic_skills benchmark.

    Attributes:
        ppo: Shared PPO hyperparameters (network, optimizer, logging, etc.).
        eval_seeds: Number of evaluation seeds for benchmark scoring.
        mode: Training mode. ``"mixed"`` trains one shared policy on all
            levels simultaneously. ``"single"`` trains a separate policy
            per level (one run each).
    """

    ppo: PPOConfig = dataclasses.field(default_factory=PPOConfig)
    eval_seeds: int = 10
    mode: str = "mixed"


# ---------------------------------------------------------------------------
# Per-level collection (with per-level reward function)
# ---------------------------------------------------------------------------


def make_collect_fn(
    env: FactoriaXEnv,
    bench_level: BenchmarkLevel,
    reward_fn: Callable,
    network: ActorCritic,
    config: Config,
) -> tuple[Callable, Callable]:
    """Build a JIT-compiled collection function for one benchmark level.

    Each level gets its own reward function, which is the key difference
    from the mining baseline.

    Args:
        env: FactoriaX environment instance.
        bench_level: Level to build the collector for.
        reward_fn: Reward function for this specific level.
        network: Shared actor-critic network.
        config: Training configuration.

    Returns:
        Tuple ``(collect_fn, init_fn)``.
    """
    env_params = bench_level.env_params
    level_state = build_state(bench_level.level, env_params)
    num_envs = config.ppo.num_envs
    obs_radius = config.ppo.obs_radius

    vmap_step = jax.vmap(env.step_env, in_axes=(0, 0, 0, None))
    vmap_reward = jax.vmap(reward_fn, in_axes=(0, 0, None))

    def _obs(state: EnvState) -> jax.Array:
        return local_array(state, env_params, state.selected_player, obs_radius)

    vmap_obs = jax.vmap(_obs)

    def _broadcast(x: jax.Array) -> jax.Array:
        a = jnp.asarray(x)
        return jnp.broadcast_to(a[None], (num_envs,) + a.shape)

    fixed_states = jax.tree_util.tree_map(_broadcast, level_state)

    def init() -> tuple[EnvState, jax.Array]:
        """Return initial env states and observations for this level."""
        return fixed_states, vmap_obs(fixed_states)

    ppo = config.ppo

    @jax.jit
    def collect(
        params: Any,
        obs_stats: RunningStats,
        env_states: EnvState,
        obs: jax.Array,
        rng: jax.Array,
    ) -> tuple[Transition, EnvState, jax.Array, jax.Array, jax.Array]:
        """Collect ``rollout_steps`` transitions from ``num_envs`` environments."""

        def rollout_step(
            carry: tuple[EnvState, jax.Array, jax.Array],
            _: None,
        ) -> tuple[tuple[EnvState, jax.Array, jax.Array], Transition]:
            states, cur_obs, step_rng = carry
            step_rng, key_act, key_step = jax.random.split(step_rng, 3)

            norm = normalize_obs(obs_stats, cur_obs) if ppo.normalize_obs else cur_obs
            logits, values = network.apply(params, norm)

            actions = jax.random.categorical(key_act, logits)
            log_probs = jax.nn.log_softmax(logits)[jnp.arange(num_envs), actions]

            keys_step = jax.random.split(key_step, num_envs)
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

            return (next_states, next_obs, step_rng), Transition(
                obs=cur_obs,
                action=actions,
                log_prob=log_probs,
                value=values,
                reward=rewards,
                done=dones,
            )

        (next_states, next_obs, rng), trajectories = jax.lax.scan(
            rollout_step,
            (env_states, obs, rng),
            None,
            length=ppo.rollout_steps,
        )
        norm_last = (
            normalize_obs(obs_stats, next_obs) if ppo.normalize_obs else next_obs
        )
        _, last_values = network.apply(params, norm_last)
        return trajectories, next_states, next_obs, last_values, rng

    return collect, init


# ---------------------------------------------------------------------------
# Mixed training (all levels each iteration)
# ---------------------------------------------------------------------------


def _run_mixed(
    levels: list[BenchmarkLevel],
    level_reward_fns: list[Callable],
    env: FactoriaXEnv,
    network: ActorCritic,
    update_fn: Callable,
    params: Any,
    obs_stats: RunningStats,
    opt_state: optax.OptState,
    obs_dim: int,
    config: Config,
    rng: jax.Array,
    wandb_run: Any | None,
) -> tuple[Any, RunningStats]:
    """Train with mixed sampling: collect from all levels each iteration.

    Each level uses its own reward function. Transitions from all levels
    are concatenated into a single batch for the PPO update.

    Args:
        levels: Benchmark levels.
        level_reward_fns: Reward function per level (same order).
        env: FactoriaX environment instance.
        network: Shared actor-critic network.
        update_fn: JIT-compiled PPO update function.
        params: Initial network parameters.
        obs_stats: Initial running observation statistics.
        opt_state: Initial optimizer state.
        obs_dim: Observation vector length.
        config: Training configuration.
        rng: PRNG key.
        wandb_run: Live W&B run or None.

    Returns:
        Tuple ``(params, obs_stats)`` after training.
    """
    ppo = config.ppo
    num_levels = len(levels)
    steps_per_iter_per_level = ppo.num_envs * ppo.rollout_steps
    steps_per_iter_total = steps_per_iter_per_level * num_levels
    num_iters = max(1, ppo.total_steps // steps_per_iter_total)

    logger.info(
        "Mixed: %d levels x %d steps/iter = %d combined steps/iter (%d iters total)",
        num_levels,
        steps_per_iter_per_level,
        steps_per_iter_total,
        num_iters,
    )

    collect_fns: list[Callable] = []
    level_env_states: list[EnvState] = []
    level_obs: list[jax.Array] = []
    for bl, rfn in zip(levels, level_reward_fns):
        collect_fn, init_fn = make_collect_fn(env, bl, rfn, network, config)
        env_states_l, obs_l = init_fn()
        collect_fns.append(collect_fn)
        level_env_states.append(env_states_l)
        level_obs.append(obs_l)

    t_start = time.time()
    per_level_returns: dict[str, deque] = {bl.name: deque(maxlen=500) for bl in levels}
    running_ep_return = np.zeros(ppo.num_envs * num_levels, dtype=np.float32)
    current_step = 0

    for it in range(num_iters):
        rng, key_update = jax.random.split(rng)

        all_obs: list[jax.Array] = []
        all_actions: list[jax.Array] = []
        all_log_probs: list[jax.Array] = []
        all_adv: list[jax.Array] = []
        all_ret: list[jax.Array] = []

        for l_idx, (collect_fn, bl) in enumerate(zip(collect_fns, levels)):
            rng, key_l = jax.random.split(rng)
            (
                traj,
                level_env_states[l_idx],
                level_obs[l_idx],
                last_vals,
                _,
            ) = collect_fn(
                params,
                obs_stats,
                level_env_states[l_idx],
                level_obs[l_idx],
                key_l,
            )

            adv, ret = compute_gae(
                traj.reward,
                traj.value,
                traj.done,
                last_vals,
                ppo.gamma,
                ppo.gae_lambda,
            )

            all_obs.append(traj.obs.reshape(-1, obs_dim))
            all_actions.append(traj.action.reshape(-1))
            all_log_probs.append(traj.log_prob.reshape(-1))
            all_adv.append(adv.reshape(-1))
            all_ret.append(ret.reshape(-1))

            # Track per-level episode returns.
            rewards_np = np.array(traj.reward)
            dones_np = np.array(traj.done)
            offset = l_idx * ppo.num_envs
            for t in range(rewards_np.shape[0]):
                running_ep_return[offset : offset + ppo.num_envs] += rewards_np[t]
                for n in np.where(dones_np[t])[0]:
                    per_level_returns[bl.name].append(
                        float(running_ep_return[offset + n])
                    )
                    running_ep_return[offset + n] = 0.0

        flat_obs = jnp.concatenate(all_obs)
        obs_stats = update_running_stats(obs_stats, flat_obs)

        params, opt_state, metrics, _ = update_fn(
            params,
            opt_state,
            obs_stats,
            flat_obs,
            jnp.concatenate(all_actions),
            jnp.concatenate(all_log_probs),
            jnp.concatenate(all_adv),
            jnp.concatenate(all_ret),
            key_update,
        )
        current_step += steps_per_iter_total

        if (it + 1) % ppo.log_interval == 0 or it == num_iters - 1:
            elapsed = time.time() - t_start
            sps = current_step / elapsed
            log_data: dict[str, float] = {
                "train/step": float(current_step),
                "train/sps": sps,
                **{f"train/{k}": float(v) for k, v in metrics.items()},
            }
            ret_parts = []
            for bl in levels:
                rets = per_level_returns[bl.name]
                mean_r = float(np.mean(list(rets))) if rets else 0.0
                log_data[f"train/{bl.name}/mean_ep_return"] = mean_r
                ret_parts.append(f"{bl.name}={mean_r:.2f}")

            logger.info(
                "iter=%d/%d  step=%d  sps=%.0f  %s  loss=%.4f  entropy=%.4f",
                it + 1,
                num_iters,
                current_step,
                sps,
                "  ".join(ret_parts),
                float(metrics["loss/total"]),
                float(metrics["loss/entropy"]),
            )
            if wandb_run is not None:
                wandb_run.log(log_data, step=current_step)

    return params, obs_stats


# ---------------------------------------------------------------------------
# Benchmark evaluation
# ---------------------------------------------------------------------------


def _plot_eval_diagnostics(
    traj_paths: dict[str, Path],
) -> dict[str, dict[str, plt.Figure]]:
    """Generate reward and action-sequence diagnostic plots per level.

    For each level, produces:
    - ``rewards``: per-step and cumulative reward curves.
    - ``ngram_sweep``: top-3 n-grams for n = 2..10.

    Args:
        traj_paths: Mapping from level name to saved ``.npz`` path.

    Returns:
        Nested dict ``{level_name: {plot_name: Figure}}``.
    """
    from factoriax.analysis.actions import action_raster, plot_ngram_sweep
    from factoriax.analysis.state import plot_episode_rewards
    from factoriax.analysis.trajectory import Trajectory

    results: dict[str, dict[str, plt.Figure]] = {}
    for name, path in traj_paths.items():
        traj = Trajectory.load(str(path))
        figs: dict[str, plt.Figure] = {}

        if traj.rewards is not None:
            fig_r, _ = plot_episode_rewards(
                traj,
                episode=0,
                title=f"{name} -- evaluation rewards",
            )
            figs["rewards"] = fig_r

        fig_ar, _ = action_raster(
            traj,
            title=f"{name} -- action raster",
        )
        figs["action_raster"] = fig_ar

        fig_ng, _ = plot_ngram_sweep(
            traj,
            n_range=(2, 5),
            top_k=5,
            title=f"{name} -- top 5 n-grams (n=2..5)",
        )
        figs["ngram_sweep"] = fig_ng

        results[name] = figs

    return results


def _plot_benchmark_boxplots(results: list) -> plt.Figure:
    """Box plot of per-level and aggregate scores across evaluation seeds."""
    level_names = [lr.level_name for lr in results[0].level_results]
    labels = level_names + ["aggregate"]

    level_scores = np.array(
        [[lr.weighted_score for lr in r.level_results] for r in results]
    )
    agg_scores = np.array([r.aggregate_score for r in results])
    all_scores = np.hstack([level_scores, agg_scores[:, None]])

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.6), 5))
    bp = ax.boxplot(
        all_scores,
        labels=labels,
        patch_artist=True,
        medianprops={"color": "crimson", "linewidth": 1.5},
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor("steelblue" if i < len(level_names) else "seagreen")
        patch.set_alpha(0.65)

    ax.set_ylabel("Score")
    ax.set_title(
        f"basic_skills score distribution ({len(results)} seeds)"
        f"  -- aggregate median = {float(np.median(agg_scores)):.3f}"
    )
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _train_run(
    config: Config,
    run_tag: str,
    train_levels: list[BenchmarkLevel],
    train_rfns: list[Callable],
    benchmark: BasicSkillsBenchmark,
    all_level_reward_fns: list[Callable],
    network: ActorCritic,
    optimizer: optax.GradientTransformation,
    obs_dim: int,
    rng: jax.Array,
) -> None:
    """Execute one training run: train, evaluate, render, and upload.

    Handles wandb lifecycle internally so it can be called once (mixed)
    or per-level (single).

    Args:
        config: Training configuration.
        run_tag: Short label used in wandb run name and output directory.
        train_levels: Levels to train on.
        train_rfns: Reward functions for training levels (same order).
        benchmark: Full benchmark used for evaluation.
        all_level_reward_fns: Reward functions for all benchmark levels.
        network: Actor-critic network (architecture only, params are freshly
            initialised each call).
        optimizer: Optax optimizer.
        obs_dim: Observation vector length.
        rng: PRNG key.
    """
    from factoriax.benchmarks.single_agent_mining.analysis import (
        render_level_video,
        save_mp4,
    )

    ppo = config.ppo

    # wandb init.
    wandb_run = None
    if ppo.use_wandb:
        try:
            import wandb  # type: ignore[import-untyped]

            run_name = ppo.wandb_run_name or run_tag
            wandb_run = wandb.init(
                project=ppo.wandb_project,
                name=run_name,
                config=dataclasses.asdict(config),
                tags=[
                    "basic_skills",
                    config.mode,
                    f"obs_local_r{ppo.obs_radius}",
                ],
            )
        except ImportError:
            logger.error("wandb not found. Install with: uv add wandb")

    env = FactoriaXEnv()

    # Fresh network parameters and optimizer state per run.
    rng, key_init = jax.random.split(rng)
    params = network.init(key_init, jnp.zeros(obs_dim))
    opt_state = optimizer.init(params)
    obs_stats = init_running_stats(obs_dim)
    update_fn = make_update_fn(network, optimizer, ppo)

    logger.info(
        "[%s] Starting training: %d total steps  %d levels  %d envs/level",
        run_tag,
        ppo.total_steps,
        len(train_levels),
        ppo.num_envs,
    )

    rng, key_train = jax.random.split(rng)
    params, obs_stats = _run_mixed(
        levels=train_levels,
        level_reward_fns=train_rfns,
        env=env,
        network=network,
        update_fn=update_fn,
        params=params,
        obs_stats=obs_stats,
        opt_state=opt_state,
        obs_dim=obs_dim,
        config=config,
        rng=key_train,
        wandb_run=wandb_run,
    )

    # Save checkpoint.
    out_dir = Path(ppo.save_path or ".") / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "final.pkl"
    data = {
        "params": jax.device_get(jax.tree_util.tree_map(np.array, params)),
        "obs_mean": np.array(obs_stats.mean),
        "obs_var": np.array(obs_stats.var),
        "obs_count": int(obs_stats.count),
        "config": dataclasses.asdict(config),
    }
    with open(ckpt_path, "wb") as f:
        pickle.dump(data, f)
    logger.info("[%s] Checkpoint saved -> %s", run_tag, ckpt_path)

    # Build a pure policy function for batched evaluation.
    # Closes over trained params and obs stats so the signature is
    # (obs, rng_key) -> action, compatible with vmap + scan.
    _params = params
    _obs_stats = obs_stats
    _normalize = ppo.normalize_obs

    def _pure_policy(obs: jax.Array, key: jax.Array) -> jax.Array:
        """Stochastic policy: normalize, forward pass, sample."""
        norm = normalize_obs(_obs_stats, obs) if _normalize else obs
        logits, _ = network.apply(_params, norm)
        return jax.random.categorical(key, logits)

    def eval_obs_fn(
        state: EnvState, env_params: EnvParams, player_idx: int
    ) -> jax.Array:
        """Extract local observations for the benchmark runner."""
        return local_array(state, env_params, player_idx, ppo.obs_radius)

    # Evaluate on the full benchmark (all seeds in parallel).
    logger.info(
        "[%s] Evaluating (%d seeds, batched)...",
        run_tag,
        config.eval_seeds,
    )
    runner = BenchmarkRunner()
    eval_results = runner.run_batched(
        benchmark,
        policy_fn=_pure_policy,
        seeds=list(range(config.eval_seeds)),
        obs_fn=eval_obs_fn,
    )

    # _make_policy for trajectory saving / video rendering (seed 0).
    jit_apply = jax.jit(network.apply)

    def _make_policy(
        seed: int,
    ) -> Callable[[jax.Array], jax.Array]:
        """Build a stochastic policy with a per-seed PRNG."""
        state_holder = {"rng": jax.random.PRNGKey(seed + 1000)}

        def policy(obs: jax.Array) -> jax.Array:
            state_holder["rng"], key = jax.random.split(state_holder["rng"])
            norm = normalize_obs(_obs_stats, obs) if _normalize else obs
            logits, _ = jit_apply(_params, norm)
            return jax.random.categorical(key, logits)

        return policy

    level_names = [lr.level_name for lr in eval_results[0].level_results]
    level_scores = np.array(
        [[lr.weighted_score for lr in r.level_results] for r in eval_results]
    )
    agg_scores = np.array([r.aggregate_score for r in eval_results])

    # Print results.
    print(f"\n[{run_tag}] Benchmark: basic_skills  ({config.eval_seeds} seeds)")
    print(
        f"Aggregate: {np.mean(agg_scores):.3f}"
        f" +/- {np.std(agg_scores):.3f}"
        f"  (median {np.median(agg_scores):.3f})"
    )
    print()
    for j, name in enumerate(level_names):
        s = level_scores[:, j]
        print(
            f"  {name:<28}  mean={np.mean(s):7.2f}"
            f" +/- {np.std(s):5.2f}"
            f"  median={np.median(s):7.2f}"
            f"  [{np.min(s):.0f}, {np.max(s):.0f}]"
        )

    # Save trajectories and render videos for each level.
    traj_paths = _save_eval_trajectories(
        benchmark,
        _make_policy(0),
        eval_obs_fn,
        all_level_reward_fns,
        config,
        out_dir,
    )
    video_paths: dict[str, Path] = {}
    for bl in benchmark.levels():
        frames = render_level_video(
            bl,
            _make_policy(0),
            seed=ppo.seed,
            obs_fn=eval_obs_fn,
        )
        mp4_path = out_dir / f"{bl.name}.mp4"
        save_mp4(frames, mp4_path)
        video_paths[bl.name] = mp4_path
        logger.info(
            "[%s] Rendered video: %s (%d frames)",
            run_tag,
            mp4_path,
            len(frames),
        )

    # Generate diagnostic plots (rewards + n-gram sweeps) per level.
    diag_figs = _plot_eval_diagnostics(traj_paths)
    for level_name, figs in diag_figs.items():
        for plot_name, fig in figs.items():
            fig_path = out_dir / f"{level_name}_{plot_name}.png"
            fig.savefig(fig_path, dpi=150)
            logger.info("[%s] Saved %s: %s", run_tag, plot_name, fig_path)

    # Upload everything to wandb.
    if wandb_run is not None:
        try:
            import wandb  # type: ignore[import-untyped]

            log_data: dict[str, object] = {
                "benchmark/aggregate_mean": float(np.mean(agg_scores)),
                "benchmark/aggregate_median": float(np.median(agg_scores)),
                "benchmark/aggregate_std": float(np.std(agg_scores)),
            }
            for j, name in enumerate(level_names):
                log_data[f"benchmark/{name}/mean"] = float(np.mean(level_scores[:, j]))
                log_data[f"benchmark/{name}/std"] = float(np.std(level_scores[:, j]))
            fig_box = _plot_benchmark_boxplots(eval_results)
            log_data["benchmark/score_distribution"] = wandb.Image(fig_box)
            plt.close(fig_box)

            for name, mp4_path in video_paths.items():
                log_data[f"videos/{name}"] = wandb.Video(
                    str(mp4_path),
                    fps=10,
                    format="mp4",
                )

            for level_name, figs in diag_figs.items():
                for plot_name, fig in figs.items():
                    log_data[f"{plot_name}/{level_name}"] = wandb.Image(fig)
                    plt.close(fig)

            wandb_run.log(log_data)

            # Upload trajectories as an artifact.
            artifact = wandb.Artifact(
                f"trajectories-{run_tag}",
                type="trajectory",
            )
            for name, traj_path in traj_paths.items():
                artifact.add_file(str(traj_path), name=f"{name}.npz")
            wandb_run.log_artifact(artifact)

            wandb_run.finish()
        except ImportError:
            pass
    else:
        for figs in diag_figs.values():
            for fig in figs.values():
                plt.close(fig)


def train(config: Config) -> None:
    """Run PPO training and evaluate on the basic_skills benchmark.

    In ``"mixed"`` mode, trains one shared policy on all levels. In
    ``"single"`` mode, trains a separate policy on each level.
    """
    ppo = config.ppo
    rng = jax.random.PRNGKey(ppo.seed)

    benchmark = BasicSkillsBenchmark()
    all_levels = benchmark.levels()
    all_level_reward_fns = [REWARD_FNS[bl.name] for bl in all_levels]

    # Derive obs_dim from a sample build.
    _sample_state = build_state(all_levels[0].level, all_levels[0].env_params)
    obs_dim = int(
        local_array(
            _sample_state,
            all_levels[0].env_params,
            0,
            ppo.obs_radius,
        ).shape[0]
    )
    logger.info(
        "obs_dim=%d (local obs, radius=%d)  num_actions=%d  mode=%s",
        obs_dim,
        ppo.obs_radius,
        NUM_ACTIONS,
        config.mode,
    )

    network = ActorCritic(hidden_dims=ppo.hidden_dims, num_actions=NUM_ACTIONS)
    optimizer = optax.chain(
        optax.clip_by_global_norm(ppo.max_grad_norm),
        optax.adam(ppo.learning_rate),
    )

    if config.mode == "single":
        for bl, rfn in zip(all_levels, all_level_reward_fns):
            rng, key_run = jax.random.split(rng)
            _train_run(
                config,
                run_tag=bl.name,
                train_levels=[bl],
                train_rfns=[rfn],
                benchmark=benchmark,
                all_level_reward_fns=all_level_reward_fns,
                network=network,
                optimizer=optimizer,
                obs_dim=obs_dim,
                rng=key_run,
            )
    else:
        rng, key_run = jax.random.split(rng)
        _train_run(
            config,
            run_tag="mixed",
            train_levels=all_levels,
            train_rfns=all_level_reward_fns,
            benchmark=benchmark,
            all_level_reward_fns=all_level_reward_fns,
            network=network,
            optimizer=optimizer,
            obs_dim=obs_dim,
            rng=key_run,
        )


# ---------------------------------------------------------------------------
# Trajectory saving for debugger
# ---------------------------------------------------------------------------


def _save_eval_trajectories(
    benchmark: BasicSkillsBenchmark,
    policy: Callable[[jax.Array], jax.Array],
    obs_fn: Callable,
    level_reward_fns: list[Callable],
    config: Config,
    out_dir: Path,
) -> dict[str, Path]:
    """Re-run the policy on each level and save full-state trajectories.

    Each trajectory is saved as a ``.npz`` file that the debugger can
    load directly (with full state data for game world rendering).

    Args:
        benchmark: The basic_skills benchmark.
        policy: Trained policy function.
        obs_fn: Observation extraction function.
        level_reward_fns: Reward function per level.
        config: Training configuration.
        out_dir: Directory to write trajectory files into.

    Returns:
        Mapping from level name to the saved ``.npz`` path.
    """
    from factoriax.analysis.trajectory import states_to_trajectory
    from factoriax.envs import FactoriaXEnv

    ppo = config.ppo
    env = FactoriaXEnv()
    jit_step = jax.jit(env.step_env)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    for bl, rfn in zip(benchmark.levels(), level_reward_fns):
        params_l = bl.env_params
        state = build_state(bl.level, params_l)
        rng = jax.random.PRNGKey(ppo.seed)

        states = [state]
        actions_log: list[int] = []
        rewards_log: list[float] = []

        for _ in range(params_l.max_timesteps):
            obs = obs_fn(state, params_l, 0)
            action = policy(obs)
            rng, subkey = jax.random.split(rng)
            prev_state = state
            _, state, _, done, _ = jit_step(subkey, state, action, params_l)
            reward = float(rfn(prev_state, state, params_l))
            actions_log.append(int(action))
            rewards_log.append(reward)
            states.append(state)
            if bool(done):
                break

        # Pad actions/rewards to match states length.
        act = np.array(
            actions_log + [0] * (len(states) - len(actions_log)),
            dtype=np.int32,
        )
        rew = np.array(
            rewards_log + [0.0] * (len(states) - len(rewards_log)),
            dtype=np.float32,
        )
        traj = states_to_trajectory(states, actions=act, rewards=rew)
        traj = replace(
            traj,
            observation_scheme={
                "type": 2,  # LOCAL
                "radius": ppo.obs_radius,
            },
        )
        path = out_dir / f"{bl.name}_trajectory.npz"
        traj.save(str(path))
        paths[bl.name] = path
        logger.info(
            "Saved trajectory: %s  (%d steps, reward=%.1f)",
            path,
            len(actions_log),
            sum(rewards_log),
        )

    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> Config:
    """Parse command-line arguments into a Config."""
    p = argparse.ArgumentParser(
        description="PPO training on the basic_skills benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_ppo_args(p)
    p.add_argument("--eval-seeds", type=int, default=10)
    p.add_argument(
        "--mode",
        type=str,
        default="mixed",
        choices=["single", "mixed"],
        help=("'single' trains one policy per level; 'mixed' trains one shared policy"),
    )

    args = p.parse_args()
    return Config(
        ppo=ppo_config_from_args(args, wandb_project="factoriax-basic-skills"),
        eval_seeds=args.eval_seeds,
        mode=args.mode,
    )


if __name__ == "__main__":
    train(_parse_args())
