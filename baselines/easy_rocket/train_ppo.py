"""Train PPO on the easy rocket achievement scenario.

Builds :class:`~factoriax.envs.FactoriaXEnv` with the easy_rocket
scenario's :func:`easy_rocket_conditions` bound as
``achievement_fn`` and :func:`easy_rocket_reward` as the training
signal — Craftax-style sparse +weight on each newly-unlocked
achievement.

V1 trains on a single fixed layout sampled from
:func:`build_easy_rocket_level` with ``PRNGKey(seed)``, broadcast
across all parallel envs. Procgen-per-reset is a parked follow-up;
see ``docs/specs/2026_easy_rocket_ppo.md``.

Reuses the shared PPO infrastructure from ``baselines.ppo``. The
collect/GAE/update pipeline is fused into one JIT-compiled step to
keep GPU throughput high.

Usage::

    python -m baselines.easy_rocket.train_ppo
    python -m baselines.easy_rocket.train_ppo --num-envs 4 \\
        --rollout-steps 4 --total-steps 16
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

import factoriax
from baselines.ppo.cli import add_ppo_args, ppo_config_from_args
from baselines.ppo.config import PPOConfig
from baselines.ppo.gae import Transition, compute_gae
from baselines.ppo.network import ActorCritic
from baselines.ppo.normalization import (
    RunningStats,
    init_running_stats,
    normalize_obs,
    update_running_stats,
)
from factoriax.constants import MAX_ACHIEVEMENTS, NUM_ACTIONS, Action
from factoriax.levels import build_state
from factoriax.scenarios.easy_rocket import (
    EASY_ROCKET_RECIPE_TABLE,
    MAX_EASY_ROCKET_SCORE,
    NUM_EASY_ROCKET_ACHIEVEMENTS,
    build_easy_rocket_level,
    easy_rocket_conditions,
    easy_rocket_reward,
)
from factoriax.state import EnvParams, EnvState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Reachable per-episode achievement count. Four of the 13 easy_rocket
# achievements are belt-network stubs that always read False today, so
# the soft ceiling the policy can hit is 9. Surface both numbers in
# logs so a plateau at 9 reads as expected, not a bug.
REACHABLE_ACHIEVEMENTS: int = 9


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


def _make_env_and_state(config: Config) -> tuple[Any, EnvState, EnvParams]:
    """Build the wrapped env, initial EnvState, and EnvParams.

    The scenario sets ``blocked_actions = frozenset()`` — no action
    mask is applied. Observations are global (the 16x16 map flattens
    to a small enough vector that the local-window wrapper is not
    worth the extra indirection).

    ``EASY_ROCKET_RECIPE_TABLE`` must be wired into ``EnvParams``
    explicitly: the scenario uses an 8-recipe book, while
    ``factoriax.make`` returns ``EnvParams`` with the full
    ``DEFAULT_RECIPE_TABLE`` (~60 recipes). Without this override
    the env would offer a different (and substantially harder)
    recipe surface than the bench defines.
    """
    level = build_easy_rocket_level(jax.random.PRNGKey(config.ppo.seed))
    env, env_params = factoriax.make(
        level,
        obs="global",
        achievement_fn=easy_rocket_conditions,
    )
    env_params = env_params.replace(
        num_players=1,
        max_timesteps=config.max_timesteps,
        recipe_table=EASY_ROCKET_RECIPE_TABLE,
    )
    state0 = build_state(level, env_params)
    return env, state0, env_params


def train(config: Config) -> dict[str, float]:
    """Train PPO against the easy_rocket scenario and return final metrics."""
    env, initial_state, env_params = _make_env_and_state(config)
    initial_obs = env.get_obs(initial_state, env_params)
    obs_dim = int(initial_obs.shape[0])

    ppo = config.ppo
    logger.info(
        "Easy rocket PPO  obs_dim=%d  actions=%d  envs=%d  rollout=%d  total=%dk",
        obs_dim,
        NUM_ACTIONS,
        ppo.num_envs,
        ppo.rollout_steps,
        ppo.total_steps // 1000,
    )

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

    def _broadcast(x: jax.Array) -> jax.Array:
        a = jnp.asarray(x)
        return jnp.broadcast_to(a[None], (ppo.num_envs,) + a.shape)

    fixed_states: EnvState = jax.tree_util.tree_map(_broadcast, initial_state)

    vmap_step = jax.vmap(env.step_env, in_axes=(0, 0, 0, None))
    vmap_obs = jax.vmap(env.get_obs, in_axes=(0, None))
    vmap_reward = jax.vmap(easy_rocket_reward, in_axes=(0, 0, None))

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
            _, next_states, _env_rewards, dones, _ = vmap_step(
                keys, states, actions, env_params
            )

            rewards = vmap_reward(states, next_states, env_params)

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

    env_states = fixed_states
    obs = vmap_obs(fixed_states, env_params)
    ep_returns: deque[float] = deque(maxlen=500)
    ep_ach_counts: deque[int] = deque(maxlen=500)
    running_return = np.zeros(ppo.num_envs, dtype=np.float32)
    running_peak_mask = np.zeros((ppo.num_envs, MAX_ACHIEVEMENTS), dtype=bool)
    best_ach_count_ever = 0
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
        live_mask = np.asarray(env_states.achievements_unlocked)[
            :, :NUM_EASY_ROCKET_ACHIEVEMENTS
        ]
        running_peak_mask[:, :NUM_EASY_ROCKET_ACHIEVEMENTS] = np.maximum(
            running_peak_mask[:, :NUM_EASY_ROCKET_ACHIEVEMENTS], live_mask
        )

        for t in range(rewards_np.shape[0]):
            running_return += rewards_np[t]
            done_idx = np.where(dones_np[t])[0]
            for n in done_idx:
                ep_returns.append(float(running_return[n]))
                ach_n = int(running_peak_mask[n, :NUM_EASY_ROCKET_ACHIEVEMENTS].sum())
                ep_ach_counts.append(ach_n)
                best_ach_count_ever = max(best_ach_count_ever, ach_n)
                total_episodes += 1
                running_return[n] = 0.0
                running_peak_mask[n] = False

        if (it + 1) % ppo.log_interval == 0 or it == num_iters - 1:
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
                    "ach=%.2f/%d (ceil=%d)  best=%d  ent=%.3f  | %s"
                ),
                it + 1,
                num_iters,
                current_step // 1000,
                sps,
                mean_ret,
                mean_ach,
                NUM_EASY_ROCKET_ACHIEVEMENTS,
                REACHABLE_ACHIEVEMENTS,
                best_ach_count_ever,
                float(metrics["loss/entropy"]),
                act_str,
            )

    elapsed = time.time() - t_start
    mean_ret = float(np.mean(list(ep_returns))) if ep_returns else 0.0
    mean_ach = float(np.mean(list(ep_ach_counts))) if ep_ach_counts else 0.0
    logger.info(
        (
            "Done. %dk steps in %.1fs (%.0f sps). Final ret=%.2f  "
            "ach=%.2f/%d (ceil=%d)  best=%d  episodes=%d"
        ),
        current_step // 1000,
        elapsed,
        current_step / elapsed,
        mean_ret,
        mean_ach,
        NUM_EASY_ROCKET_ACHIEVEMENTS,
        REACHABLE_ACHIEVEMENTS,
        best_ach_count_ever,
        total_episodes,
    )

    return {
        "mean_ep_return": mean_ret,
        "mean_ep_achievements": mean_ach,
        "best_achievements_ever": float(best_ach_count_ever),
        "max_possible_score": float(MAX_EASY_ROCKET_SCORE),
        "reachable_achievements": float(REACHABLE_ACHIEVEMENTS),
        "sps": current_step / elapsed,
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
    args = parser.parse_args()

    ppo = ppo_config_from_args(args)
    config = Config(
        ppo=ppo,
        max_timesteps=args.max_timesteps,
        anneal_lr=not args.no_anneal_lr,
    )
    train(config)


if __name__ == "__main__":
    main()
