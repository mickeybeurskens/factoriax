"""Debug script for PPO training on Mining-v1.

Extracted from docs/training/train_ppo_mining.ipynb.

Bug: FactoriaxEnv.step() overrides gymnax auto-reset, so LogWrapper never
resets the env after episode end. Fix: wrap with AutoResetWrapper(resample=True)
before LogWrapper so each terminal step triggers a fresh episode.
"""

import gc
import time

import chex
import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from gymnax.environments import environment, spaces
from typing import Optional, Tuple, Union, Sequence, NamedTuple, Any

from factoriax.engine.constants import Action, BlockType, NUM_ACTIONS
from factoriax.engine.envs.wrappers import AutoResetWrapper
from factoriax.make import env_from_name

# ---------------------------------------------------------------------------
# Wrappers (from PureJaxRL/wrappers.py)
# ---------------------------------------------------------------------------

class GymnaxWrapper(object):
    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)


@struct.dataclass
class LogEnvState:
    env_state: environment.EnvState
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int


class LogWrapper(GymnaxWrapper):
    def __init__(self, env: environment.Environment):
        super().__init__(env)

    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, env_state = self._env.reset_env(key, params)
        state = LogEnvState(env_state, 0, 0, 0, 0, 0)
        return obs, state

    def step(
        self,
        key: chex.PRNGKey,
        state: LogEnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step_env(
            key, state.env_state, action, params
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        log_state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = log_state.returned_episode_returns
        info["returned_episode_lengths"] = log_state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return obs, log_state, reward, done, info


# ---------------------------------------------------------------------------
# Network (from PureJaxRL/ppo.py)
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"
    hidden_size: int = 256  # 256 vs 64: more capacity for 85-action space over 255-dim obs

    @nn.compact
    def __call__(self, x):
        activation = nn.relu if self.activation == "relu" else nn.tanh
        actor_mean = nn.Dense(self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        critic = activation(critic)
        critic = nn.Dense(self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


# ---------------------------------------------------------------------------
# Training (from PureJaxRL/ppo.py, with auto-reset fix)
# ---------------------------------------------------------------------------

def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    base_env, env_params = config["ENV"], config["ENV_PARAMS"]

    # FIX: wrap with AutoResetWrapper so episodes reset automatically inside lax.scan.
    # Without this, FactoriaxEnv.step_env never resets after done=True and all
    # subsequent rollouts collect zero reward.
    #
    # resample=True: each done triggers a fresh random map (hard, requires generalisation)
    # resample=False: each done restores the same initial map per env (agents memorise per-env layout)
    resample = config.get("RESAMPLE", False)
    env = AutoResetWrapper(base_env, resample=resample)
    env = LogWrapper(env)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        network = ActorCritic(env.action_space(env_params).n, activation=config["ACTIVATION"])
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        def _update_step(runner_state, unused):
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)
                transition = Transition(done, action, value, reward, log_prob, last_obs, info)
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            train_state, env_state, last_obs, rng = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = transition.done, transition.value, transition.reward
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                        entropy = pi.entropy().mean()

                        total_loss = loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(train_state.params, traj_batch, advantages, targets)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
                shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(_update_minbatch, train_state, minibatches)
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def check_obs_intermediates(env, params):
    """Print a breakdown of each observation component for sanity checking."""
    from factoriax.engine.observations import _MAP_NORM, BLOCK_MAX_RESOURCES
    from factoriax.engine.constants import BlockType

    obs, state = env.reset_env(jax.random.PRNGKey(42), params)

    print("=== Observation intermediates ===")
    print(f"obs shape: {obs.shape}")
    # superficial_global layout: [block_type(64), machine_type(64), machine_dir(64), scalars(63)]
    n_tiles = env.map_width * env.map_height  # 64
    blocks   = np.array(obs[:n_tiles])
    machines = np.array(obs[n_tiles:2*n_tiles])
    mach_dir = np.array(obs[2*n_tiles:3*n_tiles])
    scalars  = np.array(obs[3*n_tiles:])

    iron_val = int(BlockType.IRON) / _MAP_NORM
    iron_mask = np.abs(blocks - iron_val) < 1e-3
    print(f"  block_type channel: min={blocks.min():.3f} max={blocks.max():.3f}")
    print(f"    IRON value={iron_val:.3f}  DIRT value={int(BlockType.DIRT)/_MAP_NORM:.3f}")
    print(f"    Iron tiles visible in obs: {iron_mask.sum()} (map has {int(np.sum(np.array(state.map)==int(BlockType.IRON)))})")
    print(f"  machine_type channel: all-zero={bool((machines==0).all())}  (expected: no machines)")
    print(f"  machine_dir channel:  all-zero={bool((mach_dir==0).all())}  (expected: no machines)")
    print(f"  scalars: {len(scalars)} features")
    px = float(scalars[0]) * env.map_width
    py = float(scalars[1]) * env.map_height
    direction = float(scalars[2]) * 4
    timestep  = float(scalars[3]) * params.max_timesteps
    print(f"    pos_x={px:.1f} pos_y={py:.1f} direction={direction:.0f} timestep={timestep:.0f}")
    print(f"    inventory[0:5]={np.array(state.player_inventory[0,:5])}")
    print()
    print(f"  BLOCK_MAX_RESOURCES={BLOCK_MAX_RESOURCES} (used in x_ray obs)")
    print(f"  Mining-v1 base_resources={params.base_resources}")
    print(f"  → x_ray block_resources channel would be {params.base_resources}/{BLOCK_MAX_RESOURCES} = {params.base_resources/BLOCK_MAX_RESOURCES:.6f} (near-zero, effectively unusable)")
    print(f"  → superficial_global (used here) skips block_resources entirely — correct choice for Mining-v1")
    print()


def eval_policy(trained_params, env, params, activation, n_episodes=20):
    """Evaluate trained policy, both stochastic and greedy, on fresh random maps."""
    from factoriax.engine.constants import Action
    network = ActorCritic(env.action_space(params).n, activation=activation)
    apply_fn = jax.jit(network.apply)
    step_fn = jax.jit(env.step_env)

    def _run_episodes(use_greedy):
        returns = []
        action_counts = np.zeros(env.action_space(params).n, dtype=int)
        for ep in range(n_episodes):
            ep_key = jax.random.PRNGKey(2000 + ep)
            obs, state = env.reset_env(ep_key, params)
            ep_return = 0.0
            for _ in range(params.max_timesteps):
                pi, _ = apply_fn(trained_params, obs)
                if use_greedy:
                    action = int(pi.mode())
                else:
                    ep_key, sk = jax.random.split(ep_key)
                    action = int(pi.sample(seed=sk))
                action_counts[action] += 1
                ep_key, sk2 = jax.random.split(ep_key)
                obs, state, reward, done, _ = step_fn(sk2, state, action, params)
                ep_return += float(reward)
                if done:
                    break
            returns.append(ep_return)
        return np.mean(returns), np.std(returns), action_counts

    def _top_actions(action_counts, label):
        top = np.argsort(action_counts)[::-1][:6]
        print(f"  Top actions ({label}): ", end="")
        for a in top:
            if action_counts[a] > 0:
                try:
                    name = Action(a).name
                except ValueError:
                    name = str(a)
                print(f"{name}={action_counts[a]}", end=" ")
        print()

    stoch_mean, stoch_std, stoch_counts = _run_episodes(use_greedy=False)
    _top_actions(stoch_counts, "stochastic")
    greedy_mean, greedy_std, greedy_counts = _run_episodes(use_greedy=True)
    _top_actions(greedy_counts, "greedy   ")
    return stoch_mean, stoch_std, greedy_mean, greedy_std


def main():
    env, params = env_from_name("Mining-v1")
    params = params.replace(base_resources=10)

    print("=== Environment info ===")
    obs, _ = env.reset_env(jax.random.PRNGKey(0), params)
    print(f"obs shape    : {obs.shape}")
    print(f"action count : {env.action_space(params).n}")
    print(f"max_timesteps: {params.max_timesteps}")
    print(f"max possible : 100  (10 tiles × 10 resources)")
    print()
    check_obs_intermediates(env, params)

    # Random baseline
    def rollout(key):
        def scan_body(carry, _):
            key, state = carry
            key, act_key, step_key = jax.random.split(key, 3)
            action = jax.random.randint(act_key, (), 0, NUM_ACTIONS)
            obs, state, reward, done, _ = env.step_env(step_key, state, action, params)
            return (key, state), reward
        obs, state = env.reset_env(key, params)
        (_, _), rewards = jax.lax.scan(scan_body, (key, state), None, length=params.max_timesteps)
        return rewards.sum()

    collect = jax.jit(jax.vmap(rollout))
    N = 512
    keys = jax.random.split(jax.random.PRNGKey(1), N)
    random_returns = np.asarray(collect(keys))
    print(f"\nRandom baseline ({N} eps): {random_returns.mean():.2f} ± {random_returns.std():.2f}")

    # Hyperparameter changes vs original notebook:
    # - ENT_COEF: kept at 0.01 (0.05 causes entropy collapse to always-MINE policy)
    # - TOTAL_TIMESTEPS 1M → 10M: discovering move→face→mine across 85 actions needs more samples
    # - NUM_ENVS 64 → 128: more parallel rollouts per update step
    # - Network: 256 hidden units instead of 64 (more capacity for 255-dim obs → 85 actions)
    config = {
        "LR": 2.5e-4,
        "NUM_ENVS": 128,
        "NUM_STEPS": params.max_timesteps,
        "TOTAL_TIMESTEPS": 30_000_000,
        "UPDATE_EPOCHS": 8,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ANNEAL_LR": True,
        "RESAMPLE": True,
        "ENV": env,
        "ENV_PARAMS": params,
    }

    num_updates = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    print(f"\n=== Training ===")
    print(f"NUM_UPDATES : {num_updates}")
    print(f"NUM_ENVS    : {config['NUM_ENVS']}")
    print(f"NUM_STEPS   : {config['NUM_STEPS']}")
    print(f"ENT_COEF    : {config['ENT_COEF']}")
    print(f"TOTAL_STEPS : {config['TOTAL_TIMESTEPS']:,}")
    print(f"Network     : 256 hidden units")
    print(f"RESAMPLE    : {config['RESAMPLE']}  (False=memorise per-env map, True=new map each episode)")
    print(f"base_resources: {params.base_resources}  (max score={10*params.base_resources})")
    print("Compiling and running (first run takes ~30s to compile)...")

    train_fn = jax.jit(make_train(config))
    t0 = time.perf_counter()
    out = train_fn(jax.random.PRNGKey(42))
    jax.block_until_ready(out)
    elapsed = time.perf_counter() - t0
    steps_per_sec = config["TOTAL_TIMESTEPS"] / elapsed
    print(f"Training complete in {elapsed:.1f}s  ({steps_per_sec:,.0f} steps/s)")

    trained_params = out["runner_state"][0].params
    metrics = out["metrics"]

    ep_returns = metrics["returned_episode_returns"]  # (num_updates, num_steps, num_envs)
    ep_done    = metrics["returned_episode"]

    mean_per_update = np.array([
        float(ep_returns[i][ep_done[i]].mean()) if ep_done[i].any() else float("nan")
        for i in range(ep_returns.shape[0])
    ])

    # Show learning curve at ~20 evenly spaced points
    step = max(1, len(mean_per_update) // 20)
    print(f"\n=== Learning curve ===")
    print(f"  {'update':>6}  {'mean_return':>12}")
    for i in range(0, len(mean_per_update), step):
        print(f"  {i:6d}  {mean_per_update[i]:12.2f}")
    print(f"  {len(mean_per_update)-1:6d}  {mean_per_update[-1]:12.2f}  ← final")

    # Smooth over last 10% of training to see settled performance
    tail = mean_per_update[int(0.9 * len(mean_per_update)):]
    tail_clean = tail[~np.isnan(tail)]
    print(f"\n  Last 10% updates: mean={tail_clean.mean():.2f} ± {tail_clean.std():.2f}")

    print(f"\n=== Eval on fresh random maps (20 episodes) ===")
    stoch_mean, stoch_std, greedy_mean, greedy_std = eval_policy(trained_params, env, params, config["ACTIVATION"])
    print(f"  Stochastic policy: {stoch_mean:.2f} ± {stoch_std:.2f}")
    print(f"  Greedy policy    : {greedy_mean:.2f} ± {greedy_std:.2f}")
    print(f"  Random baseline  : {random_returns.mean():.2f} ± {random_returns.std():.2f}")
    print(f"  Max possible     : {10 * params.base_resources:.0f}  (10 tiles × {params.base_resources} resources)")

    del out, train_fn
    gc.collect()
    jax.clear_caches()

    print(f"\n=== Recording episode GIF ===")
    _record_episode_gif(trained_params, env, params, config["ACTIVATION"])


def _record_episode_gif(trained_params, env, params, activation, path="scripts/ppo_episode.gif"):
    """Render one greedy episode to an animated GIF."""
    import imageio.v3 as iio
    from factoriax.engine.jax_renderer import JaxRenderer

    network = ActorCritic(env.action_space(params).n, activation=activation)
    apply_fn = jax.jit(network.apply)
    step_fn  = jax.jit(env.step_env)
    renderer = JaxRenderer(tile_px=32)

    key = jax.random.PRNGKey(2042)
    obs, state = env.reset_env(key, params)
    frames = [np.asarray(renderer.jit_render_map(state))]
    ep_return = 0.0

    for _ in range(params.max_timesteps):
        pi, _ = apply_fn(trained_params, obs)
        key, sk = jax.random.split(key)
        action = int(pi.sample(seed=sk))
        key, sk2 = jax.random.split(key)
        obs, state, reward, done, _ = step_fn(sk2, state, action, params)
        frames.append(np.asarray(renderer.jit_render_map(state)))
        ep_return += float(reward)
        if done:
            break

    iio.imwrite(path, frames, extension=".gif", duration=80, loop=0)
    print(f"  Saved {len(frames)}-frame GIF to {path}  (episode return: {ep_return:.0f})")


if __name__ == "__main__":
    main()
