"""Run a uniformly-random policy and report mean per-step reward.

The simplest possible "training loop": a no-op policy with no
gradient updates. Useful as a sanity check that the env, observation,
and reward-shaping plumbing all line up before plugging in a real
learner.

Run::

    uv run python examples/train_random_policy.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import factoriax
from factoriax import EnvParams, EnvState


def main(steps: int = 1000) -> dict[str, float]:
    """Roll out a random policy and return its summary stats.

    Args:
        steps: Number of environment steps to take.

    Returns:
        ``{"mean_reward": float, "total_reward": float}``. The reward
        is the engine's per-step shaping signal (``mining_reward``);
        it's zero-mean with random actions on the default level.
    """
    env, params = factoriax.make()
    params = _capped_horizon(params, steps)
    n_actions = int(env.action_space(params).n)

    rng = jax.random.PRNGKey(0)
    rng, key_reset = jax.random.split(rng)
    _, state = env.reset_env(key_reset, params)

    @jax.jit
    def step(
        rng: jax.Array, state: EnvState, action: jax.Array
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array]:
        rng, key = jax.random.split(rng)
        _, new_state, reward, done, _ = env.step_env(key, state, action, params)
        return rng, new_state, reward, done

    total = jnp.float32(0.0)
    for _ in range(steps):
        rng, key_act = jax.random.split(rng)
        action = jax.random.randint(key_act, (), 0, n_actions)
        rng, state, reward, done = step(rng, state, action)
        total = total + reward
        if bool(done):
            rng, key_reset = jax.random.split(rng)
            _, state = env.reset_env(key_reset, params)

    mean = float(total / steps)
    print(f"steps={steps}  total_reward={float(total):.3f}  mean={mean:.4f}")
    return {"mean_reward": mean, "total_reward": float(total)}


def _capped_horizon(params: EnvParams, steps: int) -> EnvParams:
    """Cap ``max_timesteps`` so the env reset-loop stays predictable."""
    return params.replace(max_timesteps=max(steps, params.max_timesteps))


if __name__ == "__main__":
    main()
