"""Evaluate a policy across many seeds in parallel via ``jax.vmap``.

Demonstrates the canonical batched-rollout pattern: vmap over a batch
of PRNG keys to produce many parallel environment trajectories on the
same compiled graph. Reports mean and std of cumulative reward.

Run::

    uv run python examples/batched_evaluation.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import factoriax
from factoriax import EnvState


def main(num_seeds: int = 16, steps: int = 64) -> dict[str, float]:
    """Run ``num_seeds`` parallel rollouts and return reward stats.

    Args:
        num_seeds: How many environments to roll out in parallel.
        steps: Steps per rollout. Kept small so the example stays
            fast even at high seed counts.

    Returns:
        ``{"mean_return": float, "std_return": float}`` over the seeds.
    """
    env, params = factoriax.make()
    n_actions = int(env.action_space(params).n)

    Carry = tuple[jax.Array, EnvState, jax.Array]  # noqa: N806

    def rollout(rng: jax.Array) -> jax.Array:
        rng, key_reset = jax.random.split(rng)
        _, state = env.reset_env(key_reset, params)

        def body(carry: Carry, _: None) -> tuple[Carry, None]:
            rng, state, total = carry
            rng, key_act, key_step = jax.random.split(rng, 3)
            action = jax.random.randint(key_act, (), 0, n_actions)
            _, state, reward, _, _ = env.step_env(key_step, state, action, params)
            return (rng, state, total + reward), None

        (_, _, total), _ = jax.lax.scan(
            body, (rng, state, jnp.float32(0.0)), xs=None, length=steps
        )
        out: jax.Array = total
        return out

    keys = jax.random.split(jax.random.PRNGKey(0), num_seeds)
    returns = jax.vmap(rollout)(keys)
    returns_np = jax.device_get(returns)
    mean = float(returns_np.mean())
    std = float(returns_np.std())
    print(f"seeds={num_seeds}  steps={steps}  mean_return={mean:.4f}  std={std:.4f}")
    return {"mean_return": mean, "std_return": std}


if __name__ == "__main__":
    main(num_seeds=32, steps=200)
