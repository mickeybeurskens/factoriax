"""Plug a custom reward function into a step loop.

Rewards in FactoriaX are pure functions of ``(prev_state, new_state,
params) → float``. This example writes one that scores ore-mining
events and runs it alongside the engine in a manual rollout. Use the
same pattern with any wrapper-stack a research project needs.

Run::

    uv run python examples/custom_reward.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import factoriax
from factoriax import EnvState, ItemType


def ore_delta_reward(prev: EnvState, new: EnvState, _params: object) -> jax.Array:
    """+1 per ore item mined this step (iron + copper + coal).

    Engine state stores cumulative ``items_mined[i]`` per item type;
    the per-step delta is the reward signal. Pure function, JIT-safe,
    composes inside ``jax.lax.scan`` rollouts.
    """
    ore_types = jnp.array(
        [int(ItemType.IRON_ORE), int(ItemType.COPPER_ORE), int(ItemType.COAL)],
    )
    delta = new.items_mined[ore_types] - prev.items_mined[ore_types]
    out: jax.Array = jnp.float32(delta.sum())
    return out


def main(steps: int = 256) -> dict[str, float]:
    """Roll out with the custom reward and return summary stats.

    Args:
        steps: Number of environment steps.

    Returns:
        ``{"total_reward": float, "ore_mined": float}``.
    """
    env, params = factoriax.make()
    rng = jax.random.PRNGKey(0)
    rng, key_reset = jax.random.split(rng)
    _, state = env.reset_env(key_reset, params)

    total = jnp.float32(0.0)
    n_actions = int(env.action_space(params).n)
    for _ in range(steps):
        rng, key_act, key_step = jax.random.split(rng, 3)
        action = jax.random.randint(key_act, (), 0, n_actions)
        prev = state
        _, state, _, _, _ = env.step_env(key_step, state, action, params)
        total = total + ore_delta_reward(prev, state, params)

    mined = float(state.items_mined.sum())
    print(f"steps={steps}  ore_reward={float(total):.0f}  total_mined={mined:.0f}")
    return {"total_reward": float(total), "ore_mined": mined}


if __name__ == "__main__":
    main(steps=2_000)
