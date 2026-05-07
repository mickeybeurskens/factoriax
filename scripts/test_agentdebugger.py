"""Smoke test for the agent debugger.

Launches the debugger on a small random map with a random policy.
Use this to verify the window renders, stepping works, and charts
update.

Controls:
    N         Step forward (AI picks a random action)
    Tab       Switch to human mode (then N to enter input mode)
    [ / ]     Navigate history
    ?         Help overlay
    Esc       Quit
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random

import factoriax
from factoriax.agentdebugger import Debugger
from factoriax.constants import Action
from factoriax.constraints import balance_cost, balance_cost_names
from factoriax.observations import global_array
from factoriax.rewards import achievement_reward
from factoriax.state import EnvParams


def random_policy(obs: jax.Array) -> jax.Array:
    """Policy that picks a uniformly random action each step."""
    num_actions = len(Action)
    return jnp.int32(
        int(
            random.randint(
                random.PRNGKey(hash(obs.sum().item()) % (2**31)),
                shape=(),
                minval=0,
                maxval=num_actions,
            )
        )
    )


def main() -> None:
    """Set up a small environment and launch the debugger."""
    env, _ = factoriax.make()
    params = EnvParams(map_width=16, map_height=16, num_players=1)

    rng = random.PRNGKey(0)
    _, state = env.reset_env(rng, params)

    debugger = Debugger(
        env,
        params,
        state,
        policy=random_policy,
        obs_fn=global_array,
        reward_fn=achievement_reward,
        constraint_fn=balance_cost,
        constraint_names=balance_cost_names(),
        seed=42,
    )
    debugger.run()


if __name__ == "__main__":
    main()
