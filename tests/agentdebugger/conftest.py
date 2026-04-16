"""Shared fixtures for agentdebugger tests.

Provides a lightweight Debugger instance that can be used without
launching a pygame window.
"""

import jax
import jax.numpy as jnp
import pygame
import pytest

from factoriax.agentdebugger.main import Debugger
from factoriax.constants import Action
from factoriax.envs.factoriax_env import FactoriaXEnv, make_factoriax_env
from factoriax.observations import global_array
from factoriax.state import EnvParams


@pytest.fixture(scope="session", autouse=True)
def pygame_font_session() -> None:
    """Initialise pygame font subsystem once per session."""
    pygame.font.init()


@pytest.fixture(scope="module")
def env_and_state() -> tuple[FactoriaXEnv, EnvParams, object]:
    """Create a small environment and initial state for testing.

    Module-scoped to avoid repeated JIT compilation.

    Returns:
        Tuple of (env, params, initial_state).
    """
    env, _ = make_factoriax_env()
    params = EnvParams(map_width=8, map_height=8, num_players=1)
    _, state = env.reset_env(jax.random.PRNGKey(0), params)
    return env, params, state


@pytest.fixture
def debugger(env_and_state: tuple) -> Debugger:
    """Create a Debugger with a simple deterministic policy."""
    env, params, state = env_and_state

    def noop_policy(obs: jax.Array) -> jax.Array:
        return jnp.int32(Action.NOOP)

    def dummy_reward(
        prev: object,
        new: object,
        p: object,
    ) -> jax.Array:
        return jnp.float32(1.0)

    def dummy_constraint(
        prev: object,
        new: object,
        p: object,
    ) -> jax.Array:
        return jnp.array([0.5, 0.1], dtype=jnp.float32)

    return Debugger(
        env,
        params,
        state,
        policy=noop_policy,
        obs_fn=global_array,
        reward_fn=dummy_reward,
        constraint_fn=dummy_constraint,
        constraint_names=["cost_a", "cost_b"],
        seed=0,
    )
