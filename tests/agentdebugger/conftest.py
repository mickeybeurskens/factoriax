"""Shared fixtures for agentdebugger tests.

Provides a lightweight Debugger instance that can be used without
launching a pygame window. The function-scoped ``debugger`` fixture
preserves a fresh instance per test (so per-test ``_states`` /
``_actions`` / etc. accumulators stay clean), but the JITted step
function — the expensive piece of ``Debugger.__init__`` at
``factoriax/agentdebugger/main.py:106`` — is shared at module scope
via ``_cached_step_fn``. Without the swap, every test in
``test_stepping.py`` paid a fresh ~7s XLA compile; with it, the
compile happens once for the file.
"""

import jax
import jax.numpy as jnp
import pygame
import pytest

import factoriax
from factoriax.agentdebugger.main import Debugger
from factoriax.constants import Action
from factoriax.envs.factoriax_env import FactoriaXEnv
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
    env, _ = factoriax.make()
    params = EnvParams(map_width=8, map_height=8, num_players=1)
    _, state = env.reset_env(jax.random.PRNGKey(0), params)
    return env, params, state


def _warm_step_fn(env: FactoriaXEnv, params: EnvParams, state) -> object:
    """Build + warm ``jax.jit(env.step_env)`` against Debugger's call shape.

    ``Debugger._execute_step`` calls ``step_fn(rng, state.replace(
    selected_player=p), action, params)`` with ``action`` as a Python
    int. Warming with a different signature (e.g. ``jnp.int32(NOOP)``)
    populates a different cache slot and tests still pay a compile on
    their first call. Building a throwaway Debugger and invoking its
    ``_execute_step`` guarantees the warm-up matches every consumer.
    """
    temp = Debugger(env, params, state, policy=lambda obs: jnp.int32(Action.NOOP))
    temp._execute_step(int(Action.NOOP))  # noqa: SLF001
    return temp._step_fn  # noqa: SLF001


@pytest.fixture(scope="module")
def _cached_step_fn(env_and_state: tuple):
    """Pre-warmed ``jax.jit(env.step_env)`` shared across the module.

    Builds one wrapper and triggers its compile via the same
    Debugger call path consumers use, so subsequent tests see a cache
    hit. The function-scoped ``debugger`` fixture below swaps this in
    for each fresh Debugger's ``_step_fn``, so the XLA compile is paid
    once per module instead of once per test.
    """
    env, params, state = env_and_state
    return _warm_step_fn(env, params, state)


@pytest.fixture(scope="module")
def env_and_state_2p() -> tuple[FactoriaXEnv, EnvParams, object]:
    """Two-player variant of ``env_and_state`` for TestMultiPlayer."""
    env, _ = factoriax.make()
    params = EnvParams(map_width=8, map_height=8, num_players=2)
    _, state = env.reset_env(jax.random.PRNGKey(0), params)
    return env, params, state


@pytest.fixture(scope="module")
def _cached_step_fn_2p(env_and_state_2p: tuple):
    """Pre-warmed 2-player step_fn — same pattern as ``_cached_step_fn``."""
    env, params, state = env_and_state_2p
    return _warm_step_fn(env, params, state)


@pytest.fixture
def debugger(env_and_state: tuple, _cached_step_fn) -> Debugger:
    """Create a Debugger with a simple deterministic policy.

    Constructs a fresh Debugger per test (so mutable accumulators like
    ``_states`` and ``_actions`` start clean), then overwrites the
    Debugger's internally-built ``_step_fn`` with the module-scoped
    pre-warmed wrapper. The new Debugger drops its own freshly-created
    ``jax.jit`` wrapper for garbage collection.
    """
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

    dbg = Debugger(
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
    dbg._step_fn = _cached_step_fn  # noqa: SLF001 — surgical compile reuse
    return dbg
