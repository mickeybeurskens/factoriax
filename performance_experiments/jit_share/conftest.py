"""Session-scoped fixtures for the jit-share A/B harness.

The single fixture below is the load-bearing piece of Task 1.2's
``TestSharedSessionScope``: every test in the class consumes the same
``(env, params, jit_step_fn, initial_state)`` tuple, so the ten tests
together pay exactly one XLA compile instead of ten.

See ``SPEC_TEST_SUITE.md`` Phase 1.
"""

from __future__ import annotations

from typing import Any

import jax
import pytest
from jax import random

from factoriax.envs.factoriax_env import FactoriaXEnv
from factoriax.state import EnvParams


@pytest.fixture(scope="session")
def canonical_env_8x8_1p() -> tuple[FactoriaXEnv, EnvParams, Any, Any]:
    """Session-scoped 8x8 single-player env + JITted step + reset state.

    The shared fixture is the whole point of Class B — every consuming
    test rides on a single :func:`jax.jit` wrapper, so the XLA compile
    of ``env.step_env`` happens exactly once for the class.

    Returns:
        Tuple ``(env, params, jit_step_fn, initial_state)``. The state
        is the post-reset state at ``timestep == 0`` from
        ``random.PRNGKey(0)``; tests step *from* it without mutating
        it (JAX pytrees are immutable by construction).
    """
    env = FactoriaXEnv()
    params = EnvParams(map_width=8, map_height=8, num_players=1)
    _, initial_state = env.reset_env(random.PRNGKey(0), params)
    jit_step_fn = jax.jit(env.step_env)
    return env, params, jit_step_fn, initial_state
