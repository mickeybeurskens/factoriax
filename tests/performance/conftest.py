"""Shared fixtures for performance tests."""

from __future__ import annotations

import time
from pathlib import Path

import jax
import pytest

from factoriax.envs.factoriax_env import FactoriaXEnv
from factoriax.jax_renderer import JaxRenderer
from factoriax.state import EnvParams

MAP_SIZE = 16
TILE_PX = 8
NUM_STEPS = 100


@pytest.fixture(scope="session")
def perf_params() -> EnvParams:
    """Environment params shared across all performance tests."""
    return EnvParams(
        map_width=MAP_SIZE,
        map_height=MAP_SIZE,
        num_players=1,
        max_timesteps=NUM_STEPS,
    )


@pytest.fixture(scope="session")
def renderer() -> JaxRenderer:
    """Shared JaxRenderer instance."""
    return JaxRenderer(tile_px=TILE_PX)


@pytest.fixture(scope="session")
def perf_log_dir() -> Path:
    """Create a timestamped log directory for this test session."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = Path("tests/performance/logs") / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def make_batched_envs(
    n: int, params: EnvParams
) -> tuple[jax.Array, object]:
    """Create N parallel environment states via vmapped reset.

    Args:
        n: Number of parallel environments.
        params: Environment parameters.

    Returns:
        Tuple of (rng_keys, batched_states).
    """
    env = FactoriaXEnv()
    keys = jax.random.split(jax.random.key(42), n)
    _, states = jax.vmap(env.reset_env, in_axes=(0, None))(keys, params)
    return keys, states
