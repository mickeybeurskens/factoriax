"""Regression guard: JIT must not retrace after warmup.

The step function is JIT-compiled once during warmup. Subsequent
calls with the post-step state and different action values must
reuse the cached trace. A retrace causes a ~5s freeze in gameplay.

This test catches dtype mismatches between reset state and post-step
state (e.g. Python int vs jnp.int32, int16 promoted to int32) that
would invalidate the JIT cache.
"""

from __future__ import annotations

import jax
import pytest
from jax import random

from factoriax.achievements import core_game_conditions
from factoriax.constants import Action
from factoriax.envs.factoriax_env import FactoriaXEnv
from factoriax.state import EnvParams

# Each test JIT-compiles step_env — necessarily slow, but catches the
# "game freezes for 5s on first retrace" regression. CI-only.
pytestmark = pytest.mark.slow


def _setup():
    """Create env, reset, and JIT step_fn."""
    env = FactoriaXEnv(achievement_fn=core_game_conditions)
    params = EnvParams(map_width=8, map_height=8, num_players=1)
    rng = random.PRNGKey(42)
    rng, k = random.split(rng)
    _, state = env.reset_env(k, params)
    step_fn = jax.jit(env.step_env)
    return step_fn, state, params, rng


class TestNoRetrace:
    """step_fn must not retrace after the first call."""

    def test_second_call_reuses_cache(self) -> None:
        """Calling step_fn twice with post-step state: cache stays 1."""
        step_fn, state, params, rng = _setup()

        rng, k = random.split(rng)
        _, state, _, _, _ = step_fn(k, state, int(Action.NOOP), params)
        assert step_fn._cache_size() == 1

        rng, k = random.split(rng)
        _, state, _, _, _ = step_fn(k, state, int(Action.NOOP), params)
        assert step_fn._cache_size() == 1, (
            "JIT retraced on second call with same action. "
            "State dtype mismatch between reset and post-step."
        )

    def test_different_action_reuses_cache(self) -> None:
        """Different action values must not cause retrace."""
        step_fn, state, params, rng = _setup()

        rng, k = random.split(rng)
        _, state, _, _, _ = step_fn(k, state, int(Action.NOOP), params)
        assert step_fn._cache_size() == 1

        rng, k = random.split(rng)
        _, state, _, _, _ = step_fn(k, state, int(Action.MINE), params)
        assert step_fn._cache_size() == 1, (
            "JIT retraced on different action value. Action may be traced as static."
        )

    def test_repeated_mine_same_params_reuses_cache(self) -> None:
        """Repeated MINE with the same params must not retrace."""
        step_fn, state, params, rng = _setup()

        for _ in range(3):
            rng, k = random.split(rng)
            _, state, _, _, _ = step_fn(k, state, int(Action.MINE), params)

        assert step_fn._cache_size() == 1, (
            "JIT retraced on repeated MINE with identical params. "
            "player_mining_yield should be traced (not static)."
        )


class TestDtypeConsistency:
    """Reset state dtypes must match post-step state dtypes."""

    def test_all_leaves_match(self) -> None:
        """Every pytree leaf must have same shape and dtype after step."""
        step_fn, state, params, rng = _setup()

        rng, k = random.split(rng)
        _, stepped, _, _, _ = step_fn(k, state, int(Action.NOOP), params)

        leaves0, treedef0 = jax.tree.flatten(state)
        leaves1, treedef1 = jax.tree.flatten(stepped)

        assert treedef0 == treedef1, "Pytree structure changed"
        assert len(leaves0) == len(leaves1)

        for i, (l0, l1) in enumerate(zip(leaves0, leaves1)):
            if hasattr(l0, "dtype") and hasattr(l1, "dtype"):
                assert l0.shape == l1.shape, f"Leaf {i}: shape {l0.shape} -> {l1.shape}"
                assert l0.dtype == l1.dtype, f"Leaf {i}: dtype {l0.dtype} -> {l1.dtype}"
            else:
                assert type(l0) is type(l1), f"Leaf {i}: type {type(l0)} -> {type(l1)}"
