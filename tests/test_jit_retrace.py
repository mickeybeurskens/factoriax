"""Regression guard: JIT must not retrace after warmup.

The step function is JIT-compiled once during warmup. Subsequent
calls with the post-step state and different action values must
reuse the cached trace. A retrace causes a ~5s freeze in gameplay.

This module catches dtype mismatches between reset state and
post-step state (e.g. Python int vs ``jnp.int32``, ``int16`` promoted
to ``int32``) that would invalidate the JIT cache.

Single test by design: each call sequence below asserts the retrace
property at a different point in time, and they all share one
``jax.jit(env.step_env)`` wrapper. Splitting into multiple tests
would pay one ~7s XLA compile each (~28s total); a single sequence
covers the same regression surface in ~7s.
"""

from __future__ import annotations

import jax
import pytest
from jax import random

from factoriax.achievements import core_game_conditions
from factoriax.constants import Action
from factoriax.envs.factoriax_env import FactoriaXEnv
from factoriax.state import EnvParams

# Single retrace guard JIT-compiles step_env once — necessarily slow
# but catches the "game freezes for 5s on first retrace" regression
# class. CI-only.
pytestmark = pytest.mark.slow


def test_no_retrace_across_call_patterns() -> None:
    """``jax.jit(env.step_env)`` keeps ``_cache_size() == 1`` across:

    * Two consecutive NOOP steps (post-step state matches reset state).
    * NOOP followed by MINE (different action value, same arg signature).
    * Three repeated MINEs (params with ``player_mining_yield`` must be
      traced, not static).

    State pytree structure and per-leaf shapes/dtypes must also remain
    identical after the first step — any drift here would invalidate
    the cache invisibly.
    """
    env = FactoriaXEnv(achievement_fn=core_game_conditions)
    params = EnvParams(map_width=8, map_height=8, num_players=1)
    rng = random.PRNGKey(42)
    rng, k = random.split(rng)
    _, state0 = env.reset_env(k, params)
    step_fn = jax.jit(env.step_env)

    # Phase 1: NOOP twice — second call must reuse cache.
    rng, k = random.split(rng)
    _, state1, _, _, _ = step_fn(k, state0, int(Action.NOOP), params)
    assert step_fn._cache_size() == 1, (
        "JIT retraced on first NOOP. State dtype mismatch between reset "
        "and post-step state."
    )

    # Pytree structure + per-leaf shapes/dtypes must match.
    leaves0, treedef0 = jax.tree.flatten(state0)
    leaves1, treedef1 = jax.tree.flatten(state1)
    assert treedef0 == treedef1, "Pytree structure changed after step"
    assert len(leaves0) == len(leaves1)
    for i, (l0, l1) in enumerate(zip(leaves0, leaves1, strict=True)):
        if hasattr(l0, "dtype") and hasattr(l1, "dtype"):
            assert l0.shape == l1.shape, f"Leaf {i}: shape {l0.shape} -> {l1.shape}"
            assert l0.dtype == l1.dtype, f"Leaf {i}: dtype {l0.dtype} -> {l1.dtype}"
        else:
            assert type(l0) is type(l1), f"Leaf {i}: type {type(l0)} -> {type(l1)}"

    rng, k = random.split(rng)
    _, state2, _, _, _ = step_fn(k, state1, int(Action.NOOP), params)
    assert step_fn._cache_size() == 1, (
        "JIT retraced on second NOOP call with post-step state."
    )

    # Phase 2: different action value must not retrace.
    rng, k = random.split(rng)
    _, state3, _, _, _ = step_fn(k, state2, int(Action.MINE), params)
    assert step_fn._cache_size() == 1, (
        "JIT retraced on different action value. Action may be traced as static."
    )

    # Phase 3: repeated MINE with identical params must not retrace —
    # ``player_mining_yield`` should be traced (not static).
    state = state3
    for _ in range(3):
        rng, k = random.split(rng)
        _, state, _, _, _ = step_fn(k, state, int(Action.MINE), params)
    assert step_fn._cache_size() == 1, (
        "JIT retraced on repeated MINE with identical params. "
        "player_mining_yield should be traced (not static)."
    )
