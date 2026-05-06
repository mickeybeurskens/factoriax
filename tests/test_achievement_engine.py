"""Tests for the achievement-evaluation pass baked into FactoriaXEnv.step_env.

The achievement system used to live in a wrapper (AchievementWrapper);
this set of tests pins the engine-state version: a constructor argument
on FactoriaXEnv, evaluated and OR-folded inside step_env, with the
result available on state.achievements_unlocked. Spec: SPEC.md Phase A
item 1.2.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import random

from factoriax.constants import MAX_ACHIEVEMENTS
from factoriax.envs.factoriax_env import FactoriaXEnv
from factoriax.state import EnvState


def _all_true(state: EnvState) -> jnp.ndarray:
    """Achievement fn that unconditionally returns all-True."""
    return jnp.ones(MAX_ACHIEVEMENTS, dtype=jnp.bool_)


def _fire_when_timestep_one(state: EnvState) -> jnp.ndarray:
    """Achievement fn whose condition is True only when state.timestep == 1.

    Used to verify that latching outlives the condition: once the
    achievement fires at step 1, it must stay set even though the
    condition flips back to False at step 2.
    """
    one = jnp.ones(MAX_ACHIEVEMENTS, dtype=jnp.bool_)
    zero = jnp.zeros(MAX_ACHIEVEMENTS, dtype=jnp.bool_)
    return jnp.where(state.timestep == 1, one, zero)


def test_default_achievement_fn_keeps_unlocks_all_false() -> None:
    """No achievement_fn → achievements_unlocked stays all-False after a step."""
    env = FactoriaXEnv()
    params = env.default_params
    rng = random.PRNGKey(0)
    _, state = env.reset_env(rng, params)

    rng, step_key = random.split(rng)
    _, state, _, _, _ = env.step_env(step_key, state, 0, params)

    assert not bool(state.achievements_unlocked.any())


def test_engine_latches_unlocks_from_achievement_fn() -> None:
    """An achievement_fn that returns all-True unlocks every bit on step 1."""
    env = FactoriaXEnv(achievement_fn=_all_true)
    params = env.default_params
    rng = random.PRNGKey(0)
    _, state = env.reset_env(rng, params)

    assert not bool(state.achievements_unlocked.any())

    rng, step_key = random.split(rng)
    _, state, _, _, _ = env.step_env(step_key, state, 0, params)

    assert bool(state.achievements_unlocked.all())


def test_unlocks_are_latched_across_steps() -> None:
    """Once an achievement unlocks, it stays unlocked when the condition flips off."""
    env = FactoriaXEnv(achievement_fn=_fire_when_timestep_one)
    params = env.default_params
    rng = random.PRNGKey(0)
    _, state = env.reset_env(rng, params)

    rng, step_key = random.split(rng)
    _, state, _, _, _ = env.step_env(step_key, state, 0, params)
    assert bool(state.achievements_unlocked.all())  # condition fired at timestep=1

    rng, step_key = random.split(rng)
    _, state, _, _, _ = env.step_env(step_key, state, 0, params)
    # condition is False at timestep=2 but unlocks must persist
    assert bool(state.achievements_unlocked.all())
