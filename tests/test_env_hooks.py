"""Tests for the additive FactoriaxEnv extension points (Phase 0 of the
scenario refactor): ``reset_fn``, ``step_hooks``, ``reward_fn``, the
``achievement_hook`` helper, and ``AutoResetWrapper(resample=True)``.

Kept cheap: one 8x8 level built once, at most one step per test, no recompiles.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import pytest
from jax import random

from factoriax.engine.constants import MAX_ACHIEVEMENTS, Action
from factoriax.engine.envs.base import FactoriaxEnv
from factoriax.engine.envs.wrappers import AutoResetWrapper
from factoriax.engine.envs.base import achievement_hook
from factoriax.engine.levels import LevelBuilder, build_state

_NOOP = int(Action.NOOP)


@pytest.fixture(scope="module")
def level8():
    return LevelBuilder(8, 8).build("hooks")


@pytest.fixture(scope="module")
def params(level8):
    return FactoriaxEnv(level=level8).default_params


def _bit(index: int) -> jnp.ndarray:
    return jnp.zeros(MAX_ACHIEVEMENTS, dtype=jnp.bool_).at[index].set(True)


def test_reset_fn_overrides_level(level8, params) -> None:
    """A supplied ``reset_fn`` is used and the bound ``level`` is ignored."""

    def custom_reset(key, p):
        del key
        return build_state(level8, p).replace(achievements_unlocked=_bit(2))

    env = FactoriaxEnv(reset_fn=custom_reset, level=level8)
    _, state = env.reset_env(random.PRNGKey(0), params)
    assert bool(state.achievements_unlocked[2])


def test_step_hooks_run_after_step(params) -> None:
    """Step hooks are applied after the physics step, in order."""

    def set_bit0(key, state, p):
        del key, p
        return state.replace(
            achievements_unlocked=state.achievements_unlocked.at[0].set(True)
        )

    env = FactoriaxEnv(step_hooks=(set_bit0,))
    _, state = env.reset_env(random.PRNGKey(0), params)
    assert not bool(state.achievements_unlocked[0])
    _, new_state, _, _, _ = env.step_env(random.PRNGKey(1), state, _NOOP, params)
    assert bool(new_state.achievements_unlocked[0])


def test_reward_fn_used_else_zero(params) -> None:
    """``step_env`` returns ``reward_fn``'s value, or 0.0 when unset."""
    env_r = FactoriaxEnv(reward_fn=lambda prev, new, p: jnp.float32(7.0))
    _, state = env_r.reset_env(random.PRNGKey(0), params)
    _, _, reward, _, _ = env_r.step_env(random.PRNGKey(1), state, _NOOP, params)
    assert float(reward) == 7.0

    env_0 = FactoriaxEnv()
    _, state0 = env_0.reset_env(random.PRNGKey(0), params)
    _, _, reward0, _, _ = env_0.step_env(random.PRNGKey(1), state0, _NOOP, params)
    assert float(reward0) == 0.0


def test_achievement_hook_latches(level8, params) -> None:
    """``achievement_hook`` OR-folds bits and they latch across applications."""
    hook_a = achievement_hook(lambda s: _bit(3))
    hook_b = achievement_hook(lambda s: _bit(5))
    state = build_state(level8, params)
    assert not bool(jnp.any(state.achievements_unlocked))
    state = hook_a(random.PRNGKey(0), state, params)
    assert bool(state.achievements_unlocked[3])
    state = hook_b(random.PRNGKey(0), state, params)
    assert bool(state.achievements_unlocked[3]) and bool(state.achievements_unlocked[5])


def test_autoreset_resample_regenerates_on_done(level8, params) -> None:
    """``resample=True`` reruns ``reset_fn`` on a fresh key at termination;
    ``resample=False`` restores the cached reset state."""

    def stamped_reset(key, p):
        idx = jax.random.randint(key, (), 0, MAX_ACHIEVEMENTS)
        return build_state(level8, p).replace(
            achievements_unlocked=jnp.zeros(MAX_ACHIEVEMENTS, jnp.bool_)
            .at[idx]
            .set(True)
        )

    p1 = dataclasses.replace(params, max_timesteps=1)
    inner = FactoriaxEnv(reset_fn=stamped_reset)
    step_key = random.PRNGKey(1)
    reset_key = jax.random.split(step_key)[1]

    resampling = AutoResetWrapper(inner, resample=True)
    _, st = resampling.reset_env(random.PRNGKey(0), p1)
    _, st1, _, done, _ = resampling.step_env(step_key, st, _NOOP, p1)
    assert bool(done)
    expected = stamped_reset(reset_key, p1).achievements_unlocked
    assert bool(jnp.array_equal(st1.env_state.achievements_unlocked, expected))

    cached = AutoResetWrapper(inner, resample=False)
    _, st2 = cached.reset_env(random.PRNGKey(0), p1)
    _, st3, _, done2, _ = cached.step_env(step_key, st2, _NOOP, p1)
    assert bool(done2)
    assert bool(
        jnp.array_equal(
            st3.env_state.achievements_unlocked, st2.reset_state.achievements_unlocked
        )
    )
