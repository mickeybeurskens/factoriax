"""Tests for the gymnax-shaped ``factoriax.make()`` factory.

Spec: SPEC.md Phase B item 5. The factory builds the canonical
wrapper stack:
``Inner -> LocalObservation -> ActionMask -> ScienceTally -> AutoReset``
returning ``(env, params)`` like ``gymnax.make()``.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import random

import factoriax
from factoriax.engine.constants import MAX_ACHIEVEMENTS, Action
from factoriax.engine.levels import LevelBuilder
from factoriax.envs.action_mask_wrapper import ActionMaskWrapper
from factoriax.envs.auto_reset_wrapper import AutoResetWrapper
from factoriax.envs.factoriax_env import FactoriaXEnv
from factoriax.envs.local_observation_wrapper import LocalObservationWrapper


def test_make_level_by_registry_name() -> None:
    """A string level is resolved against ``LEVELS``."""
    env, params = factoriax.make("15x15_resources")
    _, state = env.reset_env(random.PRNGKey(0), params)
    # Level state has fixed geometry — different rngs yield identical maps.
    _, state2 = env.reset_env(random.PRNGKey(99), params)
    assert jnp.array_equal(state.map, state2.map)


def test_make_level_by_instance() -> None:
    """A ``Level`` instance is wired straight into the env constructor."""
    level = LevelBuilder(8, 8).build("tiny")
    env, params = factoriax.make(level)
    _, state = env.reset_env(random.PRNGKey(0), params)
    assert state.map.shape == (8, 8)


def test_make_achievement_fn_is_bound_to_inner_env() -> None:
    """``achievement_fn`` reaches the inner ``FactoriaXEnv`` through the stack."""

    def all_true(state):  # type: ignore[no-untyped-def]
        return jnp.ones(MAX_ACHIEVEMENTS, dtype=jnp.bool_)

    env, params = factoriax.make(achievement_fn=all_true)
    _, state = env.reset_env(random.PRNGKey(0), params)
    rng, step_key = random.split(random.PRNGKey(0))
    _, state, _, _, _ = env.step_env(step_key, state, 0, params)
    assert bool(state.achievements_unlocked.all())


def test_make_canonical_wrapper_order() -> None:
    """When all options are requested, wrappers compose in the canonical order.

    Outermost-to-innermost: AutoReset -> ActionMask -> LocalObservation -> Inner.
    """
    env, _ = factoriax.make(
        obs="local",
        obs_radius=3,
        auto_reset=True,
        blocked_actions=(int(Action.MINE),),
    )
    # Outermost is AutoResetWrapper.
    assert isinstance(env, AutoResetWrapper)
    inner1 = env._inner  # noqa: SLF001
    assert isinstance(inner1, ActionMaskWrapper)
    inner2 = inner1._inner  # noqa: SLF001
    assert isinstance(inner2, LocalObservationWrapper)
    inner3 = inner2._inner  # noqa: SLF001
    assert isinstance(inner3, FactoriaXEnv)


def test_make_unknown_level_raises() -> None:
    """A registry-name miss raises ``KeyError`` with a useful message."""
    import pytest

    with pytest.raises(KeyError, match="not_a_level"):
        factoriax.make("not_a_level")
