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
from factoriax.constants import MAX_ACHIEVEMENTS, NUM_ITEM_TYPES, Action
from factoriax.envs.action_mask_wrapper import ActionMaskWrapper
from factoriax.envs.auto_reset_wrapper import AutoResetState, AutoResetWrapper
from factoriax.envs.factoriax_env import FactoriaXEnv
from factoriax.envs.local_observation_wrapper import LocalObservationWrapper
from factoriax.levels import LevelBuilder
from factoriax.state import EnvParams


def test_make_default_returns_env_and_params() -> None:
    """``factoriax.make()`` with no args returns a usable env + params."""
    env, params = factoriax.make()
    assert isinstance(params, EnvParams)
    # The bare env is an unwrapped FactoriaXEnv when nothing else is requested.
    assert isinstance(env, FactoriaXEnv)


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


def test_make_local_obs_wraps_in_local_observation_wrapper() -> None:
    """``obs='local'`` adds ``LocalObservationWrapper`` around the inner env."""
    env, _ = factoriax.make(obs="local", obs_radius=3)
    assert isinstance(env, LocalObservationWrapper)


def test_make_achievement_fn_is_bound_to_inner_env() -> None:
    """``achievement_fn`` reaches the inner ``FactoriaXEnv`` through the stack."""

    def all_true(state):  # type: ignore[no-untyped-def]
        return jnp.ones(MAX_ACHIEVEMENTS, dtype=jnp.bool_)

    env, params = factoriax.make(achievement_fn=all_true)
    _, state = env.reset_env(random.PRNGKey(0), params)
    rng, step_key = random.split(random.PRNGKey(0))
    _, state, _, _, _ = env.step_env(step_key, state, 0, params)
    assert bool(state.achievements_unlocked.all())


def test_make_auto_reset_wraps_in_auto_reset_wrapper() -> None:
    """``auto_reset=True`` wraps the outermost env in ``AutoResetWrapper``."""
    env, params = factoriax.make(auto_reset=True)
    _, state = env.reset_env(random.PRNGKey(0), params)
    assert isinstance(env, AutoResetWrapper)
    assert isinstance(state, AutoResetState)


def test_make_blocked_actions_wraps_in_action_mask_wrapper() -> None:
    """Non-empty ``blocked_actions`` adds ``ActionMaskWrapper``."""
    env, _ = factoriax.make(blocked_actions=(int(Action.MINE),))
    assert isinstance(env, ActionMaskWrapper)


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


def test_make_returns_default_params_for_unspecified() -> None:
    """The returned params match the inner env's defaults when no level is set."""
    env, params = factoriax.make()
    assert params == env.default_params


def test_make_unknown_level_raises() -> None:
    """A registry-name miss raises ``KeyError`` with a useful message."""
    import pytest

    with pytest.raises(KeyError, match="not_a_level"):
        factoriax.make("not_a_level")


# Pin against the stale obs vector size when nothing is wrapped.
def test_make_default_global_obs_shape_pins_to_inner() -> None:
    """Default ``obs='global'`` keeps the inner env's observation_space."""
    env, params = factoriax.make()
    inner = FactoriaXEnv()
    assert env.observation_space(params).shape == inner.observation_space(params).shape


# Extra defence: NUM_ITEM_TYPES is imported into this file so future schema
# bumps surface as a typecheck failure here as well as in the obs tests.
_ = NUM_ITEM_TYPES
