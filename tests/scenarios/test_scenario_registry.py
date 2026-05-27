"""Tests for the scenario registry and ``make("<id>")`` (refactor Phase 2).

Pins the gymnax-style loading surface: the catalog lists the registered
scenarios with display metadata, ``make`` returns a steppable ``(env, params)``
pair, optional wrappers are applied, and ``factoriax.make`` dispatches string
scenario ids to the registry.
"""

from __future__ import annotations

import pytest
from jax import random

import factoriax
from factoriax.engine.constants import MAX_ACHIEVEMENTS, Action
from factoriax.engine.envs import ActionMaskWrapper, AutoResetWrapper
from factoriax.engine.scenarios import ScenarioSpec, list_scenarios, make

_NOOP = int(Action.NOOP)


def test_catalog_lists_expected_scenarios() -> None:
    catalog = dict(list_scenarios())
    assert set(catalog) == {"EasyRocket-v1", "Rocket-v1"}
    for spec in catalog.values():
        assert isinstance(spec, ScenarioSpec)
        assert spec.name and spec.description and callable(spec.build)


def test_make_easy_rocket_builds_and_steps() -> None:
    env, params = make("EasyRocket-v1")
    assert params.map_width == 16 and params.map_height == 16
    obs, state = env.reset_env(random.PRNGKey(0), params)
    assert obs.shape == env.observation_space(params).shape
    assert state.achievements_unlocked.shape == (MAX_ACHIEVEMENTS,)
    _, _, reward, done, _ = env.step_env(random.PRNGKey(1), state, _NOOP, params)
    assert float(reward) == 0.0 and not bool(done)


def test_make_rocket_is_32x32_and_masked() -> None:
    env, params = make("Rocket-v1")
    assert params.map_width == 32 and params.map_height == 32
    assert isinstance(env, ActionMaskWrapper)


def test_make_auto_reset_wraps() -> None:
    env, _ = make("EasyRocket-v1", auto_reset=True)
    assert isinstance(env, AutoResetWrapper)


def test_make_unknown_id_raises() -> None:
    with pytest.raises(KeyError):
        make("Nope-v1")


def test_factoriax_make_dispatches_scenario_ids() -> None:
    env, params = factoriax.make("EasyRocket-v1")
    assert params.map_width == 16
    obs, _ = env.reset_env(random.PRNGKey(0), params)
    assert obs.shape == env.observation_space(params).shape
