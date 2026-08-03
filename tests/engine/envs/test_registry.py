"""Tests for :mod:`factoriax.engine.envs.registry`.

The registry maps a scenario id to a built env. ``make`` is the entry point a
consumer binds to, so an id that the catalog lists must build and step, and a
retired id must raise rather than resolve to something else.
"""

from __future__ import annotations

import pytest
from jax import random

from factoriax.engine.constants import MAX_ACHIEVEMENTS, Action
from factoriax.engine.envs.registry import ScenarioSpec, list_scenarios, make
from factoriax.engine.envs.wrappers import ActionMaskWrapper, AutoResetWrapper

_NOOP = int(Action.NOOP)


def test_catalog_lists_expected_scenarios() -> None:
    catalog = dict(list_scenarios())
    assert set(catalog) == {
        "EasyRocket-v1",
        "Rocket-v1",
        "Mining-v1",
        "MinerBootstrap-v1",
        "ScienceTiers-v1",
    }
    for spec in catalog.values():
        assert isinstance(spec, ScenarioSpec)
        assert spec.name and spec.description and callable(spec.build)


def test_make_easy_rocket_builds_and_steps() -> None:
    env, params = make("EasyRocket-v1", auto_reset=False)
    assert env.map_width == 16 and env.map_height == 16
    obs, state = env.reset_env(random.PRNGKey(0), params)
    assert obs.shape == env.observation_space(params).shape
    assert state.achievements_unlocked.shape == (MAX_ACHIEVEMENTS,)
    _, _, reward, done, _ = env.step_env(random.PRNGKey(1), state, _NOOP, params)
    assert float(reward) == 0.0 and not bool(done)


def test_make_rocket_is_32x32_and_masked() -> None:
    env, params = make("Rocket-v1", auto_reset=False)
    assert env.map_width == 32 and env.map_height == 32
    assert isinstance(env, ActionMaskWrapper)


def test_make_auto_reset_wraps() -> None:
    env, _ = make("EasyRocket-v1", auto_reset=True)
    assert isinstance(env, AutoResetWrapper)


def test_make_unknown_id_raises() -> None:
    with pytest.raises(KeyError):
        make("Nope-v1")


@pytest.mark.parametrize("env_id", ["MineOres-v1", "PlaceMiners-v1", "CraftMiners-v1"])
def test_removed_curriculum_ids_raise(env_id: str) -> None:
    """The forward-curriculum stage envs are gone. States define the
    backward curriculum, and environments do not."""
    with pytest.raises(KeyError):
        make(env_id)
