"""Lightweight gymnax-contract tests for :class:`FactoriaxEnv`.

Pins the RL interface that consumers bind to: the declared spaces match the
real observation/action, ``step_env`` returns the gymnax 5-tuple with the right
types, the env is vmappable, and auto-reset restores a fresh episode on
termination. Kept cheap on purpose -- one 8x8 level (built once), at most one
step per test, no recompiles across map sizes.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import pytest
from jax import random

from factoriax.engine.constants import NUM_ACTIONS, Action
from factoriax.engine.envs.base import FactoriaxEnv
from factoriax.engine.envs.wrappers import AutoResetWrapper
from factoriax.engine.levels import LevelBuilder


@pytest.fixture(scope="module")
def level8():
    """A tiny 8x8 level, built once for the whole module."""
    return LevelBuilder(8, 8).build("contract")


@pytest.fixture(scope="module")
def default_env(level8):
    """The default (global-obs, unwrapped) env."""
    env = FactoriaxEnv(level=level8)
    return env, env.default_params


def test_action_space_is_discrete_num_actions(default_env) -> None:
    env, params = default_env
    assert env.action_space(params).n == NUM_ACTIONS


def test_observation_space_matches_obs(default_env) -> None:
    """The declared global ``observation_space`` matches a real obs."""
    env, params = default_env
    obs, _ = env.reset_env(random.PRNGKey(0), params)
    space = env.observation_space(params)
    assert obs.shape == space.shape
    assert obs.dtype == space.dtype
    assert bool(jnp.all(obs >= space.low)) and bool(jnp.all(obs <= space.high))


def test_local_observation_space_matches_obs(level8) -> None:
    """``observation_space`` matches ``get_obs`` for a local x_ray env."""
    env = FactoriaxEnv(level=level8, obs="x_ray_local", obs_radius=3)
    params = env.default_params
    obs, _ = env.reset_env(random.PRNGKey(0), params)
    space = env.observation_space(params)
    assert obs.shape == space.shape
    assert obs.dtype == space.dtype


def test_step_returns_gymnax_five_tuple(default_env) -> None:
    env, params = default_env
    _, state = env.reset_env(random.PRNGKey(0), params)
    out = env.step_env(random.PRNGKey(1), state, int(Action.NOOP), params)
    assert len(out) == 5
    obs, _, reward, done, info = out
    assert obs.shape == env.observation_space(params).shape
    assert jnp.asarray(reward).shape == ()
    assert jnp.asarray(reward).dtype == jnp.float32
    assert jnp.asarray(done).shape == ()
    assert jnp.asarray(done).dtype == jnp.bool_
    assert isinstance(info, dict)


def test_step_is_vmappable(default_env) -> None:
    """reset + step vmap over a batch -- the env's whole reason to be JAX."""
    env, params = default_env
    keys = random.split(random.PRNGKey(0), 4)
    _, states = jax.vmap(env.reset_env, in_axes=(0, None))(keys, params)
    step_keys = random.split(random.PRNGKey(1), 4)
    actions = jnp.zeros(4, dtype=jnp.int32)
    obs, _, reward, done, _ = jax.vmap(env.step_env, in_axes=(0, 0, 0, None))(
        step_keys, states, actions, params
    )
    assert obs.shape[0] == 4
    assert reward.shape == (4,)
    assert done.shape == (4,)


def test_auto_reset_restores_episode_on_done(level8) -> None:
    """AutoResetWrapper restores the cached reset state when ``done`` fires."""
    env = AutoResetWrapper(FactoriaxEnv(level=level8))
    params = dataclasses.replace(env.default_params, max_timesteps=1)
    _, state = env.reset_env(random.PRNGKey(0), params)
    _, state1, _, done, _ = env.step_env(
        random.PRNGKey(1), state, int(Action.NOOP), params
    )
    assert bool(done)  # timestep reaches max_timesteps (1)
    # Restored to the cached reset: timestep is back to the episode start,
    # not the post-step value it would hold without auto-reset.
    assert int(state1.env_state.timestep) == int(state.reset_state.timestep)
    assert int(state1.env_state.timestep) < int(params.max_timesteps)
