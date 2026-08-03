"""Gymnax API conformance for :class:`FactoriaxEnv`.

This file asserts a rule that no single module owns, so it is not in the
mirror. A consumer binds to the gymnax interface: the declared spaces must
match the real observation and action, ``step_env`` must return the
5-tuple, the state dtypes must hold, and the whole thing must survive
``jit`` and ``vmap``.
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
from factoriax.engine.observations import (
    NUM_PLAYER_SCALARS,
    NUM_SPATIAL_CHANNELS,
)


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
    # not the post-step value that it holds without auto-reset.
    assert int(state1.env_state.timestep) == int(state.reset_state.timestep)
    assert int(state1.env_state.timestep) < int(params.max_timesteps)


#: State fields whose width bounds a game value, and the dtype they must keep.
#: ``block_resources`` at int16 is what caps ``BLOCK_MAX_RESOURCES``; the count
#: fields at int16 cap what a machine slot can hold. Widening or narrowing one
#: changes those limits and the observation built from them.
_PINNED_STATE_DTYPES = {
    "block_resources": jnp.int16,
    "ent_buf_count": jnp.int16,
    "ent_asm_in_count": jnp.int16,
    "ent_asm_out_count": jnp.int16,
    "ent_health": jnp.int16,
}


@pytest.mark.parametrize(("field", "dtype"), sorted(_PINNED_STATE_DTYPES.items()))
def test_state_array_dtypes(field, dtype, default_env) -> None:
    """A reset state carries the dtypes the engine's value ranges depend on."""
    env, params = default_env
    _, state = env.reset_env(random.PRNGKey(0), params)
    got = getattr(state, field).dtype
    assert got == dtype, f"{field}: {got}, expected {jnp.dtype(dtype)}"


# -------------------------------------------------------------------------
# Assertions the functions above do not make
# -------------------------------------------------------------------------


def test_reset_starts_at_timestep_zero() -> None:
    """Reset returns an observation and a state."""
    env = FactoriaxEnv()
    params = env.default_params
    rng = random.PRNGKey(0)
    obs, state = env.reset_env(rng, params)

    assert obs is not None
    assert state is not None
    assert state.timestep == 0


def test_step_increments_timestep() -> None:
    """Each step increments the timestep."""
    env = FactoriaxEnv()
    params = env.default_params
    rng = random.PRNGKey(0)
    rng, reset_key, step_key = random.split(rng, 3)
    _, state = env.reset_env(reset_key, params)

    _, new_state, _, _, _ = env.step_env(
        step_key,
        state,
        Action.NOOP,
        params,
    )
    assert new_state.timestep == state.timestep + 1


def test_observation_space_matches_the_channel_formula() -> None:
    """The observation space matches the expected dimensions."""
    env = FactoriaxEnv()
    params = env.default_params
    obs_space = env.observation_space(params)
    expected_size = (
        NUM_SPATIAL_CHANNELS["x_ray"] * env.map_width * env.map_height
        + NUM_PLAYER_SCALARS["x_ray"]
    )
    assert obs_space.shape == (expected_size,)


def test_env_is_jit_compilable() -> None:
    """The environment is JIT-compilable."""
    env = FactoriaxEnv()
    params = env.default_params

    @jax.jit
    def run_episode(rng: jax.Array) -> jax.Array:
        rng, reset_key = random.split(rng)
        obs, state = env.reset_env(reset_key, params)
        rng, step_key = random.split(rng)
        obs, state, reward, done, _ = env.step_env(
            step_key, state, Action.RIGHT, params
        )
        return reward

    reward = run_episode(random.PRNGKey(0))
    assert reward is not None
