"""Gymnax API conformance for :class:`FactoriaxEnv`.

This file asserts a rule that no single module owns, so it is not in the
mirror. A consumer binds to the gymnax interface: the declared spaces must
match the real observation and action, ``step_env`` must return the 5-tuple,
and the whole thing must survive ``jax.jit``.
"""

from __future__ import annotations

import jax
from jax import random

from factoriax.engine.constants import NUM_ACTIONS, Action
from factoriax.engine.envs.base import FactoriaxEnv
from factoriax.engine.observations import NUM_PLAYER_SCALARS, NUM_SPATIAL_CHANNELS


class TestEnvironment:
    """Tests for the gymnax environment interface."""

    def test_make(self) -> None:
        """The environment factory returns an env and its params."""
        env = FactoriaxEnv()
        params = env.default_params
        assert env is not None
        assert params is not None

    def test_reset_returns_obs_and_state(self) -> None:
        """Reset returns an observation and a state."""
        env = FactoriaxEnv()
        params = env.default_params
        rng = random.PRNGKey(0)
        obs, state = env.reset_env(rng, params)

        assert obs is not None
        assert state is not None
        assert state.timestep == 0

    def test_step_returns_correct_tuple(self) -> None:
        """Step returns (obs, state, reward, done, info)."""
        env = FactoriaxEnv()
        params = env.default_params
        rng = random.PRNGKey(0)
        rng, reset_key, step_key = random.split(rng, 3)
        obs, state = env.reset_env(reset_key, params)

        obs, new_state, reward, done, info = env.step_env(
            step_key, state, Action.RIGHT, params
        )

        assert obs is not None
        assert new_state is not None
        assert isinstance(float(reward), float)
        assert isinstance(bool(done), bool)
        assert isinstance(info, dict)

    def test_step_increments_timestep(self) -> None:
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

    def test_action_space(self) -> None:
        """The action space matches the number of defined actions."""
        env = FactoriaxEnv()
        params = env.default_params
        action_space = env.action_space(params)
        assert action_space.n == NUM_ACTIONS

    def test_observation_space(self) -> None:
        """The observation space matches the expected dimensions."""
        env = FactoriaxEnv()
        params = env.default_params
        obs_space = env.observation_space(params)
        expected_size = (
            NUM_SPATIAL_CHANNELS["x_ray"] * env.map_width * env.map_height
            + NUM_PLAYER_SCALARS["x_ray"]
        )
        assert obs_space.shape == (expected_size,)

    def test_jit_compilation(self) -> None:
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
