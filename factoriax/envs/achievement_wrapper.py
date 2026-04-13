"""Gymnax wrapper that adds achievement tracking to FactoriaXEnv.

The core :class:`~factoriax.envs.factoriax_env.FactoriaXEnv` is a pure
simulation engine with no opinions about achievements or rewards. This
wrapper composes over it, evaluating an achievement condition function
each step and latching results into its own state. Play mode and
benchmarks use this to track progression through the standard gymnax
``step()`` interface.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces  # type: ignore[import-untyped]

from factoriax.constants import MAX_ACHIEVEMENTS
from factoriax.envs.factoriax_env import FactoriaXEnv
from factoriax.levels import Level
from factoriax.state import EnvParams, EnvState


def _no_achievements(state: EnvState) -> jax.Array:
    """Placeholder returning no unlocked achievements.

    The core game condition functions reference the old grid-based
    ``machine_inventory`` field that was replaced by entity arrays
    in the architecture redesign. Until those are ported, this
    no-op prevents runtime errors.

    Args:
        state: Current environment state (unused).

    Returns:
        All-false boolean array of shape ``(MAX_ACHIEVEMENTS,)``.
    """
    return jnp.zeros(MAX_ACHIEVEMENTS, dtype=jnp.bool_)


class AchievementState(struct.PyTreeNode):  # type: ignore[no-untyped-call]
    """Composite state pairing game state with achievement tracking.

    Attributes:
        env_state: Inner game state from :class:`FactoriaXEnv`.
        achievements_unlocked: Latched achievement flags, shape
            ``(MAX_ACHIEVEMENTS,)``, bool.
    """

    env_state: EnvState
    achievements_unlocked: jnp.ndarray


class AchievementWrapper(environment.Environment[AchievementState, EnvParams]):  # type: ignore[misc]
    """Gymnax wrapper that tracks achievement progression.

    Evaluates an achievement condition function after each step and
    latches newly satisfied conditions. The wrapper does not compute
    rewards, that responsibility belongs to skill-specific wrappers.

    Args:
        inner: Core FactoriaX environment to wrap.
        achievement_fn: Pure function ``(EnvState) -> jax.Array``
            returning a boolean array of shape ``(MAX_ACHIEVEMENTS,)``.
            Defaults to a no-op placeholder until the core game
            conditions are ported to the entity-based state.
    """

    def __init__(
        self,
        inner: FactoriaXEnv,
        achievement_fn: Callable[
            [EnvState], jax.Array
        ] = _no_achievements,
    ) -> None:
        """Initialize the wrapper.

        Args:
            inner: Core environment instance.
            achievement_fn: Achievement condition function.
        """
        super().__init__()
        self._inner = inner
        self._achievement_fn = achievement_fn

    @property
    def default_params(self) -> EnvParams:
        """Return default environment parameters.

        Returns:
            Default EnvParams from the inner environment.
        """
        return self._inner.default_params

    def step_env(
        self,
        key: jax.Array,
        state: AchievementState,
        action: int | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, AchievementState, jax.Array, jax.Array, dict[str, Any]]:
        """Step the environment and update achievement tracking.

        Args:
            key: JAX random key.
            state: Current wrapped state.
            action: Action to take.
            params: Environment parameters.

        Returns:
            Tuple of (observation, new_state, reward, done, info).
        """
        obs, new_env, reward, done, info = self._inner.step_env(
            key, state.env_state, action, params,
        )
        conditions = self._achievement_fn(new_env)
        new_state = state.replace(
            env_state=new_env,
            achievements_unlocked=state.achievements_unlocked | conditions,
        )
        return obs, new_state, reward, done, info

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, AchievementState]:
        """Reset the environment with fresh achievement state.

        Args:
            key: JAX random key for world generation.
            params: Environment parameters.

        Returns:
            Tuple of (initial_observation, wrapped_state).
        """
        obs, env_state = self._inner.reset_env(key, params)
        state = AchievementState(
            env_state=env_state,
            achievements_unlocked=jnp.zeros(
                MAX_ACHIEVEMENTS, dtype=jnp.bool_
            ),
        )
        return obs, state

    def reset_from_level(
        self, level: Level, params: EnvParams
    ) -> tuple[jax.Array, AchievementState]:
        """Reset the environment to a pre-built level.

        Args:
            level: Level definition.
            params: Environment parameters.

        Returns:
            Tuple of (initial_observation, wrapped_state).
        """
        obs, env_state = self._inner.reset_from_level(level, params)
        state = AchievementState(
            env_state=env_state,
            achievements_unlocked=jnp.zeros(
                MAX_ACHIEVEMENTS, dtype=jnp.bool_
            ),
        )
        return obs, state

    def get_obs(
        self, state: AchievementState, params: EnvParams
    ) -> jax.Array:
        """Get observation for the selected player.

        Args:
            state: Current wrapped state.
            params: Environment parameters.

        Returns:
            Float32 observation array.
        """
        return self._inner.get_obs(state.env_state, params)

    def is_terminal(
        self, state: AchievementState, params: EnvParams
    ) -> jax.Array:
        """Check if the current state is terminal.

        Args:
            state: Current wrapped state.
            params: Environment parameters.

        Returns:
            Boolean indicating whether state is terminal.
        """
        return self._inner.is_terminal(state.env_state, params)

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        """Return the action space.

        Args:
            params: Environment parameters.

        Returns:
            Discrete action space.
        """
        return self._inner.action_space(params)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Return the observation space.

        Args:
            params: Environment parameters.

        Returns:
            Box observation space.
        """
        return self._inner.observation_space(params)
