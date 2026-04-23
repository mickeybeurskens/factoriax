"""Gymnax wrappers for FactoriaXEnv.

The core :class:`~factoriax.envs.factoriax_env.FactoriaXEnv` is a pure
simulation engine. These wrappers add behavior on top:

- :class:`AchievementWrapper` — tracks achievement progression
- :class:`AutoResetWrapper` — cached-state auto-reset for ``lax.scan``
  training loops
- :class:`LocalObservationWrapper` — swap ``get_obs`` to a local
  radius-R window centered on the selected player
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
from factoriax.observations import (
    NUM_PLAYER_SCALARS,
    NUM_SPATIAL_CHANNELS,
    local_array,
)
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
        achievement_fn: Callable[[EnvState], jax.Array] = _no_achievements,
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
            key,
            state.env_state,
            action,
            params,
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
            achievements_unlocked=jnp.zeros(MAX_ACHIEVEMENTS, dtype=jnp.bool_),
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
            achievements_unlocked=jnp.zeros(MAX_ACHIEVEMENTS, dtype=jnp.bool_),
        )
        return obs, state

    def get_obs(self, state: AchievementState, params: EnvParams) -> jax.Array:
        """Get observation for the selected player.

        Args:
            state: Current wrapped state.
            params: Environment parameters.

        Returns:
            Float32 observation array.
        """
        return self._inner.get_obs(state.env_state, params)

    def is_terminal(self, state: AchievementState, params: EnvParams) -> jax.Array:
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


class LocalObservationWrapper(environment.Environment[EnvState, EnvParams]):  # type: ignore[misc]
    """Replace the inner env's full-map obs with a local radius-R window.

    Wraps :class:`FactoriaXEnv` directly and substitutes
    :func:`factoriax.observations.local_array` for ``get_obs``. The
    window side length is ``2 * radius + 1``; at ``radius=3`` each obs
    is a 7x7 window centered on the selected player.

    This wrapper must sit between :class:`FactoriaXEnv` and any state-
    composing wrapper (:class:`AchievementWrapper`, :class:`AutoResetWrapper`)
    because it operates on raw :class:`EnvState`, not on a composite
    state type.

    Args:
        inner: Core FactoriaX environment.
        radius: Half-width of the observation window in tiles.
    """

    def __init__(self, inner: FactoriaXEnv, radius: int) -> None:
        """Initialize the wrapper.

        Args:
            inner: Core environment instance.
            radius: Half-width of the observation window.
        """
        super().__init__()
        self._inner = inner
        self._radius = int(radius)

    @property
    def default_params(self) -> EnvParams:
        """Return default environment parameters."""
        return self._inner.default_params

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[str, Any]]:
        """Forward to inner, then recompute obs with the local window.

        The inner env already computes a global obs inside ``step_env``;
        we discard it and return our local obs instead. XLA's dead-code
        elimination drops the unused global computation inside JIT.
        """
        _, new_state, reward, done, info = self._inner.step_env(
            key,
            state,
            action,
            params,
        )
        obs = self.get_obs(new_state, params)
        return obs, new_state, reward, done, info

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        """Reset the inner env and return a local observation."""
        _, state = self._inner.reset_env(key, params)
        return self.get_obs(state, params), state

    def reset_from_level(
        self, level: Level, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        """Reset to a pre-built level and return a local observation."""
        _, state = self._inner.reset_from_level(level, params)
        return self.get_obs(state, params), state

    def get_obs(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Get the local observation for the selected player."""
        return local_array(
            state,
            params,
            state.selected_player,
            radius=self._radius,
        )

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Check if the current state is terminal."""
        return self._inner.is_terminal(state, params)

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        """Return the action space (unchanged by this wrapper)."""
        return self._inner.action_space(params)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Return the local observation space.

        Shape is ``(NUM_SPATIAL_CHANNELS * (2r+1)^2 + NUM_PLAYER_SCALARS,)``.
        """
        size = 2 * self._radius + 1
        obs_size = NUM_SPATIAL_CHANNELS * size * size + NUM_PLAYER_SCALARS
        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_size,),
            dtype=jnp.float32,
        )


class AutoResetState(struct.PyTreeNode):  # type: ignore[no-untyped-call]
    """State that caches the initial configuration for cheap auto-reset.

    Stores a frozen copy of the initial state alongside the live state.
    On episode termination, ``lax.select`` swaps in the cached copy
    instead of calling ``reset_env`` (which runs full procedural
    terrain generation).

    The memory cost is one extra copy of ``EnvState`` per batch
    element. At 32x32 that is roughly 12 KB per element. At 128x128
    with ``max_machines=4096`` it grows to roughly 185 KB per element,
    which doubles total state memory. For large maps, reduce
    ``batch_size`` or ``max_machines`` accordingly.

    Attributes:
        env_state: Live environment state that evolves each step.
        reset_state: Frozen copy from the initial reset, restored
            on episode termination.
    """

    env_state: EnvState
    reset_state: EnvState


class AutoResetWrapper(environment.Environment[AutoResetState, EnvParams]):  # type: ignore[misc]
    """Gymnax wrapper providing cached-state auto-reset.

    The default :meth:`FactoriaXEnv.step` does not auto-reset because
    the gymnax base implementation calls ``reset_env`` every tick,
    running full procedural terrain generation unconditionally. That
    roughly triples the per-step cost.

    This wrapper restores auto-reset cheaply by caching the initial
    state at reset time and using ``lax.select`` on termination. The
    terrain generation cost is paid once at reset, not every step.

    Use this wrapper when you need auto-reset inside ``lax.scan``
    training loops (e.g. PureJaxRL-style PPO). For manual episode
    management, use :class:`FactoriaXEnv` directly.

    Args:
        inner: Core FactoriaX environment to wrap.
    """

    def __init__(self, inner: FactoriaXEnv) -> None:
        """Initialize the auto-reset wrapper.

        Args:
            inner: Core environment instance.
        """
        super().__init__()
        self._inner = inner

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
        state: AutoResetState,
        action: int | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, AutoResetState, jax.Array, jax.Array, dict[str, Any]]:
        """Step the environment with cached auto-reset on termination.

        When ``done`` is True, the live state is replaced with the
        cached reset state via ``lax.select``. No terrain generation
        occurs.

        Args:
            key: JAX random key.
            state: Current wrapped state with cached reset.
            action: Action to take.
            params: Environment parameters.

        Returns:
            Tuple of (observation, new_state, reward, done, info).
        """
        obs_step, new_env, reward, done, info = self._inner.step_env(
            key,
            state.env_state,
            action,
            params,
        )
        reset_env = state.reset_state
        final_env = jax.tree.map(
            lambda r, s: jax.lax.select(done, r, s),
            reset_env,
            new_env,
        )
        obs_reset = self._inner.get_obs(reset_env, params)
        final_obs = jax.lax.select(done, obs_reset, obs_step)
        new_state = state.replace(env_state=final_env)
        return final_obs, new_state, reward, done, info

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, AutoResetState]:
        """Reset and cache the initial state for future auto-resets.

        Args:
            key: JAX random key for world generation.
            params: Environment parameters.

        Returns:
            Tuple of (initial_observation, wrapped_state).
        """
        obs, env_state = self._inner.reset_env(key, params)
        state = AutoResetState(
            env_state=env_state,
            reset_state=env_state,
        )
        return obs, state

    def reset_from_level(
        self, level: Level, params: EnvParams
    ) -> tuple[jax.Array, AutoResetState]:
        """Reset to a level and cache the state for auto-resets.

        Args:
            level: Level definition.
            params: Environment parameters.

        Returns:
            Tuple of (initial_observation, wrapped_state).
        """
        obs, env_state = self._inner.reset_from_level(level, params)
        state = AutoResetState(
            env_state=env_state,
            reset_state=env_state,
        )
        return obs, state

    def get_obs(self, state: AutoResetState, params: EnvParams) -> jax.Array:
        """Get observation for the selected player.

        Args:
            state: Current wrapped state.
            params: Environment parameters.

        Returns:
            Float32 observation array.
        """
        return self._inner.get_obs(state.env_state, params)

    def is_terminal(self, state: AutoResetState, params: EnvParams) -> jax.Array:
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
