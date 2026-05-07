"""Local-observation wrapper for FactoriaXEnv.

Replaces the inner env's full-map ``get_obs`` with a windowed local
observation centered on the selected player. Used by baselines that
train policies on a partial-observability view.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces  # type: ignore[import-untyped]

from factoriax.envs.factoriax_env import FactoriaXEnv
from factoriax.observations import (
    NUM_PLAYER_SCALARS,
    NUM_SPATIAL_CHANNELS,
    local_array,
)
from factoriax.state import EnvParams, EnvState


class LocalObservationWrapper(environment.Environment[EnvState, EnvParams]):  # type: ignore[misc]
    """Replace the inner env's full-map obs with a local radius-R window.

    Wraps :class:`FactoriaXEnv` directly and substitutes
    :func:`factoriax.observations.local_array` for ``get_obs``. The
    window side length is ``2 * radius + 1``; at ``radius=3`` each obs
    is a 7x7 window centered on the selected player.

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
