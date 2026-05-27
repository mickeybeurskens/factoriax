"""Auto-reset wrapper for FactoriaXEnv.

Provides cheap auto-reset for ``lax.scan`` training loops. The default
:meth:`FactoriaXEnv.step` skips auto-reset for performance — the gymnax
base ``step`` runs full procedural terrain generation every tick, which
roughly triples the per-step cost. This wrapper caches the initial
state at reset time and uses ``lax.select`` on termination, paying the
generation cost once instead of every step.
"""

from __future__ import annotations

from typing import Any

import jax
from flax import struct
from gymnax.environments import environment, spaces  # type: ignore[import-untyped]

from factoriax.engine.envs.factoriax_env import FactoriaXEnv
from factoriax.engine.state import EnvParams, EnvState


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

    Use this wrapper when you need auto-reset inside ``lax.scan``
    training loops (e.g. PureJaxRL-style PPO). For manual episode
    management, use :class:`FactoriaXEnv` directly.

    Args:
        inner: Core FactoriaX environment to wrap.

    Example:
        >>> import factoriax
        >>> env, params = factoriax.make(auto_reset=True)
        >>> # ``env.step_env`` now returns the next-episode reset state
        >>> # whenever ``done`` flips True, with no Python-side branch.
    """

    def __init__(self, inner: FactoriaXEnv, resample: bool = False) -> None:
        """Initialize the auto-reset wrapper.

        Args:
            inner: Core environment instance.
            resample: When ``False`` (default), termination restores the cached
                initial state (cheap, same layout every episode). When ``True``,
                termination calls ``inner.reset_env`` with a fresh key, so each
                episode regenerates from the scenario's ``reset_fn`` — at the cost
                of running generation every step (under ``vmap``/``select`` both
                branches execute), so reserve it for cheap generators.
        """
        super().__init__()
        self._inner = inner
        self._resample = resample

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
        step_key, reset_key = jax.random.split(key)
        obs_step, new_env, reward, done, info = self._inner.step_env(
            step_key,
            state.env_state,
            action,
            params,
        )
        if self._resample:
            obs_reset, reset_env = self._inner.reset_env(reset_key, params)
        else:
            reset_env = state.reset_state
            obs_reset = self._inner.get_obs(reset_env, params)
        final_env = jax.tree.map(
            lambda r, s: jax.lax.select(done, r, s),
            reset_env,
            new_env,
        )
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
