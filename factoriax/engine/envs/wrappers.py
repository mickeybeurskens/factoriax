"""FactoriaX environment wrappers."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces  # type: ignore[import-untyped]

from factoriax.engine.constants import Action
from factoriax.engine.envs.base import FactoriaxEnv
from factoriax.engine.state import EnvParams, EnvState


# ---------------------------------------------------------------------------
# Auto-reset wrapper
# ---------------------------------------------------------------------------

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

    Parameters
    ----------

    Returns
    -------

    
    """

    env_state: EnvState
    reset_state: EnvState


class AutoResetWrapper(environment.Environment[AutoResetState, EnvParams]):  # type: ignore[misc]
    """Gymnax wrapper providing cached-state auto-reset.

    Use this wrapper when you need auto-reset inside ``lax.scan``
    training loops (e.g. PureJaxRL-style PPO). For manual episode
    management, use :class:`FactoriaxEnv` directly.

    Parameters
    ----------
    inner :
        Core FactoriaX environment to wrap.
        Examples
        --------

    Returns
    -------

    
    >>> import factoriax
        >>> env, params = factoriax.make("EasyRocket-v1", auto_reset=True)
        >>> # ``env.step_env`` now returns the next-episode reset state
        >>> # whenever ``done`` flips True, with no Python-side branch.
    """

    def __init__(self, inner: FactoriaxEnv, resample: bool = False) -> None:
        """Initialize the auto-reset wrapper.

        Parameters
        ----------
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
        """Delegate to the inner env's default params."""
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
        
        Parameters
        ----------
            key: JAX random key.
            state: Current wrapped state with cached reset.
            action: Action to take.

        Parameters
        ----------
        key : jax.Array :
            
        state : AutoResetState :
            
        action : int | jax.Array :
            
        params : EnvParams :
            
        key: jax.Array :
            
        state: AutoResetState :
            
        action: int | jax.Array :
            
        params: EnvParams :
            

        Returns
        -------

        
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
        
        Parameters
        ----------
            key: JAX random key for world generation.

        Parameters
        ----------
        key : jax.Array :
            
        params : EnvParams :
            
        key: jax.Array :
            
        params: EnvParams :
            

        Returns
        -------

        
        """
        obs, env_state = self._inner.reset_env(key, params)
        state = AutoResetState(
            env_state=env_state,
            reset_state=env_state,
        )
        return obs, state

    def get_obs(self, state: AutoResetState, params: EnvParams) -> jax.Array:
        """Pass-through observation from the inner env.

        Parameters
        ----------
        state : AutoResetState :
            
        params : EnvParams :
            
        state: AutoResetState :
            
        params: EnvParams :
            

        Returns
        -------

        
        """
        return self._inner.get_obs(state.env_state, params)

    def is_terminal(self, state: AutoResetState, params: EnvParams) -> jax.Array:
        """Delegate termination to the inner env.

        Parameters
        ----------
        state : AutoResetState :
            
        params : EnvParams :
            
        state: AutoResetState :
            
        params: EnvParams :
            

        Returns
        -------

        
        """
        return self._inner.is_terminal(state.env_state, params)

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        """Action space is unchanged.

        Parameters
        ----------
        params : EnvParams :
            
        params: EnvParams :
            

        Returns
        -------

        
        """
        return self._inner.action_space(params)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space is unchanged.

        Parameters
        ----------
        params : EnvParams :
            
        params: EnvParams :
            

        Returns
        -------

        
        """
        return self._inner.observation_space(params)


# ---------------------------------------------------------------------------
# Action-mask wrapper
# ---------------------------------------------------------------------------

class ActionMaskWrapper(environment.Environment[EnvState, EnvParams]):  # type: ignore[misc]
    """Replace blocked actions with :data:`Action.NOOP`.

    Parameters
    ----------
    inner :
        Environment to wrap. Any gymnax-compatible env that uses
        :class:`EnvState` as its state type works.
    blocked_actions :
        Iterable of ``Action`` integers to block. The
        mask is captured at construction time and baked into the
        JIT graph of :meth:`step_env`.
        Examples
        --------

    Returns
    -------

    
    >>> from factoriax import ActionMaskWrapper, FactoriaxEnv, Action
        >>> env = ActionMaskWrapper(FactoriaxEnv(), blocked_actions=(int(Action.MINE),))
        >>> # MINE actions become NOOPs inside ``env.step_env``.
    """

    def __init__(
        self,
        inner: environment.Environment[Any, EnvParams],
        blocked_actions: set[int] | list[int] | tuple[int, ...],
    ) -> None:
        super().__init__()
        self._inner = inner
        self._num_actions = len(Action)
        mask = jnp.zeros((self._num_actions,), dtype=jnp.bool_)
        for a in blocked_actions:
            mask = mask.at[int(a)].set(True)
        self._mask: jnp.ndarray = mask
        self._noop = jnp.int32(int(Action.NOOP))

    @property
    def default_params(self) -> EnvParams:
        """Delegate to the inner env's default params."""
        # gymnax's environment.Environment is untyped so the inner attribute
        # is Any; the runtime contract guarantees an EnvParams.
        params: EnvParams = self._inner.default_params
        return params

    def _rewrite(self, action: int | jax.Array) -> jax.Array:
        """Rewrite a blocked action to NOOP; pass others through unchanged."""
        action_i = jnp.asarray(action, dtype=jnp.int32)
        is_blocked = self._mask[action_i]
        return jnp.asarray(jnp.where(is_blocked, self._noop, action_i))

    def step_env(
        self,
        key: jax.Array,
        state: Any,
        action: int | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, Any, jax.Array, jax.Array, dict[str, Any]]:
        """Step the inner env after rewriting blocked actions to NOOP.

        Parameters
        ----------
        key : jax.Array :
            
        state : Any :
            
        action : int | jax.Array :
            
        params : EnvParams :
            
        key: jax.Array :
            
        state: Any :
            
        action: int | jax.Array :
            
        params: EnvParams :
            

        Returns
        -------

        
        """
        result: tuple[jax.Array, Any, jax.Array, jax.Array, dict[str, Any]] = (
            self._inner.step_env(key, state, self._rewrite(action), params)
        )
        return result

    def reset_env(
        self,
        key: jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, Any]:
        """Pass-through reset to the inner env."""
        result: tuple[jax.Array, Any] = self._inner.reset_env(key, params)
        return result

    def get_obs(self, state: Any, params: EnvParams) -> jax.Array:
        """Pass-through observation from the inner env."""
        obs: jax.Array = self._inner.get_obs(state, params)
        return obs

    def is_terminal(self, state: Any, params: EnvParams) -> jax.Array:
        """Pass-through termination check to the inner env."""
        terminal: jax.Array = self._inner.is_terminal(state, params)
        return terminal

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        """Action space is unchanged."""
        return self._inner.action_space(params)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space is unchanged."""
        return self._inner.observation_space(params)
