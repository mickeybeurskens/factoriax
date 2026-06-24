"""Gymnax wrapper that converts masked actions to ``NOOP`` before stepping.

Used by scenarios that want to exclude a subset of the action space —
e.g. the rocket scenario masks all hand-crafting actions so the agent
must use placed machines for production.

The mask is a frozen ``jnp.ndarray`` of shape ``(NUM_ACTIONS,)`` bool,
where ``True`` means the action is blocked. Blocked actions are
silently replaced with :data:`Action.NOOP`; nothing about the env is
aware of the wrapping beyond that substitution.

Kept JIT-friendly: the mask is passed as a closure variable, so the
wrapper's ``step_env`` is fully compilable alongside the inner env.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces  # type: ignore[import-untyped]

from factoriax.engine.constants import Action
from factoriax.engine.state import EnvParams, EnvState


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

    
    >>> from factoriax import ActionMaskWrapper, FactoriaXEnv, Action
        >>> env = ActionMaskWrapper(FactoriaXEnv(), blocked_actions=(int(Action.MINE),))
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
        """ """
        # gymnax's environment.Environment is untyped so the inner attribute
        # is Any; the runtime contract guarantees an EnvParams.
        params: EnvParams = self._inner.default_params
        return params

    def _rewrite(self, action: int | jax.Array) -> jax.Array:
        """

        Parameters
        ----------
        action : int | jax.Array :
            
        action: int | jax.Array :
            

        Returns
        -------

        
        """
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
        """

        Parameters
        ----------
        key : jax.Array :
            
        params : EnvParams :
            
        key: jax.Array :
            
        params: EnvParams :
            

        Returns
        -------

        
        """
        result: tuple[jax.Array, Any] = self._inner.reset_env(key, params)
        return result

    def get_obs(self, state: Any, params: EnvParams) -> jax.Array:
        """

        Parameters
        ----------
        state : Any :
            
        params : EnvParams :
            
        state: Any :
            
        params: EnvParams :
            

        Returns
        -------

        
        """
        obs: jax.Array = self._inner.get_obs(state, params)
        return obs

    def is_terminal(self, state: Any, params: EnvParams) -> jax.Array:
        """

        Parameters
        ----------
        state : Any :
            
        params : EnvParams :
            
        state: Any :
            
        params: EnvParams :
            

        Returns
        -------

        
        """
        terminal: jax.Array = self._inner.is_terminal(state, params)
        return terminal

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        """

        Parameters
        ----------
        params : EnvParams :
            
        params: EnvParams :
            

        Returns
        -------

        
        """
        return self._inner.action_space(params)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """

        Parameters
        ----------
        params : EnvParams :
            
        params: EnvParams :
            

        Returns
        -------

        
        """
        return self._inner.observation_space(params)
