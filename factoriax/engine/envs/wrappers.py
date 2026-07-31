"""The wrappers that compose over a Factoriax environment."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces  # type: ignore[import-untyped]

from factoriax.engine.constants import Action
from factoriax.engine.envs.base import FactoriaxEnv
from factoriax.engine.state import EnvParams, EnvState

__all__ = [
    "AutoResetState",
    "AutoResetWrapper",
    "ActionMaskWrapper",
    "LogEnvState",
    "LogWrapper",
]


# ---------------------------------------------------------------------------
# Auto-reset wrapper
# ---------------------------------------------------------------------------


class AutoResetState(struct.PyTreeNode):  # type: ignore[no-untyped-call]
    """State that holds a copy of the start world, for a cheap auto-reset.

    The class keeps a frozen copy of the start state next to the live state. At
    the end of an episode, ``lax.select`` puts the copy back. It does not call
    ``reset_env``, which runs the whole terrain generation.

    The cost is one extra copy of ``EnvState`` for each batch element. At 32x32
    that is about 12 KB for each element. At 128x128 with
    ``max_machines=4096`` it grows to about 185 KB for each element, which
    doubles the total state memory. On a large map, lower ``batch_size`` or
    ``max_machines``.
    """

    env_state: EnvState
    reset_state: EnvState


class AutoResetWrapper(environment.Environment[AutoResetState, EnvParams]):  # type: ignore[misc]
    """A gymnax wrapper that auto-resets from a cached state.

    Use this wrapper when you need an auto-reset inside a ``lax.scan`` training
    loop, such as a PPO loop in the PureJaxRL style. To manage the episodes
    yourself, use :class:`FactoriaxEnv` directly.

    Parameters
    ----------
    inner
        Core Factoriax environment to wrap.
    resample
        Whether a new episode builds a new world. See :meth:`__init__`.
    """

    def __init__(self, inner: FactoriaxEnv, resample: bool = False) -> None:
        """Initialize the auto-reset wrapper.

        Parameters
        ----------
        inner
            Core environment instance.
        resample
            Whether each episode gets a new world. ``False`` puts the cached
            start state back. That is cheap, but it repeats one layout for the
            whole run, and an agent can overfit to it. ``True`` calls
            ``inner.reset_env`` with a new key. That runs the world generation
            in every step, and not only in the steps that end an episode,
            because both sides of the select always run. Use ``True`` only
            with a cheap generator.
        """
        super().__init__()
        self._inner = inner
        self._resample = resample

    @property
    def default_params(self) -> EnvParams:
        """Return the default parameters of the inner environment."""
        return self._inner.default_params

    @property
    def map_width(self) -> int:
        """Return the map width in tiles, from the inner environment."""
        return self._inner.map_width

    @property
    def map_height(self) -> int:
        """Return the map height in tiles, from the inner environment."""
        return self._inner.map_height

    @property
    def num_players(self) -> int:
        """Return the number of players, from the inner environment."""
        return self._inner.num_players  # type: ignore[no-any-return]

    @property
    def obs(self) -> str:
        """Return the observation variant key, from the inner environment."""
        return self._inner.obs

    @property
    def obs_radius(self) -> int:
        """Return the half-width of the local window, from the inner environment."""
        return self._inner.obs_radius

    def step_env(
        self,
        key: jax.Array,
        state: AutoResetState,
        action: int | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, AutoResetState, jax.Array, jax.Array, dict[str, Any]]:
        """Step the environment, and auto-reset from the cache at the end.

        When ``done`` is True, ``lax.select`` puts the cached reset state in
        place of the live state. No terrain generation runs.

        Parameters
        ----------
        key
            PRNG key. The method splits it into a step key and a reset key, so
            the reset branch draws a new world under ``resample``.
        state
            Wrapped state, which holds both the live world and the cached
            world.
        action
            Action for the selected player.
        params
            Environment parameters.

        Returns
        -------
        tuple
            ``(obs, new_state, reward, done, info)``. When ``done`` is True,
            the state in the result is already the first state of the next
            episode, and ``reward`` and ``done`` still describe the step that
            ended. Both branches run in every step, because
            :func:`jax.lax.select` evaluates both.
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
        """Reset, and cache the start state for the later auto-resets.

        Parameters
        ----------
        key
            PRNG key for the world generation.
        params
            Environment parameters.

        Returns
        -------
        tuple
            ``(obs, state)``. The method takes the cached copy here and never
            replaces it. With ``resample=False``, every episode of the run
            therefore repeats this exact world.
        """
        obs, env_state = self._inner.reset_env(key, params)
        state = AutoResetState(
            env_state=env_state,
            reset_state=env_state,
        )
        return obs, state

    def get_obs(self, state: AutoResetState, params: EnvParams) -> jax.Array:
        """Return the observation of the inner environment.

        Parameters
        ----------
        state
            Wrapped state.
        params
            Environment parameters.

        Returns
        -------
        jax.Array
            The observation that the inner environment makes of the live
            state, and not of the cached state.
        """
        return self._inner.get_obs(state.env_state, params)

    def is_terminal(self, state: AutoResetState, params: EnvParams) -> jax.Array:
        """Ask the inner environment whether the episode is at its end.

        Parameters
        ----------
        state
            Wrapped state.
        params
            Environment parameters.

        Returns
        -------
        jax.Array
            Scalar bool from the inner environment.
        """
        return self._inner.is_terminal(state.env_state, params)

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        """Return the action space of the inner environment, unchanged.

        Parameters
        ----------
        params
            The method does not read this argument. It is present for the
            gymnax signature.

        Returns
        -------
        spaces.Discrete
            The action space of the inner environment, unchanged.
        """
        return self._inner.action_space(params)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Return the observation space of the inner environment, unchanged.

        Parameters
        ----------
        params
            The method does not read this argument. It is present for the
            gymnax signature.

        Returns
        -------
        spaces.Box
            The observation space of the inner environment, unchanged.
        """
        return self._inner.observation_space(params)


# ---------------------------------------------------------------------------
# Action-mask wrapper
# ---------------------------------------------------------------------------


class ActionMaskWrapper(environment.Environment[EnvState, EnvParams]):  # type: ignore[misc]
    """Replace a blocked action with :data:`Action.NOOP`.

    The wrapper rewrites a blocked action and does not remove it. The action
    stays in the action space and still costs a timestep. An agent that sends
    one therefore loses a turn, and nothing stops it from choosing that action.

    Parameters
    ----------
    inner
        Environment to wrap. Any gymnax-compatible environment with the state
        type :class:`EnvState` works.
    blocked_actions
        ``Action`` integers to block. The constructor captures them and writes
        them into the JIT graph of :meth:`step_env`. A different set therefore
        needs a new environment and a new compile.
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
        """Return the default parameters of the inner environment."""
        params: EnvParams = self._inner.default_params
        return params

    @property
    def map_width(self) -> int:
        """Return the map width in tiles, from the inner environment."""
        return self._inner.map_width  # type: ignore[no-any-return]

    @property
    def map_height(self) -> int:
        """Return the map height in tiles, from the inner environment."""
        return self._inner.map_height  # type: ignore[no-any-return]

    @property
    def num_players(self) -> int:
        """Return the number of players, from the inner environment."""
        return self._inner.num_players  # type: ignore[no-any-return]

    def _rewrite(self, action: int | jax.Array) -> jax.Array:
        """Rewrite a blocked action to NOOP, and return any other action as it is."""
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
        """Rewrite a blocked action to NOOP, then step the inner environment.

        Parameters
        ----------
        key
            PRNG key. The method passes it to the inner environment.
        state
            State to advance.
        action
            Action to take. The method rewrites it to NOOP when it is blocked.
        params
            Environment parameters.

        Returns
        -------
        tuple
            ``(obs, new_state, reward, done, info)`` from the inner
            environment, after the rewrite. A blocked action still costs a
            timestep.
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
        """Reset the inner environment and return its result.

        Parameters
        ----------
        key
            PRNG key for the world generation.
        params
            Environment parameters.

        Returns
        -------
        tuple
            ``(obs, state)`` from the inner environment.
        """
        result: tuple[jax.Array, Any] = self._inner.reset_env(key, params)
        return result

    def get_obs(self, state: Any, params: EnvParams) -> jax.Array:
        """Return the observation of the inner environment.

        Parameters
        ----------
        state
            State to observe.
        params
            Environment parameters.

        Returns
        -------
        jax.Array
            The observation of the inner environment.
        """
        obs: jax.Array = self._inner.get_obs(state, params)
        return obs

    def is_terminal(self, state: Any, params: EnvParams) -> jax.Array:
        """Ask the inner environment whether the episode is at its end.

        Parameters
        ----------
        state
            State to test.
        params
            Environment parameters.

        Returns
        -------
        jax.Array
            Scalar bool from the inner environment.
        """
        terminal: jax.Array = self._inner.is_terminal(state, params)
        return terminal

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        """Return the action space of the inner environment, unchanged.

        Parameters
        ----------
        params
            The method does not read this argument. It is present for the
            gymnax signature.

        Returns
        -------
        spaces.Discrete
            The action space of the inner environment, unchanged. A blocked
            action stays in the space, and the wrapper rewrites it instead of
            removing it. An agent can therefore still send it, and the output
            width of a policy does not depend on the mask.
        """
        return self._inner.action_space(params)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Return the observation space of the inner environment, unchanged.

        Parameters
        ----------
        params
            The method does not read this argument. It is present for the
            gymnax signature.

        Returns
        -------
        spaces.Box
            The observation space of the inner environment, unchanged.
        """
        return self._inner.observation_space(params)


# ---------------------------------------------------------------------------
# Episode-logging wrapper
# ---------------------------------------------------------------------------


class LogEnvState(struct.PyTreeNode):  # type: ignore[no-untyped-call]
    """State that totals the return and the length of each episode, for the log.

    Every scalar field is a JAX array, so the state is safe under vmap and
    under scan. The ``returned_*`` fields hold the numbers of the last finished
    episode. They are non-zero only on the step where ``done`` becomes True.

    Attributes
    ----------
    env_state :
        State of the inner wrapper. It can be any JAX pytree.
    episode_returns :
        Running return of the current episode. It returns to 0 on ``done``.
    episode_lengths :
        Running step count of the current episode. It returns to 0 on ``done``.
    returned_episode_returns :
        Return of the episode that just finished. It is 0 while an episode
        runs.
    returned_episode_lengths :
        Length of the episode that just finished. It is 0 while an episode
        runs.
    timestep :
        Global step counter. Nothing resets it.
    """

    env_state: Any
    episode_returns: jnp.ndarray
    episode_lengths: jnp.ndarray
    returned_episode_returns: jnp.ndarray
    returned_episode_lengths: jnp.ndarray
    timestep: jnp.ndarray


class LogWrapper(environment.Environment[LogEnvState, EnvParams]):  # type: ignore[misc]
    """Add the numbers of each episode to the ``info`` dict in every step.

    This wrapper takes any gymnax-compatible environment. Every call to
    :meth:`step_env` adds these keys to the ``info`` dict:

    - ``"returned_episode_returns"``, the return of the episode that finished
      last. It is non-zero only on the final step of an episode.
    - ``"returned_episode_lengths"``, the length of the episode that finished
      last. It is non-zero only on the final step of an episode.
    - ``"returned_episode"``, which is ``True`` on the step where an episode
      ends.
    - ``"timestep"``, the global step counter, which rises with every call.

    These keys follow the ``LogWrapper`` convention of PureJaxRL. A training
    loop can therefore read the episode numbers directly from
    ``traj_batch.info``, which ``jax.lax.scan`` returns.

    Examples
    --------
    >>> import factoriax
    >>> env, params = factoriax.make("Mining-v1", auto_reset=True)
    >>> from factoriax.engine.envs.wrappers import LogWrapper
    >>> log_env = LogWrapper(env)
    >>> # log_env.step_env(...) now returns info["returned_episode_returns"]
    """

    def __init__(self, inner: environment.Environment[Any, EnvParams]) -> None:
        super().__init__()
        self._inner = inner

    @property
    def default_params(self) -> EnvParams:
        """Return the default parameters of the inner environment."""
        return self._inner.default_params  # type: ignore[return-value]

    @property
    def map_width(self) -> int:
        """Return the map width in tiles, from the inner environment."""
        return self._inner.map_width  # type: ignore[no-any-return]

    @property
    def map_height(self) -> int:
        """Return the map height in tiles, from the inner environment."""
        return self._inner.map_height  # type: ignore[no-any-return]

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, LogEnvState]:
        """Reset the inner environment, and set every episode counter to zero.

        Parameters
        ----------
        key
            PRNG key for the world generation.
        params
            Environment parameters.

        Returns
        -------
        tuple
            ``(obs, state)``, with the running episode counters at zero.
        """
        obs, env_state = self._inner.reset_env(key, params)
        state = LogEnvState(
            env_state=env_state,
            episode_returns=jnp.float32(0.0),
            episode_lengths=jnp.int32(0),
            returned_episode_returns=jnp.float32(0.0),
            returned_episode_lengths=jnp.int32(0),
            timestep=jnp.int32(0),
        )
        return obs, state

    def step_env(
        self,
        key: jax.Array,
        state: LogEnvState,
        action: int | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, LogEnvState, jax.Array, jax.Array, dict[str, Any]]:
        """Step the inner environment, and update the episode counters.

        On the step where ``done`` is True:

        - ``returned_episode_returns`` takes the return that just finished.
        - ``returned_episode_lengths`` takes the length that just finished.
        - ``episode_returns`` and ``episode_lengths`` return to zero.

        Every info key is always present. Use ``info["returned_episode"]`` as a
        boolean mask to select the valid ``returned_episode_returns`` entries.

        Parameters
        ----------
        key
            PRNG key. The method passes it to the inner environment.
        state
            Wrapped state, which carries the running counters.
        action
            Action for the selected player.
        params
            Environment parameters.

        Returns
        -------
        tuple
            ``(obs, new_state, reward, done, info)``. ``info`` carries the
            return and the length of the episode. Those two have meaning only
            on the step where ``done`` is True.
        """
        obs, env_state, reward, done, info = self._inner.step_env(
            key, state.env_state, action, params
        )
        done_f = done.astype(jnp.float32)
        done_i = done.astype(jnp.int32)
        new_return = state.episode_returns + reward
        new_length = state.episode_lengths + jnp.int32(1)
        new_state = state.replace(
            env_state=env_state,
            episode_returns=new_return * (jnp.float32(1.0) - done_f),
            episode_lengths=new_length * (jnp.int32(1) - done_i),
            returned_episode_returns=(
                state.returned_episode_returns * (jnp.float32(1.0) - done_f)
                + new_return * done_f
            ),
            returned_episode_lengths=(
                state.returned_episode_lengths * (jnp.int32(1) - done_i)
                + new_length * done_i
            ),
            timestep=state.timestep + jnp.int32(1),
        )
        info["returned_episode_returns"] = new_state.returned_episode_returns
        info["returned_episode_lengths"] = new_state.returned_episode_lengths
        info["returned_episode"] = done
        info["timestep"] = new_state.timestep
        return obs, new_state, reward, done, info

    def get_obs(self, state: LogEnvState, params: EnvParams) -> jax.Array:
        """Return the observation of the inner environment.

        Parameters
        ----------
        state
            Wrapped state.
        params
            Environment parameters.

        Returns
        -------
        jax.Array
            The observation of the inner environment.
        """
        return self._inner.get_obs(state.env_state, params)

    def is_terminal(self, state: LogEnvState, params: EnvParams) -> jax.Array:
        """Ask the inner environment whether the episode is at its end.

        Parameters
        ----------
        state
            Wrapped state.
        params
            Environment parameters.

        Returns
        -------
        jax.Array
            Scalar bool from the inner environment.
        """
        return self._inner.is_terminal(state.env_state, params)

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        """Return the action space of the inner environment, unchanged.

        Parameters
        ----------
        params
            The method does not read this argument. It is present for the
            gymnax signature.

        Returns
        -------
        spaces.Discrete
            The action space of the inner environment, unchanged.
        """
        return self._inner.action_space(params)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Return the observation space of the inner environment, unchanged.

        Parameters
        ----------
        params
            The method does not read this argument. It is present for the
            gymnax signature.

        Returns
        -------
        spaces.Box
            The observation space of the inner environment, unchanged.
        """
        return self._inner.observation_space(params)
