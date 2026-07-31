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
    inner
        Core FactoriaX environment to wrap.
    resample
        Whether a new episode regenerates its world. See :meth:`__init__`.
    """

    def __init__(self, inner: FactoriaxEnv, resample: bool = False) -> None:
        """Initialize the auto-reset wrapper.

        Parameters
        ----------
        inner
            Core environment instance.
        resample
            Whether each episode gets a fresh world. ``False`` restores the
            cached initial state, which is cheap but replays one layout for
            the whole run, so an agent can overfit to it. ``True`` calls
            ``inner.reset_env`` with a new key. That runs world generation
            on every step rather than only on the ones that terminate,
            because both sides of the select are always evaluated, so it is
            worth it only for cheap generators.
        """
        super().__init__()
        self._inner = inner
        self._resample = resample

    @property
    def default_params(self) -> EnvParams:
        """Delegate to the inner env's default params."""
        return self._inner.default_params

    @property
    def map_width(self) -> int:
        """Map width in tiles, from the inner env."""
        return self._inner.map_width

    @property
    def map_height(self) -> int:
        """Map height in tiles, from the inner env."""
        return self._inner.map_height

    @property
    def num_players(self) -> int:
        """Player count, from the inner env."""
        return self._inner.num_players  # type: ignore[no-any-return]

    @property
    def obs(self) -> str:
        """Observation variant key, from the inner env."""
        return self._inner.obs

    @property
    def obs_radius(self) -> int:
        """Local-window half-width, from the inner env."""
        return self._inner.obs_radius

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
        key
            PRNG key. Split into a step key and a reset key, so the
            reset branch draws a fresh world under ``resample``.
        state
            Wrapped state holding both the live and cached worlds.
        action
            Action for the selected player.
        params
            Environment parameters.

        Returns
        -------
        tuple
            ``(obs, new_state, reward, done, info)``. When ``done`` is True
            the returned state is already the next episode's first state,
            while ``reward`` and ``done`` still describe the step that
            ended. Both branches run every step: :func:`jax.lax.select`
            does not short circuit.
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
        key
            PRNG key for world generation.
        params
            Environment parameters.

        Returns
        -------
        tuple
            ``(obs, state)``. The cached copy is taken here and never
            refreshed, so under ``resample=False`` every episode in the run
            replays this exact world.
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
        state
            Wrapped state.
        params
            Environment parameters.

        Returns
        -------
        jax.Array
            The inner env's observation of the live state, not the cached
            one.
        """
        return self._inner.get_obs(state.env_state, params)

    def is_terminal(self, state: AutoResetState, params: EnvParams) -> jax.Array:
        """Delegate termination to the inner env.

        Parameters
        ----------
        state
            Wrapped state.
        params
            Environment parameters.

        Returns
        -------
        jax.Array
            Scalar bool from the inner env.
        """
        return self._inner.is_terminal(state.env_state, params)

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        """Return the inner env's action space, unchanged.

        Parameters
        ----------
        params
            Unused. Present for the gymnax signature.

        Returns
        -------
        spaces.Discrete
            The inner env's action space, unchanged.
        """
        return self._inner.action_space(params)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Return the inner env's observation space, unchanged.

        Parameters
        ----------
        params
            Unused. Present for the gymnax signature.

        Returns
        -------
        spaces.Box
            The inner env's observation space, unchanged.
        """
        return self._inner.observation_space(params)


# ---------------------------------------------------------------------------
# Action-mask wrapper
# ---------------------------------------------------------------------------


class ActionMaskWrapper(environment.Environment[EnvState, EnvParams]):  # type: ignore[misc]
    """Replace blocked actions with :data:`Action.NOOP`.

    Parameters
    ----------
    A blocked action is rewritten, not removed. It stays in the action
    space and still costs a timestep, so an agent that emits one loses a
    turn rather than being prevented from choosing it.

    Parameters
    ----------
    inner
        Environment to wrap. Any gymnax-compatible env whose state type is
        :class:`EnvState` works.
    blocked_actions
        ``Action`` integers to block. Captured at construction and baked
        into the JIT graph of :meth:`step_env`, so changing the set means a
        new env and a new compile.
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
        params: EnvParams = self._inner.default_params
        return params

    @property
    def map_width(self) -> int:
        """Map width in tiles, from the inner env."""
        return self._inner.map_width  # type: ignore[no-any-return]

    @property
    def map_height(self) -> int:
        """Map height in tiles, from the inner env."""
        return self._inner.map_height  # type: ignore[no-any-return]

    @property
    def num_players(self) -> int:
        """Player count, from the inner env."""
        return self._inner.num_players  # type: ignore[no-any-return]

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
        key
            PRNG key, passed to the inner env.
        state
            State to advance.
        action
            Action to take. Rewritten to NOOP when blocked.
        params
            Environment parameters.

        Returns
        -------
        tuple
            ``(obs, new_state, reward, done, info)`` from the inner env,
            after the action was rewritten. A blocked action still costs a
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
        """Pass-through reset to the inner env.

        Parameters
        ----------
        key
            PRNG key for world generation.
        params
            Environment parameters.

        Returns
        -------
        tuple
            ``(obs, state)`` from the inner env.
        """
        result: tuple[jax.Array, Any] = self._inner.reset_env(key, params)
        return result

    def get_obs(self, state: Any, params: EnvParams) -> jax.Array:
        """Pass-through observation from the inner env.

        Parameters
        ----------
        state
            State to observe.
        params
            Environment parameters.

        Returns
        -------
        jax.Array
            The inner env's observation.
        """
        obs: jax.Array = self._inner.get_obs(state, params)
        return obs

    def is_terminal(self, state: Any, params: EnvParams) -> jax.Array:
        """Pass-through termination check to the inner env.

        Parameters
        ----------
        state
            State to test.
        params
            Environment parameters.

        Returns
        -------
        jax.Array
            Scalar bool from the inner env.
        """
        terminal: jax.Array = self._inner.is_terminal(state, params)
        return terminal

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        """Return the inner env's action space, unchanged.

        Parameters
        ----------
        params
            Unused. Present for the gymnax signature.

        Returns
        -------
        spaces.Discrete
            The inner env's action space, unchanged. Blocked actions stay
            in the space and are rewritten rather than removed, so an
            agent can still emit them and a policy's output width does not
            depend on the mask.
        """
        return self._inner.action_space(params)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Return the inner env's observation space, unchanged.

        Parameters
        ----------
        params
            Unused. Present for the gymnax signature.

        Returns
        -------
        spaces.Box
            The inner env's observation space, unchanged.
        """
        return self._inner.observation_space(params)


# ---------------------------------------------------------------------------
# Episode-logging wrapper
# ---------------------------------------------------------------------------


class LogEnvState(struct.PyTreeNode):  # type: ignore[no-untyped-call]
    """State that accumulates per-episode return and length for logging.

    All scalar fields are JAX arrays so the state is vmap- and scan-safe.
    ``returned_*`` hold the last *completed* episode's stats; they are
    non-zero only on the step where ``done`` fires.

    Parameters
    ----------
    env_state :
        Inner wrapper's state (any JAX pytree).
    episode_returns :
        Running return for the current episode (reset to 0 on done).
    episode_lengths :
        Running step count for the current episode (reset to 0 on done).
    returned_episode_returns :
        Return of the episode that just completed (0 while in-progress).
    returned_episode_lengths :
        Length of the episode that just completed (0 while in-progress).
    timestep :
        Global step counter (never reset).
    """

    env_state: Any
    episode_returns: jnp.ndarray
    episode_lengths: jnp.ndarray
    returned_episode_returns: jnp.ndarray
    returned_episode_lengths: jnp.ndarray
    timestep: jnp.ndarray


class LogWrapper(environment.Environment[LogEnvState, EnvParams]):  # type: ignore[misc]
    """Inject per-episode statistics into the ``info`` dict on every step.

    Wraps any gymnax-compatible env and augments the step ``info`` dict
    with the following keys on every call to :meth:`step_env`:

    - ``"returned_episode_returns"`` — return of the most recently completed
      episode (non-zero only on the terminal step).
    - ``"returned_episode_lengths"`` — length of the most recently completed
      episode (non-zero only on the terminal step).
    - ``"returned_episode"`` — ``True`` on the step an episode ends.
    - ``"timestep"`` — global step counter, incremented every call.

    These keys mirror the PureJaxRL ``LogWrapper`` convention so downstream
    training loops can extract episode stats directly from ``traj_batch.info``
    returned by ``jax.lax.scan``.

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
        """Delegate to the inner env's default params."""
        return self._inner.default_params  # type: ignore[return-value]

    @property
    def map_width(self) -> int:
        """Map width in tiles, from the inner env."""
        return self._inner.map_width  # type: ignore[no-any-return]

    @property
    def map_height(self) -> int:
        """Map height in tiles, from the inner env."""
        return self._inner.map_height  # type: ignore[no-any-return]

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, LogEnvState]:
        """Reset the inner env and initialise all episode counters to zero.

        Parameters
        ----------
        key
            PRNG key for world generation.
        params
            Environment parameters.

        Returns
        -------
        tuple
            ``(obs, state)`` with the running episode counters zeroed.
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
        """Step the inner env and update episode accumulators.

        On termination (``done=True``):
        - ``returned_episode_returns`` captures the just-completed return.
        - ``returned_episode_lengths`` captures the just-completed length.
        - ``episode_returns`` and ``episode_lengths`` reset to zero.

        All info keys are always present; use ``info["returned_episode"]``
        as a boolean mask to select valid ``returned_episode_returns`` entries.

        Parameters
        ----------
        key
            PRNG key, passed to the inner env.
        state
            Wrapped state carrying the running counters.
        action
            Action for the selected player.
        params
            Environment parameters.

        Returns
        -------
        tuple
            ``(obs, new_state, reward, done, info)``. ``info`` carries the
            episode return and length, which are only meaningful on the
            step where ``done`` is True.
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
        """Pass-through observation from the inner env.

        Parameters
        ----------
        state
            Wrapped state.
        params
            Environment parameters.

        Returns
        -------
        jax.Array
            The inner env's observation.
        """
        return self._inner.get_obs(state.env_state, params)

    def is_terminal(self, state: LogEnvState, params: EnvParams) -> jax.Array:
        """Pass-through termination check to the inner env.

        Parameters
        ----------
        state
            Wrapped state.
        params
            Environment parameters.

        Returns
        -------
        jax.Array
            Scalar bool from the inner env.
        """
        return self._inner.is_terminal(state.env_state, params)

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        """Return the inner env's action space, unchanged.

        Parameters
        ----------
        params
            Unused. Present for the gymnax signature.

        Returns
        -------
        spaces.Discrete
            The inner env's action space, unchanged.
        """
        return self._inner.action_space(params)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Return the inner env's observation space, unchanged.

        Parameters
        ----------
        params
            Unused. Present for the gymnax signature.

        Returns
        -------
        spaces.Box
            The inner env's observation space, unchanged.
        """
        return self._inner.observation_space(params)
