"""FactoriaX environment implementing the gymnax interface."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces  # type: ignore[import-untyped]

from factoriax.engine.constants import NUM_ACTIONS
from factoriax.engine.envs.hooks import StepHook, achievement_hook
from factoriax.engine.game_logic import factoriax_step, is_game_over
from factoriax.engine.levels import Level, build_state, generate_state
from factoriax.engine.observations import (
    NUM_PLAYER_SCALARS,
    NUM_SPATIAL_CHANNELS,
    OBSERVATIONS,
)
from factoriax.engine.state import EnvParams, EnvState

AchievementFn = Callable[[EnvState], jax.Array]
#: World generator: ``(key, params) -> EnvState``. Overrides the level/procgen
#: branch in :meth:`FactoriaXEnv.reset_env` when supplied.
ResetFn = Callable[[jax.Array, EnvParams], EnvState]
#: Per-step reward: ``(prev_state, new_state, params) -> float``.
RewardFn = Callable[[EnvState, EnvState, EnvParams], jax.Array]


class FactoriaXEnv(environment.Environment[EnvState, EnvParams]):  # type: ignore[misc]
    """FactoriaX JAX-based grid environment.
    
    Advances world state (terrain, machines, player inventories) and
    evaluates an optional achievement-condition function each step.
    Reward computation and other policy-shaping concerns still belong
    in gymnax wrappers that compose over this environment. Pixel
    rendering is provided by :class:`factoriax.engine.jax_renderer.JaxRenderer`;
    the env itself does not expose a render method.
    
    The ``achievement_fn`` parameter is captured at construction time
    and folded into ``state.achievements_unlocked`` inside
    :meth:`step_env`. Different functions produce different JIT
    cache entries via ``static_argnames=("self",)`` on :meth:`step`.
    Pass ``None`` to skip the eval pass entirely (zero added cost).
    
    Reset behavior is selected by ``reset_fn`` then ``level``: when
    ``reset_fn`` is supplied, :meth:`reset_env` calls it with the PRNG key
    (the scenario owns world generation); otherwise a ``None`` ``level``
    means procedural generation from the key, and a supplied
    :class:`~factoriax.engine.levels.Level` is materialized via
    :func:`~factoriax.engine.levels.build_state` (key unused for layout).
    gymnax conformance — ``reset_env(key, params)`` is the only reset
    surface.

    Parameters
    ----------
    achievement_fn :
        Pure function ``(EnvState) -> bool[MAX_ACHIEVEMENTS]``
        evaluated each step. Returned True bits are OR-folded into
        ``state.achievements_unlocked`` and latch for the rest of
        the episode. ``None`` (default) skips evaluation.
    level :
        Fixed :class:`Level` used by :meth:`reset_env`. ``None``
        (default) means procedural generation from the PRNG key.
        Examples
        --------

    Returns
    -------

    
    >>> import jax
        >>> from factoriax import FactoriaXEnv
        >>> env = FactoriaXEnv()
        >>> params = env.default_params
        >>> obs, state = env.reset_env(jax.random.PRNGKey(0), params)
        >>> obs.shape == env.observation_space(params).shape
        True
    """

    def __init__(
        self,
        achievement_fn: AchievementFn | None = None,
        level: Level | None = None,
        reset_fn: ResetFn | None = None,
        step_hooks: tuple[StepHook, ...] = (),
        reward_fn: RewardFn | None = None,
        obs: str = "x_ray_global",
        obs_radius: int = 7,
    ) -> None:
        """Initialize the environment.

        Parameters
        ----------
            achievement_fn: Optional achievement condition function. Convenience
                for ``step_hooks=(achievement_hook(achievement_fn),)``; when both
                are given the achievement fold runs after the explicit hooks.
            level: Fixed level for :meth:`reset_env`, or ``None`` for procedural
                generation. Ignored when ``reset_fn`` is supplied.
            reset_fn: World generator ``(key, params) -> EnvState``. When set,
                :meth:`reset_env` calls it instead of the level/procgen branch.
            step_hooks: Post-step transforms applied in order after
                :func:`factoriax_step`, each ``(key, state, params) -> state``.
            reward_fn: Per-step reward ``(prev, new, params) -> float``. When
                ``None``, :meth:`step_env` returns ``0.0`` (rewards then belong to
                a wrapper or the training loop).
            obs: Observation variant key into
                :data:`~factoriax.engine.observations.OBSERVATIONS`. One of
                ``"x_ray_global"``, ``"x_ray_local"``,
                ``"superficial_global"``, ``"superficial_local"``.
            obs_radius: Half-width of the local window; ignored for
                ``_global`` obs variants.
        """
        super().__init__()
        if obs not in OBSERVATIONS:
            raise ValueError(
                f"unknown obs {obs!r}; valid choices: {sorted(OBSERVATIONS)}"
            )
        self._achievement_fn = achievement_fn
        self._level = level
        self._reset_fn = reset_fn
        self._reward_fn = reward_fn
        hooks = tuple(step_hooks)
        if achievement_fn is not None:
            hooks = hooks + (achievement_hook(achievement_fn),)
        self._step_hooks = hooks
        self.obs = obs
        self.obs_radius = int(obs_radius)
        self._obs_profile = "superficial" if obs.startswith("superficial") else "x_ray"
        self._obs_is_local = obs.endswith("_local")

    @property
    def default_params(self) -> EnvParams:
        """ """
        if self._level is None:
            return EnvParams()
        return EnvParams(
            map_width=self._level.map_width,
            map_height=self._level.map_height,
        )

    @partial(jax.jit, static_argnames=("self",))
    def step(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | jax.Array,
        params: EnvParams | None = None,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[str, Any]]:
        """Step the environment without auto-reset.
        
        Overrides the gymnax base ``step()`` which unconditionally
        calls ``reset_env`` every tick for auto-reset. That design
        runs full procedural terrain generation on every step even
        when the episode is not done, roughly tripling the per-step
        cost. This override simply calls ``step_env`` directly.
        
        Use :class:`~factoriax.engine.envs.auto_reset_wrapper.AutoResetWrapper`
        if you need auto-reset for ``lax.scan`` training loops.
        
        Parameters
        ----------
            key: JAX random key.
            state: Current environment state.
            action: Action to take.

        Parameters
        ----------
        key : jax.Array :
            
        state : EnvState :
            
        action : int | jax.Array :
            
        params : EnvParams | None :
            (Default value = None)
        key: jax.Array :
            
        state: EnvState :
            
        action: int | jax.Array :
            
        params: EnvParams | None :
             (Default value = None)

        Returns
        -------

        
        """
        if params is None:
            params = self.default_params
        return self.step_env(key, state, action, params)

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[str, Any]]:
        """Execute one environment step.
        
        Runs game mechanics, then evaluates the optional achievement
        condition function and OR-folds the result into
        ``state.achievements_unlocked``. The eval pass is skipped
        entirely when no ``achievement_fn`` was provided at
        construction (the ``is None`` check is resolved at JIT trace
        time, so there is no per-step branch cost in that path).
        
        Parameters
        ----------
            key: JAX random key.
            state: Current environment state.
            action: Action to take.

        Parameters
        ----------
        key : jax.Array :
            
        state : EnvState :
            
        action : int | jax.Array :
            
        params : EnvParams :
            
        key: jax.Array :
            
        state: EnvState :
            
        action: int | jax.Array :
            
        params: EnvParams :
            

        Returns
        -------

        
        """
        action_arr = jnp.int32(action)
        new_state = factoriax_step(key, state, action_arr, params)
        for hook in self._step_hooks:
            new_state = hook(key, new_state, params)
        reward = (
            self._reward_fn(state, new_state, params)
            if self._reward_fn is not None
            else jnp.float32(0.0)
        )
        done = is_game_over(new_state, params)
        obs = self.get_obs(new_state, params)
        info: dict[str, Any] = {}
        return obs, new_state, reward, done, info

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        """Reset the environment to an initial state.
        
        Dispatches on the env's bound ``level``: when ``None``,
        procedurally generates a world from the PRNG key; when a
        :class:`Level` is bound, materializes that level via
        :func:`~factoriax.engine.levels.build_state` (the key is unused for
        layout).
        
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
        if self._reset_fn is not None:
            state = self._reset_fn(key, params)
        elif self._level is None:
            state = generate_state(key, params)
        else:
            state = build_state(self._level, params)
        obs = self.get_obs(state, params)
        return obs, state

    def get_obs(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Observation for the selected player via the configured variant.

        Parameters
        ----------
        state : EnvState :
            
        params : EnvParams :
            
        state: EnvState :
            
        params: EnvParams :
            

        Returns
        -------

        
        """
        fn = OBSERVATIONS[self.obs]
        if self._obs_is_local:
            return fn(state, params, state.selected_player, radius=self.obs_radius)
        return fn(state, params, state.selected_player)

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Whether the episode has ended.

        Parameters
        ----------
        state : EnvState :
            
        params : EnvParams :
            
        state: EnvState :
            
        params: EnvParams :
            

        Returns
        -------

        
        """
        return is_game_over(state, params)

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        """Discrete action space over all actions.

        Parameters
        ----------
        params : EnvParams :
            
        params: EnvParams :
            

        Returns
        -------

        
        """
        return spaces.Discrete(NUM_ACTIONS)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Box observation space sized for the configured obs variant.

        Parameters
        ----------
        params : EnvParams :
            
        params: EnvParams :
            

        Returns
        -------

        
        """
        if self._obs_is_local:
            side = 2 * self.obs_radius + 1
            tiles = side * side
        else:
            tiles = params.map_width * params.map_height
        obs_size = (
            NUM_SPATIAL_CHANNELS[self._obs_profile] * tiles
            + NUM_PLAYER_SCALARS[self._obs_profile]
        )
        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_size,),
            dtype=jnp.float32,
        )
