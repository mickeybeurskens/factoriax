"""FactoriaX environment implementing the gymnax interface."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces  # type: ignore[import-untyped]

from factoriax.constants import NUM_ACTIONS
from factoriax.game_logic import factoriax_step, is_game_over
from factoriax.jax_renderer import JaxRenderer
from factoriax.levels import Level, build_state, generate_state
from factoriax.observations import (
    NUM_PLAYER_SCALARS,
    NUM_SPATIAL_CHANNELS,
    global_array,
)
from factoriax.renderer import render_pixels
from factoriax.state import EnvParams, EnvState

AchievementFn = Callable[[EnvState], jax.Array]


class FactoriaXEnv(environment.Environment[EnvState, EnvParams]):  # type: ignore[misc]
    """FactoriaX JAX-based grid environment.

    Advances world state (terrain, machines, player inventories) and
    evaluates an optional achievement-condition function each step.
    Reward computation and other policy-shaping concerns still belong
    in gymnax wrappers that compose over this environment.

    The ``tile_px`` parameter controls the pixel size for the JAX
    renderer. A :class:`~factoriax.jax_renderer.JaxRenderer` is
    created at init and used by :meth:`render` and :meth:`render_hud`.

    The ``achievement_fn`` parameter is captured at construction time
    and folded into ``state.achievements_unlocked`` inside
    :meth:`step_env`. Different functions produce different JIT
    cache entries via ``static_argnames=("self",)`` on :meth:`step`.
    Pass ``None`` to skip the eval pass entirely (zero added cost).

    Args:
        tile_px: Tile side length in pixels for the JAX renderer.
        achievement_fn: Pure function ``(EnvState) -> bool[MAX_ACHIEVEMENTS]``
            evaluated each step. Returned True bits are OR-folded into
            ``state.achievements_unlocked`` and latch for the rest of
            the episode. ``None`` (default) skips evaluation.
    """

    def __init__(
        self,
        tile_px: int = 8,
        achievement_fn: AchievementFn | None = None,
    ) -> None:
        """Initialize the environment.

        Args:
            tile_px: Tile side length in pixels for the JAX renderer.
            achievement_fn: Optional achievement condition function.
        """
        super().__init__()
        self.jax_renderer = JaxRenderer(tile_px=tile_px)
        self._achievement_fn = achievement_fn

    @property
    def default_params(self) -> EnvParams:
        """Return default environment parameters.

        Returns:
            Default EnvParams instance.
        """
        return EnvParams()

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

        Use :class:`~factoriax.envs.achievement_wrapper.AutoResetWrapper`
        if you need auto-reset for ``lax.scan`` training loops.

        Args:
            key: JAX random key.
            state: Current environment state.
            action: Action to take.
            params: Environment parameters. Uses default when None.

        Returns:
            Tuple of (observation, new_state, reward, done, info).
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

        Args:
            key: JAX random key.
            state: Current environment state.
            action: Action to take.
            params: Environment parameters.

        Returns:
            Tuple of (observation, new_state, reward, done, info).
        """
        action_arr = jnp.int32(action)
        new_state = factoriax_step(key, state, action_arr, params)
        if self._achievement_fn is not None:
            conditions = self._achievement_fn(new_state)
            new_state = new_state.replace(
                achievements_unlocked=new_state.achievements_unlocked | conditions,
            )
        done = is_game_over(new_state, params)
        obs = self.get_obs(new_state, params)
        info: dict[str, Any] = {}
        return obs, new_state, jnp.float32(0.0), done, info

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        """Reset the environment to an initial state.

        Args:
            key: JAX random key for world generation.
            params: Environment parameters.

        Returns:
            Tuple of (initial_observation, initial_state).
        """
        state = generate_state(key, params)
        obs = self.get_obs(state, params)
        return obs, state

    def reset_from_level(
        self, level: Level, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        """Reset the environment to a pre-built level.

        Args:
            level: Level definition.
            params: Environment parameters.

        Returns:
            Tuple of (initial_observation, initial_state).
        """
        state = build_state(level, params)
        obs = self.get_obs(state, params)
        return obs, state

    def get_obs(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Get observation for the selected player.

        Args:
            state: Current environment state.
            params: Environment parameters.

        Returns:
            Float32 observation array.
        """
        return global_array(state, params, state.selected_player)

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Check if the current state is terminal.

        Args:
            state: Current environment state.
            params: Environment parameters.

        Returns:
            Boolean indicating whether state is terminal.
        """
        return is_game_over(state, params)

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        """Return the action space.

        Args:
            params: Environment parameters.

        Returns:
            Discrete action space.
        """
        return spaces.Discrete(NUM_ACTIONS)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Return the observation space.

        Args:
            params: Environment parameters.

        Returns:
            Box observation space.
        """
        obs_size = (
            NUM_SPATIAL_CHANNELS * params.map_width * params.map_height
            + NUM_PLAYER_SCALARS
        )
        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_size,),
            dtype=jnp.float32,
        )

    def render(self, state: EnvState) -> jax.Array:
        """Render the map using the JAX renderer.

        Returns a JIT-compiled, GPU-accelerated pixel render of the
        map with terrain, machines, and players. For the full HUD
        (including inventory, inspector, crafting), use :meth:`render_hud`.

        Args:
            state: Current environment state (single, non-batched).

        Returns:
            uint8 JAX array of shape (H * tile_px, W * tile_px, 3).
        """
        return self.jax_renderer.jit_render_map(state)

    def render_hud(self, state: EnvState) -> jax.Array:
        """Render the map + full HUD using the JAX renderer.

        Returns a JIT-compiled, GPU-accelerated pixel render with the
        map on top and a 4-quadrant info panel below (tile inspector,
        machine inventory, player inventory, crafting menu).

        Args:
            state: Current environment state (single, non-batched).

        Returns:
            uint8 JAX array of shape (2 * H * tile_px, W * tile_px, 3).
        """
        return self.jax_renderer.jit_render_hud(state)

    def render_cpu(self, state: EnvState) -> jax.Array:
        """Render using the legacy NumPy CPU renderer.

        For backward compatibility with the editor and play modes.
        Cannot be JIT-compiled or vmapped.

        Args:
            state: Current environment state.

        Returns:
            RGB pixel array (NumPy, wrapped in JAX).
        """
        return jnp.array(render_pixels(state))


def make_factoriax_env() -> tuple[FactoriaXEnv, EnvParams]:
    """Create a FactoriaX environment.

    Returns:
        Tuple of (environment, default_params).
    """
    env = FactoriaXEnv()
    params = env.default_params
    return env, params
