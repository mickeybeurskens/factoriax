"""FactoriaX environment implementing the gymnax interface."""

from typing import Any

import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces

from factoriax.constants import NUM_ACTIONS, NUM_INVENTORY_SLOTS
from factoriax.game_logic import factoriax_step, is_game_over
from factoriax.levels import Level, build_state, generate_state
from factoriax.observations import NUM_PLAYER_SCALARS, global_array
from factoriax.renderer import render_pixels
from factoriax.rewards import achievement_reward
from factoriax.state import EnvParams, EnvState


class FactoriaXEnv(environment.Environment[EnvState, EnvParams]):  # type: ignore[misc]
    """FactoriaX JAX-based grid environment.

    A minimal environment where a player can move on a grid of dirt and water tiles.
    Water tiles block movement. The environment follows the gymnax interface for
    compatibility with JAX-based RL algorithms.
    """

    def __init__(self) -> None:
        """Initialize the environment."""
        super().__init__()

    @property
    def default_params(self) -> EnvParams:
        """Return default environment parameters.

        Returns:
            Default EnvParams instance
        """
        return EnvParams()

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[str, Any]]:
        """Execute one environment step.

        Args:
            key: JAX random key
            state: Current environment state
            action: Action to take
            params: Environment parameters

        Returns:
            Tuple of (observation, new_state, reward, done, info)
        """
        action_arr = jnp.int32(action)
        prev_state = state
        new_state = factoriax_step(key, prev_state, action_arr, params)
        reward = achievement_reward(prev_state, new_state, params)
        done = is_game_over(new_state, params)
        obs = self.get_obs(new_state, params)
        info: dict[str, Any] = {}
        return obs, new_state, jnp.float32(reward), done, info

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        """Reset the environment to an initial state.

        Args:
            key: JAX random key for world generation
            params: Environment parameters

        Returns:
            Tuple of (initial_observation, initial_state)
        """
        state = generate_state(key, params)
        obs = self.get_obs(state, params)
        return obs, state

    def reset_from_level(
        self, level: Level, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        """Reset the environment to a pre-built level.

        Players are placed near the centre of the map according to
        ``params.num_players``.  No random key is needed because pre-built
        levels are deterministic.

        Args:
            level: Level definition.  Its dimensions must match those in
                ``params``.
            params: Environment parameters, including ``num_players``.

        Returns:
            Tuple of ``(initial_observation, initial_state)``.
        """
        state = build_state(level, params)
        obs = self.get_obs(state, params)
        return obs, state

    def get_obs(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Get observation from the current state for the selected player.

        Delegates to :func:`factoriax.observations.global_array` using
        ``state.selected_player`` as the player index.

        Args:
            state: Current environment state.
            params: Environment parameters.

        Returns:
            Float32 array of shape
            ``(map_h * map_w + NUM_PLAYER_SCALARS + 2 * NUM_INVENTORY_SLOTS,)``.
        """
        return global_array(state, params, state.selected_player)

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Check if the current state is terminal.

        Args:
            state: Current environment state
            params: Environment parameters

        Returns:
            Boolean indicating whether state is terminal
        """
        return is_game_over(state, params)

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        """Return the action space.

        Args:
            params: Environment parameters

        Returns:
            Discrete action space with 5 actions (NOOP, LEFT, RIGHT, UP, DOWN)
        """
        return spaces.Discrete(NUM_ACTIONS)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Return the observation space.

        Args:
            params: Environment parameters

        Returns:
            Box observation space matching the flattened observation
        """
        obs_size = (
            params.map_width * params.map_height
            + NUM_PLAYER_SCALARS
            + NUM_INVENTORY_SLOTS * 2
        )
        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_size,),
            dtype=jnp.float32,
        )

    def render(self, state: EnvState) -> jax.Array:
        """Render the environment state as pixels.

        Args:
            state: Current environment state

        Returns:
            RGB pixel array of the rendered scene
        """
        return jnp.array(render_pixels(state))


def make_factoriax_env() -> tuple[FactoriaXEnv, EnvParams]:
    """Create a FactoriaX environment with default parameters.

    Returns:
        Tuple of (environment, default_params)
    """
    env = FactoriaXEnv()
    params = env.default_params
    return env, params
