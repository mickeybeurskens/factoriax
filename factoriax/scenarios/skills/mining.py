"""Mining skill: dense reward for ore extraction.

The agent navigates to ore tiles and mines them. Reward is the number
of ore items gained each step, making this a dense signal that
directly tracks the agent's mining productivity.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from gymnax.environments import environment, spaces  # type: ignore[import-untyped]

from factoriax.constants import NUM_ACTIONS, BlockType
from factoriax.envs.factoriax_env import FactoriaXEnv
from factoriax.levels import Level, LevelBuilder
from factoriax.observations import NUM_PLAYER_SCALARS, NUM_SPATIAL_CHANNELS
from factoriax.state import EnvParams, EnvState

_ORE_TYPES = [BlockType.IRON, BlockType.COPPER, BlockType.COAL]


class MiningSkill(environment.Environment[EnvState, EnvParams]):  # type: ignore[misc]
    """Gymnax environment rewarding ore extraction.

    Wraps :class:`~factoriax.envs.factoriax_env.FactoriaXEnv` and
    returns the per-step change in total items mined as the reward.

    Args:
        inner: Core environment instance. Created automatically
            when ``None``.
    """

    def __init__(self, inner: FactoriaXEnv | None = None) -> None:
        """Initialize the mining skill wrapper.

        Args:
            inner: Core environment. A fresh instance is created
                when ``None``.
        """
        super().__init__()
        self._inner = inner or FactoriaXEnv()

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
        state: EnvState,
        action: int | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[str, Any]]:
        """Step the environment and compute mining reward.

        Reward equals the number of new ore items mined this step.

        Args:
            key: JAX random key.
            state: Current environment state.
            action: Action to take.
            params: Environment parameters.

        Returns:
            Tuple of (observation, new_state, reward, done, info).
        """
        prev_mined = state.items_mined
        obs, new_state, _, done, info = self._inner.step_env(
            key,
            state,
            action,
            params,
        )
        delta = jnp.sum(new_state.items_mined - prev_mined)
        reward = delta.astype(jnp.float32)
        return obs, new_state, reward, done, info

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        """Reset the environment.

        Args:
            key: JAX random key for world generation.
            params: Environment parameters.

        Returns:
            Tuple of (initial_observation, initial_state).
        """
        return self._inner.reset_env(key, params)

    def get_obs(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Get observation for the selected player.

        Args:
            state: Current environment state.
            params: Environment parameters.

        Returns:
            Float32 observation array.
        """
        return self._inner.get_obs(state, params)

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Check if the current state is terminal.

        Args:
            state: Current environment state.
            params: Environment parameters.

        Returns:
            Boolean indicating whether state is terminal.
        """
        return self._inner.is_terminal(state, params)

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
        return spaces.Box(0.0, 1.0, shape=(obs_size,), dtype=jnp.float32)


def mining_level(
    map_size: int = 5,
    ore_fraction: float = 0.5,
    min_res: int = 1,
    max_res: int = 3,
    seed: int = 0,
    max_timesteps: int = 200,
) -> tuple[Level, EnvParams]:
    """Generate a parameterized mining level.

    Scatters ore tiles across the map at the given density. Each ore
    tile gets a random resource count in ``[min_res, max_res]``.

    Args:
        map_size: Width and height of the square map.
        ore_fraction: Fraction of tiles that should be ore.
        min_res: Minimum resources per ore tile.
        max_res: Maximum resources per ore tile.
        seed: Numpy RNG seed for reproducible layouts.
        max_timesteps: Episode length.

    Returns:
        Tuple of (Level, EnvParams) ready to use.
    """
    rng = np.random.default_rng(seed)
    n_tiles = map_size * map_size
    n_ore = int(n_tiles * ore_fraction)

    builder = LevelBuilder(map_size, map_size)
    center = map_size // 2

    ore_indices = rng.choice(n_tiles, size=n_ore, replace=False)
    for idx in ore_indices:
        y, x = divmod(int(idx), map_size)
        if x == center and y == center:
            continue
        ore_type = BlockType(int(rng.choice(_ORE_TYPES)))
        res = int(rng.integers(min_res, max_res + 1))
        builder.fill_rect(x, y, 1, 1, ore_type, resources=res)

    builder.set_player_position(center, center)

    level = builder.build("mining_skill")
    params = EnvParams(
        map_width=map_size,
        map_height=map_size,
        num_players=1,
        max_machines=8,
        max_timesteps=max_timesteps,
    )
    return level, params
