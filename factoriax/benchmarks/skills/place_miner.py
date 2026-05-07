"""Place miner skill: reward for miners on ore tiles.

The agent starts with miners in inventory and places them on ore
patches. Every timestep the reward equals the number of miners
currently sitting on ore, incentivizing placement and discouraging
removal.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from gymnax.environments import environment, spaces  # type: ignore[import-untyped]

from factoriax.benchmarks.skills import count_miners_on_ore
from factoriax.constants import (
    NUM_ACTIONS,
    BlockType,
    ItemType,
)
from factoriax.envs.factoriax_env import FactoriaXEnv
from factoriax.levels import Level, LevelBuilder
from factoriax.observations import NUM_PLAYER_SCALARS, NUM_SPATIAL_CHANNELS
from factoriax.state import EnvParams, EnvState

_ORE_TYPES = [BlockType.IRON, BlockType.COPPER, BlockType.COAL]


class PlaceMinerSkill(environment.Environment[EnvState, EnvParams]):  # type: ignore[misc]
    """Gymnax environment rewarding miner placement on ore.

    Wraps :class:`~factoriax.envs.factoriax_env.FactoriaXEnv` and
    returns the current count of miners on ore tiles as the reward
    each step.

    Args:
        inner: Core environment instance. Created automatically
            when ``None``.
    """

    def __init__(self, inner: FactoriaXEnv | None = None) -> None:
        """Initialize the place miner skill wrapper.

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
        """Step the environment and compute placement reward.

        Reward equals the number of miners on ore right now.

        Args:
            key: JAX random key.
            state: Current environment state.
            action: Action to take.
            params: Environment parameters.

        Returns:
            Tuple of (observation, new_state, reward, done, info).
        """
        obs, new_state, _, done, info = self._inner.step_env(
            key,
            state,
            action,
            params,
        )
        reward = count_miners_on_ore(new_state).astype(jnp.float32)
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


def place_miner_level(
    map_size: int = 5,
    num_patches: int = 5,
    num_miners: int = 5,
    seed: int = 0,
    max_timesteps: int = 200,
) -> tuple[Level, EnvParams]:
    """Generate a parameterized miner placement level.

    Creates multi-tile ore patches and gives the player miners to
    place. Patches are 2-4 tiles each, randomly chosen from iron,
    copper, and coal.

    Args:
        map_size: Width and height of the square map.
        num_patches: Number of ore patches to place.
        num_miners: Number of miners in starting inventory.
        seed: Numpy RNG seed for reproducible layouts.
        max_timesteps: Episode length.

    Returns:
        Tuple of (Level, EnvParams) ready to use.
    """
    rng = np.random.default_rng(seed)
    builder = LevelBuilder(map_size, map_size)
    center = map_size // 2

    placed = np.zeros((map_size, map_size), dtype=bool)
    placed[center, center] = True

    for _ in range(num_patches):
        ore_type = BlockType(int(rng.choice(_ORE_TYPES)))
        patch_w = int(rng.integers(1, 3))
        patch_h = int(rng.integers(1, 3))
        for _attempt in range(50):
            px = int(rng.integers(0, map_size - patch_w + 1))
            py = int(rng.integers(0, map_size - patch_h + 1))
            region = placed[py : py + patch_h, px : px + patch_w]
            if not np.any(region):
                builder.fill_rect(px, py, patch_w, patch_h, ore_type, resources=3)
                placed[py : py + patch_h, px : px + patch_w] = True
                break

    builder.set_player_position(center, center)
    builder.set_player_inventory([(int(ItemType.MINER), num_miners)])

    level = builder.build("place_miner_skill")
    params = EnvParams(
        map_width=map_size,
        map_height=map_size,
        num_players=1,
        max_machines=num_miners + 4,
        max_timesteps=max_timesteps,
    )
    return level, params
