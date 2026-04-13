"""Fuel miner skill: reward for fueling pre-placed miners.

Pre-placed miners sit on ore tiles. The agent starts with coal and
must deposit it into miners to fuel them. The reward combines:

- ``+1`` for each newly fueled miner (sparse goal signal)
- ``-1`` for each miner picked up / removed (penalty)
- Small proximity bonus toward the nearest unfueled miner (dense
  shaping signal to guide exploration on larger maps)
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from gymnax.environments import environment, spaces  # type: ignore[import-untyped]

from factoriax.benchmarks.skills import (
    count_fueled_miners_on_ore,
    count_miners_on_ore,
)
from factoriax.constants import (
    MINEABLE_BLOCKS,
    NUM_ACTIONS,
    NUM_TECHNOLOGIES,
    BlockType,
    Direction,
    ItemType,
    MachineType,
)
from factoriax.envs.factoriax_env import FactoriaXEnv
from factoriax.levels import Level, LevelBuilder
from factoriax.observations import NUM_PLAYER_SCALARS, NUM_SPATIAL_CHANNELS
from factoriax.state import EnvParams, EnvState

_ORE_TYPES = [BlockType.IRON, BlockType.COPPER, BlockType.COAL]


def _proximity_bonus(
    state: EnvState,
    params: EnvParams,
) -> jax.Array:
    """Compute proximity bonus toward the nearest unfueled miner on ore.

    Returns a scalar in [0, 1] where 1 means the player is on top of
    an unfueled miner and 0 means no unfueled miners remain (or
    maximum distance). Uses Manhattan distance, fully JIT-compatible.

    Args:
        state: Current environment state.
        params: Environment parameters.

    Returns:
        Scalar float32 proximity bonus.
    """
    px, py = state.player_positions[0]
    active = state.ent_y >= 0
    is_miner = state.ent_type == MachineType.MINER
    not_fueled = state.ent_fuel <= 0
    ey = jnp.clip(state.ent_y, 0)
    ex = jnp.clip(state.ent_x, 0)
    tile = state.map[ey, ex]
    on_ore = jnp.isin(tile, MINEABLE_BLOCKS)
    target = active & is_miner & not_fueled & on_ore

    dist = jnp.abs(ex - px) + jnp.abs(ey - py)
    max_dist = params.map_width + params.map_height
    # Use max_dist for non-target entities so they don't affect the min.
    masked_dist = jnp.where(target, dist, max_dist)
    min_dist = jnp.min(masked_dist)
    has_target = jnp.any(target)
    bonus = jnp.where(
        has_target,
        1.0 - min_dist.astype(jnp.float32) / max_dist,
        0.0,
    )
    return bonus


class FuelMinerSkill(environment.Environment[EnvState, EnvParams]):  # type: ignore[misc]
    """Gymnax environment rewarding miner fueling.

    Wraps :class:`~factoriax.envs.factoriax_env.FactoriaXEnv`. The
    reward combines a sparse delta signal (``+1`` per newly fueled
    miner, ``-1`` per picked-up miner) with a small proximity bonus
    toward the nearest unfueled miner on ore.

    Args:
        inner: Core environment instance. Created automatically
            when ``None``.
        proximity_scale: Weight of the proximity bonus (default 0.01).
    """

    def __init__(
        self,
        inner: FactoriaXEnv | None = None,
        proximity_scale: float = 0.01,
    ) -> None:
        """Initialize the fuel miner skill wrapper.

        Args:
            inner: Core environment. A fresh instance is created
                when ``None``.
            proximity_scale: Weight of the proximity bonus.
        """
        super().__init__()
        self._inner = inner or FactoriaXEnv()
        self._proximity_scale = proximity_scale

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
        """Step the environment and compute fueling reward.

        Reward is ``fuel_delta + pickup_penalty + proximity_bonus``.

        Args:
            key: JAX random key.
            state: Current environment state.
            action: Action to take.
            params: Environment parameters.

        Returns:
            Tuple of (observation, new_state, reward, done, info).
        """
        prev_fueled = count_fueled_miners_on_ore(state)
        prev_total = count_miners_on_ore(state)
        obs, new_state, _, done, info = self._inner.step_env(
            key,
            state,
            action,
            params,
        )
        new_fueled = count_fueled_miners_on_ore(new_state)
        new_total = count_miners_on_ore(new_state)
        fuel_delta = (new_fueled - prev_fueled).astype(jnp.float32)
        pickup_penalty = jnp.minimum(
            new_total - prev_total,
            0,
        ).astype(jnp.float32)
        prox = self._proximity_scale * _proximity_bonus(new_state, params)
        reward = fuel_delta + pickup_penalty + prox
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
        return self._inner.reset_from_level(level, params)

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
            + NUM_TECHNOLOGIES * 2
        )
        return spaces.Box(0.0, 1.0, shape=(obs_size,), dtype=jnp.float32)


def fuel_miner_level(
    map_size: int = 5,
    num_miners: int = 5,
    seed: int = 0,
    max_timesteps: int = 200,
) -> tuple[Level, EnvParams]:
    """Generate a parameterized miner fueling level.

    Places miners on ore tiles in a checkerboard-like pattern so
    each miner has at least one adjacent walkable tile. The player
    starts with a full stack of coal.

    Args:
        map_size: Width and height of the square map.
        num_miners: Number of pre-placed miners on ore.
        seed: Numpy RNG seed for reproducible layouts.
        max_timesteps: Episode length.

    Returns:
        Tuple of (Level, EnvParams) ready to use.
    """
    rng = np.random.default_rng(seed)
    builder = LevelBuilder(map_size, map_size)

    candidates: list[tuple[int, int]] = []
    for y in range(map_size):
        for x in range(map_size):
            if (x + y) % 2 == 0:
                candidates.append((x, y))

    center = map_size // 2
    candidates = [(x, y) for x, y in candidates if not (x == center and y == center)]

    rng.shuffle(candidates)
    placed = candidates[: min(num_miners, len(candidates))]

    for x, y in placed:
        ore_type = int(rng.choice(_ORE_TYPES))
        builder.fill_rect(x, y, 1, 1, ore_type, resources=100)
        builder.place_machine(x, y, int(MachineType.MINER), int(Direction.DOWN))

    builder.set_player_position(center, center)
    builder.set_player_inventory([(int(ItemType.COAL), 64)])

    level = builder.build("fuel_miner_skill")
    params = EnvParams(
        map_width=map_size,
        map_height=map_size,
        num_players=1,
        max_machines=num_miners + 4,
        max_timesteps=max_timesteps,
    )
    return level, params
