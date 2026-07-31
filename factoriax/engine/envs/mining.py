"""The Mining scenario: an 8x8 benchmark for mining by hand.

The agent earns 1 point for each ore item that it mines by hand. Each episode
places ten iron-ore tiles at random positions, and each tile holds 3
resources. An agent can therefore collect 30 items in total. The episode ends
after 100 steps.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from factoriax.engine.constants import BlockType
from factoriax.engine.envs.base import FactoriaxEnv
from factoriax.engine.state import EnvParams, EnvState

_MAP_SIZE: int = 8
_N_ORE_TILES: int = 10
_ORE_RESOURCES: int = 3

# Every tile position except the player spawn at (4, 4).
_VALID_POSITIONS: np.ndarray = np.array(
    [
        (x, y)
        for y in range(_MAP_SIZE)
        for x in range(_MAP_SIZE)
        if not (x == _MAP_SIZE // 2 and y == _MAP_SIZE // 2)
    ],
    dtype=np.int32,
)


def _mining_terrain(key: jax.Array, params: EnvParams) -> jax.Array:
    positions = jnp.asarray(_VALID_POSITIONS)  # (63, 2)
    shuffled = positions[jax.random.permutation(key, positions.shape[0])]
    ore_xy = shuffled[:_N_ORE_TILES]  # (10, 2)

    world = jnp.full((_MAP_SIZE, _MAP_SIZE), jnp.int8(BlockType.DIRT))

    def place(w: jax.Array, pos: jax.Array) -> tuple[jax.Array, None]:
        return w.at[pos[1], pos[0]].set(jnp.int8(BlockType.IRON)), None

    world, _ = jax.lax.scan(place, world, ore_xy)
    return world


def _mining_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    delta = jnp.sum(new_state.items_mined - prev_state.items_mined)
    return delta.astype(jnp.float32)


def mining(
    *,
    obs: str = "superficial_local",
    obs_radius: int = 7,
) -> tuple[FactoriaxEnv, EnvParams]:
    """Build the Mining-v1 environment.

    The map is 8x8 and generated. It holds ten iron-ore tiles at random
    positions, and each tile holds 3 resources. The agent earns 1 reward for
    each ore item that it mines by hand. An episode lasts 100 steps.

    The default observation is the egocentric ``superficial_local`` with radius
    7. That window covers the whole 8x8 map from any position. This scenario is
    an RL baseline, and a flat MLP policy stops near 7/30 on
    ``superficial_global`` and reaches 30/30 on ``superficial_local``. See
    :mod:`factoriax.engine.observations`. Pass ``obs="superficial_global"`` for
    the absolute-grid variant.
    """
    env = FactoriaxEnv(
        terrain_fn=_mining_terrain,
        reward_fn=_mining_reward,
        obs=obs,
        obs_radius=obs_radius,
        map_width=_MAP_SIZE,
        map_height=_MAP_SIZE,
        num_players=1,
    )
    params = EnvParams(
        max_timesteps=100,
        base_resources=_ORE_RESOURCES,
    )
    return env, params
