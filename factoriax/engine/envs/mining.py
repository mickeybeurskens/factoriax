"""Mining scenario — 8×8 hand-mining benchmark.

The agent earns 1 point for every ore item it mines manually. Ten iron-ore
tiles are placed randomly each episode, each holding 3 resources (30 total
collectable items). The episode ends after 100 steps.
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

# All tile positions except the player spawn at (4, 4).
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
    """Build the Mining-v1 env.

    8×8 procgen map with ten randomly placed iron-ore tiles (3 resources
    each). The agent earns 1 reward per ore item mined manually. Episode
    length is 100 steps.

    Defaults to the egocentric ``superficial_local`` obs with radius 7
    (the window covers the full 8×8 map from any position): this
    scenario is an RL baseline, and flat-MLP policies plateau near 7/30
    on ``superficial_global`` versus 30/30 on ``superficial_local`` —
    see :mod:`factoriax.engine.observations`. Pass
    ``obs="superficial_global"`` for the absolute-grid variant.
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
