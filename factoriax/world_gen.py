"""World generation for the FactoriaX environment."""

import jax
import jax.numpy as jnp
from jax import random

from factoriax.constants import Action, BlockType
from factoriax.state import EnvParams, EnvState


def generate_world(rng: jax.Array, params: EnvParams) -> EnvState:
    """Generate a new world with random dirt and water tiles.

    Creates a grid filled primarily with dirt tiles and some scattered water tiles.
    The player spawns at the center of the map, and we ensure the spawn location
    is always dirt by overwriting it after random generation.

    Args:
        rng: JAX random key for reproducible generation
        params: Environment parameters including map dimensions and water probability

    Returns:
        Initial environment state with generated map and player at center
    """
    rng_map, rng_spawn = random.split(rng)

    world_map = _generate_terrain(rng_map, params)
    player_x = params.map_width // 2
    player_y = params.map_height // 2
    world_map = world_map.at[player_y, player_x].set(BlockType.DIRT)
    player_position = jnp.array([player_x, player_y], dtype=jnp.int32)

    return EnvState(
        map=world_map,
        player_position=player_position,
        player_direction=Action.DOWN,
        timestep=0,
    )


def _generate_terrain(rng: jax.Array, params: EnvParams) -> jax.Array:
    """Generate terrain with dirt and water patches.

    Args:
        rng: JAX random key
        params: Environment parameters

    Returns:
        2D array of BlockType values with shape (map_height, map_width)
    """
    random_values = random.uniform(rng, (params.map_height, params.map_width))
    is_water = random_values < params.water_probability
    terrain = jnp.where(is_water, BlockType.WATER, BlockType.DIRT)
    return terrain.astype(jnp.int32)
