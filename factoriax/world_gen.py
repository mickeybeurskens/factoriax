"""World generation for the FactoriaX environment."""

import jax
import jax.numpy as jnp
from jax import random

from factoriax.constants import (
    BLOCK_MAX_RESOURCES,
    MINEABLE_BLOCKS,
    NUM_INVENTORY_SLOTS,
    Action,
    BlockType,
    MachineType,
)
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

    is_mineable = jnp.isin(world_map, MINEABLE_BLOCKS)
    block_resources = jnp.where(is_mineable, BLOCK_MAX_RESOURCES, 0).astype(jnp.int16)

    map_shape = (params.map_height, params.map_width)

    return EnvState(
        map=world_map,
        player_position=player_position,
        player_direction=Action.DOWN,
        timestep=0,
        inventory_items=jnp.zeros(NUM_INVENTORY_SLOTS, dtype=jnp.int32),
        inventory_counts=jnp.zeros(NUM_INVENTORY_SLOTS, dtype=jnp.int32),
        block_resources=block_resources,
        machine_types=jnp.full(map_shape, MachineType.NONE, dtype=jnp.int32),
        machine_power=jnp.zeros(map_shape, dtype=jnp.int32),
        machine_fuel_count=jnp.zeros(map_shape, dtype=jnp.int16),
        machine_output_item=jnp.zeros(map_shape, dtype=jnp.int32),
        machine_output_count=jnp.zeros(map_shape, dtype=jnp.int16),
    )


def _generate_terrain(rng: jax.Array, params: EnvParams) -> jax.Array:
    """Generate terrain with dirt, water, and resource patches.

    Uses cumulative probability thresholds to assign block types. The order of
    checks determines priority when probabilities overlap: water, iron, copper,
    coal, then dirt as the fallback.

    Args:
        rng: JAX random key
        params: Environment parameters

    Returns:
        2D array of BlockType values with shape (map_height, map_width)
    """
    random_values = random.uniform(rng, (params.map_height, params.map_width))

    water_threshold = params.water_probability
    iron_threshold = water_threshold + params.iron_probability
    copper_threshold = iron_threshold + params.copper_probability
    coal_threshold = copper_threshold + params.coal_probability

    terrain = jnp.full(
        (params.map_height, params.map_width), BlockType.DIRT, dtype=jnp.int32
    )
    terrain = jnp.where(random_values < coal_threshold, BlockType.COAL, terrain)
    terrain = jnp.where(random_values < copper_threshold, BlockType.COPPER, terrain)
    terrain = jnp.where(random_values < iron_threshold, BlockType.IRON, terrain)
    terrain = jnp.where(random_values < water_threshold, BlockType.WATER, terrain)

    return terrain
