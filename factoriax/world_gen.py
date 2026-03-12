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
    Players spawn near the center of the map in a horizontal line, and we ensure
    spawn locations are always dirt by overwriting after random generation.

    Args:
        rng: JAX random key for reproducible generation
        params: Environment parameters including map dimensions and water probability

    Returns:
        Initial environment state with generated map and players near center
    """
    rng_map, rng_spawn = random.split(rng)

    world_map = _generate_terrain(rng_map, params)

    center_x = params.map_width // 2
    center_y = params.map_height // 2
    player_positions = []
    for i in range(params.num_players):
        offset = i - params.num_players // 2
        px = jnp.clip(center_x + offset, 0, params.map_width - 1)
        py = center_y
        player_positions.append([px, py])
        world_map = world_map.at[py, px].set(BlockType.DIRT)

    player_positions = jnp.array(player_positions, dtype=jnp.int32)
    player_directions = jnp.full(params.num_players, Action.DOWN, dtype=jnp.int32)

    is_mineable = jnp.isin(world_map, MINEABLE_BLOCKS)
    block_resources = jnp.where(is_mineable, BLOCK_MAX_RESOURCES, 0).astype(jnp.int16)

    map_shape = (params.map_height, params.map_width)
    inv_shape = (params.num_players, NUM_INVENTORY_SLOTS)

    return EnvState(
        map=world_map,
        player_positions=player_positions,
        player_directions=player_directions,
        timestep=0,
        inventory_items=jnp.zeros(inv_shape, dtype=jnp.int32),
        inventory_counts=jnp.zeros(inv_shape, dtype=jnp.int32),
        selected_player=0,
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
