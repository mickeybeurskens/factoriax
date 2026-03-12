"""State dataclasses for the FactoriaX environment."""

from typing import ClassVar

import jax.numpy as jnp
from flax import struct


@struct.dataclass
class EnvState:
    """Immutable environment state.

    Attributes:
        map: 2D grid of block types with shape (height, width)
        player_position: (x, y) coordinates of the player
        player_direction: Direction the player is facing (Action value)
        timestep: Current timestep in the episode
        inventory_items: Array of item type IDs with shape (NUM_INVENTORY_SLOTS,)
        inventory_counts: Array of stack counts with shape (NUM_INVENTORY_SLOTS,)
    """

    map: jnp.ndarray
    player_position: jnp.ndarray
    player_direction: int
    timestep: int
    inventory_items: jnp.ndarray
    inventory_counts: jnp.ndarray


@struct.dataclass
class EnvParams:
    """Environment parameters.

    Attributes:
        max_timesteps: Maximum number of timesteps per episode
        map_width: Width of the map grid
        map_height: Height of the map grid
        water_probability: Probability of a tile being water during generation
        iron_probability: Probability of a tile being iron ore during generation
        copper_probability: Probability of a tile being copper ore during generation
        coal_probability: Probability of a tile being coal during generation
    """

    max_timesteps: int = 1000
    map_width: int = 32
    map_height: int = 32
    water_probability: float = 0.1
    iron_probability: float = 0.02
    copper_probability: float = 0.02
    coal_probability: float = 0.02

    NUM_ACTIONS: ClassVar[int] = 5
