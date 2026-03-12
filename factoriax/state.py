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
    """

    map: jnp.ndarray
    player_position: jnp.ndarray
    player_direction: int
    timestep: int


@struct.dataclass
class EnvParams:
    """Environment parameters.

    Attributes:
        max_timesteps: Maximum number of timesteps per episode
        map_width: Width of the map grid
        map_height: Height of the map grid
        water_probability: Probability of a tile being water during generation
    """

    max_timesteps: int = 1000
    map_width: int = 32
    map_height: int = 32
    water_probability: float = 0.1

    NUM_ACTIONS: ClassVar[int] = 5
