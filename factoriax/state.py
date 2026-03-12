"""State dataclasses for the FactoriaX environment."""

from typing import ClassVar

import jax.numpy as jnp
from flax import struct


@struct.dataclass
class EnvState:
    """Immutable environment state.

    Attributes:
        map: 2D grid of block types with shape (height, width)
        player_positions: (x, y) coordinates per player with shape (num_players, 2)
        player_directions: Direction each player is facing with shape (num_players,)
        timestep: Current timestep in the episode
        inventory_items: Item type IDs with shape (num_players, NUM_INVENTORY_SLOTS)
        inventory_counts: Stack counts with shape (num_players, NUM_INVENTORY_SLOTS)
        selected_player: Index of the currently selected player for UI/input
        block_resources: Remaining resources for each tile with shape (height, width)
        machine_types: Machine type at each tile with shape (height, width)
        machine_power: Remaining power for each machine with shape (height, width)
        machine_fuel_count: Coal count in fuel slot with shape (height, width)
        machine_output_item: Item type in output slot with shape (height, width)
        machine_output_count: Stack count in output slot with shape (height, width)
    """

    map: jnp.ndarray
    player_positions: jnp.ndarray
    player_directions: jnp.ndarray
    timestep: int
    inventory_items: jnp.ndarray
    inventory_counts: jnp.ndarray
    selected_player: int
    block_resources: jnp.ndarray
    machine_types: jnp.ndarray
    machine_power: jnp.ndarray
    machine_fuel_count: jnp.ndarray
    machine_output_item: jnp.ndarray
    machine_output_count: jnp.ndarray


@struct.dataclass
class EnvParams:
    """Environment parameters.

    Attributes:
        max_timesteps: Maximum number of timesteps per episode
        map_width: Width of the map grid
        map_height: Height of the map grid
        num_players: Number of players in the game
        water_probability: Probability of a tile being water during generation
        iron_probability: Probability of a tile being iron ore during generation
        copper_probability: Probability of a tile being copper ore during generation
        coal_probability: Probability of a tile being coal during generation
    """

    max_timesteps: int = 1000
    map_width: int = 32
    map_height: int = 32
    num_players: int = 2
    water_probability: float = 0.1
    iron_probability: float = 0.02
    copper_probability: float = 0.02
    coal_probability: float = 0.02

    NUM_ACTIONS: ClassVar[int] = 5
