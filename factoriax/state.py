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
        selected_slots: Currently selected inventory slot per player
            with shape (num_players,)
        selected_recipes: Currently selected recipe index per player
            with shape (num_players,)
        craft_progress: Ticks remaining in current craft per player
            with shape (num_players,)
        block_resources: Remaining resources for each tile with shape (height, width)
        machine_types: Machine type at each tile with shape (height, width)
        machine_power: Remaining power for each machine with shape (height, width)
        machine_inventory_items: Item type in each slot per tile with shape
            (height, width, MAX_MACHINE_INVENTORY_SLOTS)
        machine_inventory_counts: Stack count in each slot per tile with shape
            (height, width, MAX_MACHINE_INVENTORY_SLOTS)
        machine_selected_recipe: Active recipe index per tile with shape (height, width)
        machine_selected_slot: UI-focused slot index per tile with shape (height, width)
        machine_direction: Facing direction of each machine (Action enum value) with
            shape (height, width). Used by conveyor belts and arms to determine
            the push/pick direction.
        achievements_unlocked: Boolean array tracking unlocked achievements
            with shape (NUM_ACHIEVEMENTS,)
        items_mined: Lifetime mined count per item type with shape (NUM_ITEM_TYPES,)
    """

    map: jnp.ndarray
    player_positions: jnp.ndarray
    player_directions: jnp.ndarray
    timestep: int
    inventory_items: jnp.ndarray
    inventory_counts: jnp.ndarray
    selected_player: int
    selected_slots: jnp.ndarray
    selected_recipes: jnp.ndarray
    craft_progress: jnp.ndarray
    block_resources: jnp.ndarray
    machine_types: jnp.ndarray
    machine_power: jnp.ndarray
    machine_inventory_items: jnp.ndarray
    machine_inventory_counts: jnp.ndarray
    machine_selected_recipe: jnp.ndarray
    machine_selected_slot: jnp.ndarray
    machine_direction: jnp.ndarray
    achievements_unlocked: jnp.ndarray
    items_mined: jnp.ndarray


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
        base_resources: Starting resource count per ore tile in procedurally
            generated worlds. Does not affect levels built with
            :class:`~factoriax.levels.LevelBuilder`.
    """

    max_timesteps: int = 1000
    map_width: int = 32
    map_height: int = 32
    num_players: int = 2
    water_probability: float = 0.1
    iron_probability: float = 0.12
    copper_probability: float = 0.12
    coal_probability: float = 0.12
    base_resources: int = 1000

    NUM_ACTIONS: ClassVar[int] = 13
