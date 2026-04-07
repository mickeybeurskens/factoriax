"""State dataclasses for the FactoriaX environment."""

from typing import ClassVar

import jax.numpy as jnp
from flax import struct

from factoriax.constants import Action


class EnvState(struct.PyTreeNode):  # type: ignore[no-untyped-call]
    """Immutable environment state.

    Attributes:
        map: 2D grid of block types with shape (height, width).
        player_positions: (x, y) coordinates per player, shape (num_players, 2).
        player_directions: Facing direction per player, shape (num_players,).
        timestep: Current timestep in the episode.
        player_inventory: Item counts per type per player,
            shape (num_players, NUM_ITEM_TYPES). Indexed by ItemType.
        selected_player: Index of the currently selected player for UI/input.
        crafting_recipe: Recipe index of the craft in progress per player,
            shape (num_players,). Only meaningful when craft_progress > 0.
        craft_progress: Ticks remaining in current craft per player,
            shape (num_players,). Zero means no craft is active.
        block_resources: Remaining resources per tile, shape (height, width).
        machine_types: Machine type at each tile, shape (height, width).
        machine_power: Remaining power per machine, shape (height, width).
        machine_inventory: Item counts per type per tile,
            shape (height, width, NUM_ITEM_TYPES). Indexed by ItemType.
        machine_selected_recipe: Active assembler recipe per tile,
            shape (height, width).
        machine_direction: Facing direction per machine, shape (height, width).
            Used by conveyor belts and arms for push/pick direction.
        achievements_unlocked: Boolean array of unlocked achievements,
            shape (NUM_ACHIEVEMENTS,).
        items_mined: Lifetime mined count per item type, shape (NUM_ITEM_TYPES,).
        research_progress: Science packs consumed per technology,
            shape (NUM_TECHNOLOGIES,).
        research_unlocked: Boolean array of unlocked technologies,
            shape (NUM_TECHNOLOGIES,).
        machine_health: Current health per tile, shape (height, width).
            Zero means disabled (machine present but non-operational).
        biter_positions: (x, y) coordinates per biter, shape (max_biters, 2).
        biter_health: HP per biter, shape (max_biters,). Zero means inactive.
        scent_field: Machine scent intensity per tile, shape (height, width).
    """

    map: jnp.ndarray
    player_positions: jnp.ndarray
    player_directions: jnp.ndarray
    timestep: int
    player_inventory: jnp.ndarray
    selected_player: int
    crafting_recipe: jnp.ndarray
    craft_progress: jnp.ndarray
    block_resources: jnp.ndarray
    machine_types: jnp.ndarray
    machine_power: jnp.ndarray
    machine_inventory: jnp.ndarray
    machine_selected_recipe: jnp.ndarray
    machine_direction: jnp.ndarray
    achievements_unlocked: jnp.ndarray
    items_mined: jnp.ndarray
    research_progress: jnp.ndarray
    research_unlocked: jnp.ndarray
    machine_health: jnp.ndarray
    biter_positions: jnp.ndarray
    biter_health: jnp.ndarray
    scent_field: jnp.ndarray


class EnvParams(struct.PyTreeNode):  # type: ignore[no-untyped-call]
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
    machine_max_health: int = 100
    power_per_coal: int = 10
    miner_mining_rate: int = 3
    max_assembler_stack_size: int = 1000
    max_biters: int = 32
    biter_spawn_rate: float = 0.05
    biter_tick_interval: int = 3
    biter_attack_damage: int = 5
    biter_health_default: int = 20
    scent_decay: float = 0.8
    scent_emission: float = 1.0
    nest_probability: float = 0.02

    NUM_ACTIONS: ClassVar[int] = len(Action)
