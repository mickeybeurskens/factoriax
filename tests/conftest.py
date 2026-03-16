"""Shared test fixtures and utilities."""

import jax.numpy as jnp
import pytest

from factoriax import Action, EnvState
from factoriax.achievements import NUM_ACHIEVEMENTS
from factoriax.constants import NUM_INVENTORY_SLOTS, NUM_ITEM_TYPES, MachineType


@pytest.fixture
def state_factory():
    """Factory for creating test states with sensible defaults.

    Returns a function that creates EnvState objects. Only the world_map is
    required; all other fields have sensible defaults. This reduces boilerplate
    in tests that don't need to specify every field.

    For single-player tests, pass player_position as a tuple. For multiplayer,
    pass player_positions as an array of shape (num_players, 2).

    Example:
        def test_something(state_factory):
            state = state_factory(
                world_map=jnp.array([[BlockType.COAL]]),
                block_resources=jnp.array([[50]]),
            )
    """

    def _create(
        world_map: jnp.ndarray,
        player_position: tuple[int, int] | None = None,
        player_positions: jnp.ndarray | None = None,
        player_direction: int | None = None,
        player_directions: jnp.ndarray | None = None,
        timestep: int = 0,
        inventory_items: jnp.ndarray | None = None,
        inventory_counts: jnp.ndarray | None = None,
        selected_player: int = 0,
        selected_slots: jnp.ndarray | None = None,
        selected_recipes: jnp.ndarray | None = None,
        craft_progress: jnp.ndarray | None = None,
        num_players: int = 1,
        block_resources: jnp.ndarray | None = None,
        machine_types: jnp.ndarray | None = None,
        machine_power: jnp.ndarray | None = None,
        machine_fuel_count: jnp.ndarray | None = None,
        machine_output_item: jnp.ndarray | None = None,
        machine_output_count: jnp.ndarray | None = None,
        achievements_unlocked: jnp.ndarray | None = None,
        items_mined: jnp.ndarray | None = None,
    ) -> EnvState:
        """Create a test state with defaults for unspecified fields.

        Args:
            world_map: Block types array (required)
            player_position: Single player (x, y) position (convenience for 1 player)
            player_positions: All player positions, shape (num_players, 2)
            player_direction: Single player direction (convenience for 1 player)
            player_directions: All player directions, shape (num_players,)
            timestep: Current timestep, defaults to 0
            inventory_items: Inventory items, shape (num_players, NUM_INVENTORY_SLOTS)
            inventory_counts: Inventory counts, shape (num_players, NUM_INVENTORY_SLOTS)
            selected_player: Currently selected player index, defaults to 0
            selected_slots: Selected inventory slot per player, defaults to zeros
            selected_recipes: Selected recipe per player, defaults to zeros
            craft_progress: Crafting progress per player, defaults to zeros
            num_players: Number of players (used for defaults), defaults to 1
            block_resources: Resources per tile, defaults to zeros
            machine_types: Machine type per tile, defaults to NONE
            machine_power: Power per machine, defaults to zeros
            machine_fuel_count: Fuel per machine, defaults to zeros
            machine_output_item: Output item per machine, defaults to zeros
            machine_output_count: Output count per machine, defaults to zeros
            achievements_unlocked: Boolean array of unlocked achievements, defaults to all False
            items_mined: Lifetime mined count per item type, defaults to zeros

        Returns:
            Configured EnvState for testing
        """
        shape = world_map.shape

        if player_positions is not None:
            positions = player_positions
            num_players = positions.shape[0]
        elif player_position is not None:
            if isinstance(player_position, tuple):
                positions = jnp.array([player_position], dtype=jnp.int32)
            else:
                positions = player_position.reshape(1, 2)
            num_players = 1
        else:
            positions = jnp.array([[0, 0]], dtype=jnp.int32)
            num_players = 1

        if player_directions is not None:
            directions = player_directions
        elif player_direction is not None:
            directions = jnp.array([player_direction], dtype=jnp.int32)
        else:
            directions = jnp.full(num_players, Action.DOWN, dtype=jnp.int32)

        inv_shape = (num_players, NUM_INVENTORY_SLOTS)
        player_shape = (num_players,)

        return EnvState(
            map=world_map,
            player_positions=positions,
            player_directions=directions,
            timestep=timestep,
            inventory_items=inventory_items
            if inventory_items is not None
            else jnp.zeros(inv_shape, dtype=jnp.int32),
            inventory_counts=inventory_counts
            if inventory_counts is not None
            else jnp.zeros(inv_shape, dtype=jnp.int32),
            selected_player=selected_player,
            selected_slots=selected_slots
            if selected_slots is not None
            else jnp.zeros(player_shape, dtype=jnp.int32),
            selected_recipes=selected_recipes
            if selected_recipes is not None
            else jnp.zeros(player_shape, dtype=jnp.int32),
            craft_progress=craft_progress
            if craft_progress is not None
            else jnp.zeros(player_shape, dtype=jnp.int32),
            block_resources=block_resources
            if block_resources is not None
            else jnp.zeros(shape, dtype=jnp.int16),
            machine_types=machine_types
            if machine_types is not None
            else jnp.full(shape, MachineType.NONE, dtype=jnp.int32),
            machine_power=machine_power
            if machine_power is not None
            else jnp.zeros(shape, dtype=jnp.int32),
            machine_fuel_count=machine_fuel_count
            if machine_fuel_count is not None
            else jnp.zeros(shape, dtype=jnp.int16),
            machine_output_item=machine_output_item
            if machine_output_item is not None
            else jnp.zeros(shape, dtype=jnp.int32),
            machine_output_count=machine_output_count
            if machine_output_count is not None
            else jnp.zeros(shape, dtype=jnp.int16),
            achievements_unlocked=achievements_unlocked
            if achievements_unlocked is not None
            else jnp.zeros(NUM_ACHIEVEMENTS, dtype=jnp.bool_),
            items_mined=items_mined
            if items_mined is not None
            else jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.int32),
        )

    return _create
