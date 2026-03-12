"""Shared test fixtures and utilities."""

import jax.numpy as jnp
import pytest

from factoriax import Action, EnvState
from factoriax.constants import NUM_INVENTORY_SLOTS, MachineType


@pytest.fixture
def state_factory():
    """Factory for creating test states with sensible defaults.

    Returns a function that creates EnvState objects. Only the world_map is
    required; all other fields have sensible defaults. This reduces boilerplate
    in tests that don't need to specify every field.

    Example:
        def test_something(state_factory):
            state = state_factory(
                world_map=jnp.array([[BlockType.COAL]]),
                block_resources=jnp.array([[50]]),
            )
    """

    def _create(
        world_map: jnp.ndarray,
        player_position: tuple[int, int] | jnp.ndarray = (0, 0),
        player_direction: int = Action.DOWN,
        timestep: int = 0,
        inventory_items: jnp.ndarray | None = None,
        inventory_counts: jnp.ndarray | None = None,
        block_resources: jnp.ndarray | None = None,
        machine_types: jnp.ndarray | None = None,
        machine_power: jnp.ndarray | None = None,
        machine_fuel_count: jnp.ndarray | None = None,
        machine_output_item: jnp.ndarray | None = None,
        machine_output_count: jnp.ndarray | None = None,
    ) -> EnvState:
        """Create a test state with defaults for unspecified fields.

        Args:
            world_map: Block types array (required)
            player_position: Player (x, y) position, defaults to (0, 0)
            player_direction: Direction player faces, defaults to DOWN
            timestep: Current timestep, defaults to 0
            inventory_items: Player inventory items, defaults to zeros
            inventory_counts: Player inventory counts, defaults to zeros
            block_resources: Resources per tile, defaults to zeros
            machine_types: Machine type per tile, defaults to NONE
            machine_power: Power per machine, defaults to zeros
            machine_fuel_count: Fuel per machine, defaults to zeros
            machine_output_item: Output item per machine, defaults to zeros
            machine_output_count: Output count per machine, defaults to zeros

        Returns:
            Configured EnvState for testing
        """
        shape = world_map.shape

        if isinstance(player_position, tuple):
            player_position = jnp.array(player_position, dtype=jnp.int32)

        return EnvState(
            map=world_map,
            player_position=player_position,
            player_direction=player_direction,
            timestep=timestep,
            inventory_items=inventory_items
            if inventory_items is not None
            else jnp.zeros(NUM_INVENTORY_SLOTS, dtype=jnp.int32),
            inventory_counts=inventory_counts
            if inventory_counts is not None
            else jnp.zeros(NUM_INVENTORY_SLOTS, dtype=jnp.int32),
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
        )

    return _create
