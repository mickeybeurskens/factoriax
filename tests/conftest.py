"""Shared test fixtures and utilities."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import jax.numpy as jnp
import pytest

from factoriax import EnvState
from factoriax.constants import (
    BLOCK_RESOURCE_DTYPE,
    DEFAULT_MACHINE_MAX_HEALTH,
    DEFAULT_MAX_BITERS,
    MACHINE_INVENTORY_COUNT_DTYPE,
    MAX_ACHIEVEMENTS,
    NUM_ITEM_TYPES,
    NUM_TECHNOLOGIES,
    Direction,
    MachineType,
)


@pytest.fixture
def state_factory():
    """Factory for creating test states with sensible defaults.

    Returns a function that creates EnvState objects. Only the world_map is
    required; all other fields have sensible defaults.

    Player inventory is now a pouch: shape ``(num_players, NUM_ITEM_TYPES)``
    with counts per item type. Machine inventory is also a pouch:
    shape ``(H, W, NUM_ITEM_TYPES)``.

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
        player_inventory: jnp.ndarray | None = None,
        selected_player: int = 0,
        crafting_recipe: jnp.ndarray | None = None,
        craft_progress: jnp.ndarray | None = None,
        num_players: int = 1,
        block_resources: jnp.ndarray | None = None,
        machine_types: jnp.ndarray | None = None,
        machine_power: jnp.ndarray | None = None,
        machine_inventory: jnp.ndarray | None = None,
        machine_selected_recipe: jnp.ndarray | None = None,
        machine_direction: jnp.ndarray | None = None,
        achievements_unlocked: jnp.ndarray | None = None,
        items_mined: jnp.ndarray | None = None,
        research_progress: jnp.ndarray | None = None,
        research_unlocked: jnp.ndarray | None = None,
        machine_health: jnp.ndarray | None = None,
        biter_positions: jnp.ndarray | None = None,
        biter_health: jnp.ndarray | None = None,
        scent_field: jnp.ndarray | None = None,
    ) -> EnvState:
        """Create a test state with defaults for unspecified fields.

        Args:
            world_map: Block types array (required).
            player_position: Single player (x, y) position.
            player_positions: All player positions, shape (num_players, 2).
            player_direction: Single player direction.
            player_directions: All player directions, shape (num_players,).
            timestep: Current timestep.
            player_inventory: Pouch inventory, shape
                (num_players, NUM_ITEM_TYPES). Defaults to zeros.
            selected_player: Currently selected player index.
            crafting_recipe: Recipe in progress per player.
            craft_progress: Crafting progress per player.
            num_players: Number of players (used for defaults).
            block_resources: Resources per tile.
            machine_types: Machine type per tile.
            machine_power: Power per machine.
            machine_inventory: Machine pouch inventory, shape
                (H, W, NUM_ITEM_TYPES). Defaults to zeros.
            machine_selected_recipe: Active recipe per tile.
            machine_direction: Machine direction per tile.
            achievements_unlocked: Boolean array of achievements.
            items_mined: Lifetime mined count per item type.
            research_progress: Per-technology progress.
            research_unlocked: Boolean per-technology.
            machine_health: Health per tile.
            biter_positions: Biter positions.
            biter_health: Biter health.
            scent_field: Scent per tile.

        Returns:
            Configured EnvState for testing.
        """
        shape = world_map.shape

        if player_positions is not None:
            positions = player_positions
            num_players = positions.shape[0]
        elif player_position is not None:
            if isinstance(player_position, tuple):
                positions = jnp.array(
                    [player_position], dtype=jnp.int32,
                )
            else:
                positions = player_position.reshape(1, 2)
            num_players = 1
        else:
            positions = jnp.array([[0, 0]], dtype=jnp.int32)
            num_players = 1

        if player_directions is not None:
            directions = player_directions
        elif player_direction is not None:
            directions = jnp.array(
                [player_direction], dtype=jnp.int32,
            )
        else:
            directions = jnp.full(
                num_players, Direction.DOWN, dtype=jnp.int32,
            )

        inv_shape = (num_players, NUM_ITEM_TYPES)
        player_shape = (num_players,)
        machine_inv_shape = (*shape, NUM_ITEM_TYPES)

        return EnvState(
            map=world_map,
            player_positions=positions,
            player_directions=directions,
            timestep=timestep,
            player_inventory=(
                player_inventory
                if player_inventory is not None
                else jnp.zeros(inv_shape, dtype=jnp.int32)
            ),
            selected_player=selected_player,
            crafting_recipe=(
                crafting_recipe
                if crafting_recipe is not None
                else jnp.zeros(player_shape, dtype=jnp.int32)
            ),
            craft_progress=(
                craft_progress
                if craft_progress is not None
                else jnp.zeros(player_shape, dtype=jnp.int32)
            ),
            block_resources=(
                block_resources
                if block_resources is not None
                else jnp.zeros(shape, dtype=BLOCK_RESOURCE_DTYPE)
            ),
            machine_types=(
                machine_types
                if machine_types is not None
                else jnp.full(
                    shape, MachineType.NONE, dtype=jnp.int32,
                )
            ),
            machine_power=(
                machine_power
                if machine_power is not None
                else jnp.zeros(shape, dtype=jnp.int32)
            ),
            machine_inventory=(
                machine_inventory
                if machine_inventory is not None
                else jnp.zeros(
                    machine_inv_shape,
                    dtype=MACHINE_INVENTORY_COUNT_DTYPE,
                )
            ),
            machine_selected_recipe=(
                machine_selected_recipe
                if machine_selected_recipe is not None
                else jnp.zeros(shape, dtype=jnp.int32)
            ),
            machine_direction=(
                machine_direction
                if machine_direction is not None
                else jnp.zeros(shape, dtype=jnp.int32)
            ),
            achievements_unlocked=(
                achievements_unlocked
                if achievements_unlocked is not None
                else jnp.zeros(MAX_ACHIEVEMENTS, dtype=jnp.bool_)
            ),
            items_mined=(
                items_mined
                if items_mined is not None
                else jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.int32)
            ),
            research_progress=(
                research_progress
                if research_progress is not None
                else jnp.zeros(
                    NUM_TECHNOLOGIES, dtype=jnp.int32,
                )
            ),
            research_unlocked=(
                research_unlocked
                if research_unlocked is not None
                else jnp.zeros(
                    NUM_TECHNOLOGIES, dtype=jnp.bool_,
                )
            ),
            machine_health=(
                machine_health
                if machine_health is not None
                else jnp.where(
                    (
                        machine_types
                        if machine_types is not None
                        else jnp.full(
                            shape, MachineType.NONE,
                            dtype=jnp.int32,
                        )
                    )
                    != int(MachineType.NONE),
                    DEFAULT_MACHINE_MAX_HEALTH,
                    0,
                ).astype(jnp.int32)
            ),
            biter_positions=(
                biter_positions
                if biter_positions is not None
                else jnp.zeros(
                    (DEFAULT_MAX_BITERS, 2), dtype=jnp.int32,
                )
            ),
            biter_health=(
                biter_health
                if biter_health is not None
                else jnp.zeros(
                    DEFAULT_MAX_BITERS, dtype=jnp.int32,
                )
            ),
            scent_field=(
                scent_field
                if scent_field is not None
                else jnp.zeros(shape, dtype=jnp.float32)
            ),
        )

    return _create
