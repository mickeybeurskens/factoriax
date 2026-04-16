"""Shared test fixtures and utilities."""

import jax.numpy as jnp
import numpy as np
import pytest

from factoriax import EnvState
from factoriax.constants import (
    BLOCK_RESOURCE_DTYPE,
    NUM_ITEM_TYPES,
    NUM_TECHNOLOGIES,
    Direction,
    MachineType,
)

# Default entity capacity used by the test factory.
_TEST_MAX_MACHINES: int = 64


@pytest.fixture
def state_factory():
    """Factory for creating test states with sensible defaults.

    Returns a function that creates EnvState objects. Only the world_map is
    required; all other fields have sensible defaults.

    Old grid-based keyword arguments (machine_direction, machine_power,
    machine_fuel, buffer_type, buffer_count, asm_in_type, asm_in_count,
    asm_out_type, asm_out_count) are accepted for backward compatibility
    and translated into entity arrays automatically.

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
        num_players: int = 1,
        block_resources: jnp.ndarray | None = None,
        machine_types: jnp.ndarray | None = None,
        machine_power: jnp.ndarray | None = None,
        machine_direction: jnp.ndarray | None = None,
        machine_fuel: jnp.ndarray | None = None,
        buffer_type: jnp.ndarray | None = None,
        buffer_count: jnp.ndarray | None = None,
        asm_in_type: jnp.ndarray | None = None,
        asm_in_count: jnp.ndarray | None = None,
        asm_out_type: jnp.ndarray | None = None,
        asm_out_count: jnp.ndarray | None = None,
        items_mined: jnp.ndarray | None = None,
        research_progress: jnp.ndarray | None = None,
        research_unlocked: jnp.ndarray | None = None,
        max_machines: int = _TEST_MAX_MACHINES,
        # Backward-compat kwargs (ignored in new state)
        **_kwargs: object,
    ) -> EnvState:
        """Create a test state with defaults for unspecified fields.

        Args:
            world_map: Block types array (required).
            player_position: Single player (x, y) position.
            player_positions: All player positions.
            player_direction: Single player direction.
            player_directions: All player directions.
            timestep: Current timestep.
            player_inventory: Item counts per player.
            selected_player: Currently selected player index.
            num_players: Number of players (for defaults).
            block_resources: Resources per tile.
            machine_types: Machine type per tile.
            machine_power: Power per machine (grid, translated to entities).
            machine_direction: Direction per machine (grid, translated).
            machine_fuel: Miner coal per tile (grid, translated).
            buffer_type: Buffer item type per tile (grid, translated).
            buffer_count: Buffer item count per tile (grid, translated).
            asm_in_type: Assembler input types (grid, translated).
            asm_in_count: Assembler input counts (grid, translated).
            asm_out_type: Assembler output type (grid, translated).
            asm_out_count: Assembler output count (grid, translated).
            items_mined: Lifetime mined counts.
            research_progress: Per-technology progress.
            research_unlocked: Per-technology flags.
            max_machines: Entity array capacity.

        Returns:
            Configured EnvState for testing.
        """
        shape = world_map.shape
        mm = max_machines

        if player_positions is not None:
            positions = player_positions
            num_players = positions.shape[0]
        elif player_position is not None:
            if isinstance(player_position, tuple):
                positions = jnp.array(
                    [player_position],
                    dtype=jnp.int16,
                )
            else:
                positions = player_position.reshape(1, 2).astype(jnp.int16)
            num_players = 1
        else:
            positions = jnp.array([[0, 0]], dtype=jnp.int16)
            num_players = 1

        if player_directions is not None:
            directions = player_directions.astype(jnp.int8)
        elif player_direction is not None:
            directions = jnp.array(
                [player_direction],
                dtype=jnp.int8,
            )
        else:
            directions = jnp.full(
                num_players,
                Direction.DOWN,
                dtype=jnp.int8,
            )

        inv_shape = (num_players, NUM_ITEM_TYPES)

        mt_grid = (
            machine_types.astype(jnp.int8)
            if machine_types is not None
            else jnp.full(shape, MachineType.NONE, dtype=jnp.int8)
        )

        # Build entity arrays from the grid-based arguments.
        mt_np = np.asarray(mt_grid)
        md_np = (
            np.asarray(machine_direction)
            if machine_direction is not None
            else np.zeros(shape, dtype=np.int8)
        )
        mp_np = (
            np.asarray(machine_power)
            if machine_power is not None
            else np.zeros(shape, dtype=np.int16)
        )
        mf_np = (
            np.asarray(machine_fuel)
            if machine_fuel is not None
            else np.zeros(shape, dtype=np.int16)
        )
        bt_np = (
            np.asarray(buffer_type)
            if buffer_type is not None
            else np.zeros(shape, dtype=np.int8)
        )
        bc_np = (
            np.asarray(buffer_count)
            if buffer_count is not None
            else np.zeros(shape, dtype=np.int16)
        )
        ait_np = (
            np.asarray(asm_in_type)
            if asm_in_type is not None
            else np.zeros((*shape, 2), dtype=np.int8)
        )
        aic_np = (
            np.asarray(asm_in_count)
            if asm_in_count is not None
            else np.zeros((*shape, 2), dtype=np.int16)
        )
        aot_np = (
            np.asarray(asm_out_type)
            if asm_out_type is not None
            else np.zeros(shape, dtype=np.int8)
        )
        aoc_np = (
            np.asarray(asm_out_count)
            if asm_out_count is not None
            else np.zeros(shape, dtype=np.int16)
        )

        # Allocate entity arrays.
        ent_y = np.full(mm, -1, dtype=np.int16)
        ent_x = np.full(mm, -1, dtype=np.int16)
        ent_type = np.zeros(mm, dtype=np.int8)
        ent_dir = np.zeros(mm, dtype=np.int8)
        ent_power = np.zeros(mm, dtype=np.int16)
        ent_fuel = np.zeros(mm, dtype=np.int16)
        ent_buf_type = np.zeros(mm, dtype=np.int8)
        ent_buf_count = np.zeros(mm, dtype=np.int16)
        ent_asm_in_type = np.zeros((mm, 2), dtype=np.int8)
        ent_asm_in_count = np.zeros((mm, 2), dtype=np.int16)
        ent_asm_out_type = np.zeros(mm, dtype=np.int8)
        ent_asm_out_count = np.zeros(mm, dtype=np.int16)
        tile_ent = np.full(shape, -1, dtype=np.int16)

        idx = 0
        for y in range(shape[0]):
            for x in range(shape[1]):
                if int(mt_np[y, x]) != int(MachineType.NONE) and idx < mm:
                    ent_y[idx] = y
                    ent_x[idx] = x
                    ent_type[idx] = mt_np[y, x]
                    ent_dir[idx] = md_np[y, x]
                    ent_power[idx] = mp_np[y, x]
                    ent_fuel[idx] = mf_np[y, x]
                    ent_buf_type[idx] = bt_np[y, x]
                    ent_buf_count[idx] = bc_np[y, x]
                    ent_asm_in_type[idx] = ait_np[y, x]
                    ent_asm_in_count[idx] = aic_np[y, x]
                    ent_asm_out_type[idx] = aot_np[y, x]
                    ent_asm_out_count[idx] = aoc_np[y, x]
                    tile_ent[y, x] = idx
                    idx += 1

        return EnvState(
            map=world_map.astype(jnp.int8),
            block_resources=(
                block_resources
                if block_resources is not None
                else jnp.zeros(shape, dtype=BLOCK_RESOURCE_DTYPE)
            ),
            machine_types=mt_grid,
            tile_entity=jnp.array(tile_ent, dtype=jnp.int16),
            ent_y=jnp.array(ent_y, dtype=jnp.int16),
            ent_x=jnp.array(ent_x, dtype=jnp.int16),
            ent_type=jnp.array(ent_type, dtype=jnp.int8),
            ent_direction=jnp.array(ent_dir, dtype=jnp.int8),
            ent_power=jnp.array(ent_power, dtype=jnp.int16),
            ent_fuel=jnp.array(ent_fuel, dtype=jnp.int16),
            ent_buf_type=jnp.array(ent_buf_type, dtype=jnp.int8),
            ent_buf_count=jnp.array(ent_buf_count, dtype=jnp.int16),
            ent_asm_in_type=jnp.array(ent_asm_in_type, dtype=jnp.int8),
            ent_asm_in_count=jnp.array(ent_asm_in_count, dtype=jnp.int16),
            ent_asm_out_type=jnp.array(ent_asm_out_type, dtype=jnp.int8),
            ent_asm_out_count=jnp.array(ent_asm_out_count, dtype=jnp.int16),
            player_positions=positions.astype(jnp.int16),
            player_directions=directions,
            player_inventory=(
                player_inventory.astype(jnp.int16)
                if player_inventory is not None
                else jnp.zeros(inv_shape, dtype=jnp.int16)
            ),
            selected_player=selected_player,
            timestep=timestep,
            items_mined=(
                items_mined
                if items_mined is not None
                else jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.int32)
            ),
            research_progress=(
                research_progress
                if research_progress is not None
                else jnp.zeros(NUM_TECHNOLOGIES, dtype=jnp.int16)
            ),
            research_unlocked=(
                research_unlocked
                if research_unlocked is not None
                else jnp.zeros(NUM_TECHNOLOGIES, dtype=jnp.bool_)
            ),
        )

    return _create
