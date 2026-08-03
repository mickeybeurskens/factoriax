"""Build an :class:`EnvState` for a test from grid-shaped arguments.

The engine reads flat ``ent_*`` entity arrays plus a ``tile_entity`` lookup.
That layout is fast, but a test that sets it by hand is unreadable. This
module takes ``(H, W)``-shaped grids instead and packs them. One setup line
then places a machine at ``(y, x)`` with a facing and a buffer.

The translation is a test ergonomic, not a compatibility shim. The engine
itself never reads a grid.

The ``state_factory`` fixture in ``tests/conftest.py`` exposes
:func:`make_state` to every test.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from factoriax.engine.constants import (
    MAX_ACHIEVEMENTS,
    NUM_ITEM_TYPES,
    NUM_SCIENCE_PACK_TYPES,
    Direction,
    Machine,
)
from factoriax.engine.state import EnvState
from factoriax.engine.tables import MACHINE_MAX_HEALTH

#: Default entity capacity of a state that this module builds.
TEST_MAX_MACHINES: int = 64


def make_state(
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
    buffer_type: jnp.ndarray | None = None,
    buffer_count: jnp.ndarray | None = None,
    asm_in_type: jnp.ndarray | None = None,
    asm_in_count: jnp.ndarray | None = None,
    asm_out_type: jnp.ndarray | None = None,
    asm_out_count: jnp.ndarray | None = None,
    items_mined: jnp.ndarray | None = None,
    science_consumed_step: jnp.ndarray | None = None,
    max_machines: int = TEST_MAX_MACHINES,
) -> EnvState:
    """Create a test state, with a default for every unset field.

    Parameters
    ----------
    world_map
        Block types array. Required.
    player_position
        Single player ``(x, y)`` position.
    player_positions
        All player positions.
    player_direction
        Single player direction.
    player_directions
        All player directions.
    timestep
        Current timestep.
    player_inventory
        Item counts per player.
    selected_player
        Currently selected player index.
    num_players
        Number of players, used for the defaults.
    block_resources
        Resources per tile.
    machine_types
        Machine type per tile.
    machine_power
        Power per machine, grid form, packed into ``ent_power``.
    machine_direction
        Direction per machine, grid form, packed into ``ent_direction``.
    buffer_type
        Buffer item type per tile, grid form, packed into ``ent_buf_type``.
    buffer_count
        Buffer item count per tile, grid form, packed into ``ent_buf_count``.
    asm_in_type
        Assembler input types, grid form, packed into ``ent_asm_in_type``.
    asm_in_count
        Assembler input counts, grid form, packed into ``ent_asm_in_count``.
    asm_out_type
        Assembler output type, grid form, packed into ``ent_asm_out_type``.
    asm_out_count
        Assembler output count, grid form, packed into ``ent_asm_out_count``.
    items_mined
        Lifetime mined counts.
    science_consumed_step
        Per-step science pack consumption delta, from SCIENCE_LAB entities.
    max_machines
        Entity array capacity.

    Returns
    -------
    EnvState
        Configured state for testing.

    Examples
    --------
    >>> state = make_state(
    ...     world_map=jnp.array([[BlockType.COAL]]),
    ...     block_resources=jnp.array([[50]]),
    ... )
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
        else jnp.full(shape, Machine.NONE, dtype=jnp.int8)
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
    ent_buf_type = np.zeros(mm, dtype=np.int8)
    ent_buf_count = np.zeros(mm, dtype=np.int16)
    ent_asm_in_type = np.zeros((mm, 2), dtype=np.int8)
    ent_asm_in_count = np.zeros((mm, 2), dtype=np.int16)
    ent_asm_out_type = np.zeros(mm, dtype=np.int8)
    ent_asm_out_count = np.zeros(mm, dtype=np.int16)
    ent_health = np.zeros(mm, dtype=np.int16)
    tile_ent = np.full(shape, -1, dtype=np.int16)

    idx = 0
    for y in range(shape[0]):
        for x in range(shape[1]):
            if int(mt_np[y, x]) != int(Machine.NONE) and idx < mm:
                ent_y[idx] = y
                ent_x[idx] = x
                ent_type[idx] = mt_np[y, x]
                ent_dir[idx] = md_np[y, x]
                ent_power[idx] = mp_np[y, x]
                ent_buf_type[idx] = bt_np[y, x]
                ent_buf_count[idx] = bc_np[y, x]
                ent_asm_in_type[idx] = ait_np[y, x]
                ent_asm_in_count[idx] = aic_np[y, x]
                ent_asm_out_type[idx] = aot_np[y, x]
                ent_asm_out_count[idx] = aoc_np[y, x]
                # Full health, matching what ``place_machine`` writes.
                # Pickup is gated on full health, so leaving this at 0
                # builds machines that the engine can never pick up, and
                # makes every pickup test exercise the refused path.
                ent_health[idx] = int(MACHINE_MAX_HEALTH[int(mt_np[y, x])])
                tile_ent[y, x] = idx
                idx += 1

    return EnvState(
        map=world_map.astype(jnp.int8),
        block_resources=(
            block_resources
            if block_resources is not None
            else jnp.zeros(shape, dtype=jnp.int16)
        ),
        machine_types=mt_grid,
        tile_entity=jnp.array(tile_ent, dtype=jnp.int16),
        ent_y=jnp.array(ent_y, dtype=jnp.int16),
        ent_x=jnp.array(ent_x, dtype=jnp.int16),
        ent_type=jnp.array(ent_type, dtype=jnp.int8),
        ent_direction=jnp.array(ent_dir, dtype=jnp.int8),
        ent_power=jnp.array(ent_power, dtype=jnp.int16),
        ent_buf_type=jnp.array(ent_buf_type, dtype=jnp.int8),
        ent_buf_count=jnp.array(ent_buf_count, dtype=jnp.int16),
        ent_asm_in_type=jnp.array(ent_asm_in_type, dtype=jnp.int8),
        ent_asm_in_count=jnp.array(ent_asm_in_count, dtype=jnp.int16),
        ent_asm_out_type=jnp.array(ent_asm_out_type, dtype=jnp.int8),
        ent_asm_out_count=jnp.array(ent_asm_out_count, dtype=jnp.int16),
        ent_health=jnp.array(ent_health, dtype=jnp.int16),
        player_positions=positions.astype(jnp.int16),
        player_directions=directions,
        player_inventory=(
            player_inventory.astype(jnp.int16)
            if player_inventory is not None
            else jnp.zeros(inv_shape, dtype=jnp.int16)
        ),
        # Match env.reset_env's pytree shape: factoriax/engine/levels.py emits
        # these as jnp.int32(0). A Python-int leaf here forces
        # jax.jit(env.step_env) to retrace whenever a test feeds a state from
        # this factory through the canonical_env_8x8_1p fixture's step path.
        selected_player=jnp.int32(selected_player),
        timestep=jnp.int32(timestep),
        items_mined=(
            items_mined
            if items_mined is not None
            else jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.int32)
        ),
        science_consumed_step=(
            science_consumed_step
            if science_consumed_step is not None
            else jnp.zeros(NUM_SCIENCE_PACK_TYPES, dtype=jnp.int32)
        ),
        achievements_unlocked=jnp.zeros(MAX_ACHIEVEMENTS, dtype=jnp.bool_),
    )
