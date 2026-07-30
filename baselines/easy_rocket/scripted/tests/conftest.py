"""Shared fixtures for the easy-rocket scripted-agent tests.

Provides a JIT-free ``make_state`` factory that packs grid-form
machine descriptions into a full :class:`EnvState`, mirroring the
top-level ``tests/conftest.py`` ``state_factory`` (which co-located
tests can't see). Tests build tiny synthetic states with this and
exercise the pure readers / skills / planner helpers.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
import pytest

from factoriax.engine.constants import (
    MAX_ACHIEVEMENTS,
    NUM_ITEM_TYPES,
    NUM_SCIENCE_PACK_TYPES,
    Direction,
    Machine,
)
from factoriax.engine.state import EnvState

# One machine to place in a synthetic state: (x, y, machine_type,
# direction, buf_type, buf_count).
PlacedMachine = tuple[int, int, int, int, int, int]


def _make_state(
    world_map: jnp.ndarray,
    *,
    block_resources: jnp.ndarray | None = None,
    machines: list[PlacedMachine] | None = None,
    player_position: tuple[int, int] = (0, 0),
    player_direction: int = int(Direction.DOWN),
    player_inventory: jnp.ndarray | None = None,
    max_machines: int = 64,
) -> EnvState:
    """Build a synthetic :class:`EnvState` from a map + machine list.

    JIT-free: pure numpy/jnp array construction, no env step. Machines
    are packed into the entity arrays and the ``machine_types`` /
    ``tile_entity`` grids exactly as the engine would lay them out.
    """
    h, w = world_map.shape
    mm = max_machines
    mt_grid = np.full((h, w), int(Machine.NONE), dtype=np.int8)
    tile_ent = np.full((h, w), -1, dtype=np.int16)
    ent_y = np.full(mm, -1, dtype=np.int16)
    ent_x = np.full(mm, -1, dtype=np.int16)
    ent_type = np.zeros(mm, dtype=np.int8)
    ent_dir = np.zeros(mm, dtype=np.int8)
    ent_buf_type = np.zeros(mm, dtype=np.int8)
    ent_buf_count = np.zeros(mm, dtype=np.int16)

    for idx, (x, y, mtype, mdir, btype, bcount) in enumerate(machines or []):
        mt_grid[y, x] = mtype
        tile_ent[y, x] = idx
        ent_y[idx] = y
        ent_x[idx] = x
        ent_type[idx] = mtype
        ent_dir[idx] = mdir
        ent_buf_type[idx] = btype
        ent_buf_count[idx] = bcount

    return EnvState(
        map=world_map.astype(jnp.int8),
        block_resources=(
            block_resources
            if block_resources is not None
            else jnp.zeros((h, w), dtype=jnp.int16)
        ),
        machine_types=jnp.array(mt_grid),
        tile_entity=jnp.array(tile_ent),
        ent_y=jnp.array(ent_y),
        ent_x=jnp.array(ent_x),
        ent_type=jnp.array(ent_type),
        ent_direction=jnp.array(ent_dir),
        ent_power=jnp.zeros(mm, dtype=jnp.int16),
        ent_buf_type=jnp.array(ent_buf_type),
        ent_buf_count=jnp.array(ent_buf_count),
        ent_asm_in_type=jnp.zeros((mm, 2), dtype=jnp.int8),
        ent_asm_in_count=jnp.zeros((mm, 2), dtype=jnp.int16),
        ent_asm_out_type=jnp.zeros(mm, dtype=jnp.int8),
        ent_asm_out_count=jnp.zeros(mm, dtype=jnp.int16),
        ent_health=jnp.zeros(mm, dtype=jnp.int16),
        player_positions=jnp.array([player_position], dtype=jnp.int16),
        player_directions=jnp.array([player_direction], dtype=jnp.int8),
        player_inventory=(
            player_inventory
            if player_inventory is not None
            else jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int16)
        ),
        selected_player=jnp.int32(0),
        timestep=jnp.int32(0),
        items_mined=jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.int32),
        science_consumed_step=jnp.zeros(NUM_SCIENCE_PACK_TYPES, dtype=jnp.int32),
        achievements_unlocked=jnp.zeros(MAX_ACHIEVEMENTS, dtype=jnp.bool_),
    )


@pytest.fixture
def make_state() -> Callable[..., EnvState]:
    """Return the synthetic :class:`EnvState` builder."""
    return _make_state
