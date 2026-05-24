"""Unit tests for the scripted agent's EnvState readers (JIT-free)."""

from __future__ import annotations

import jax.numpy as jnp

from baselines.easy_rocket.scripted.state_reader import (
    block_at,
    ent_buf,
    ent_direction_at,
    entity_at,
    find_patches,
    inv_count,
    machine_at,
    player_direction,
    player_pos,
    tile_free,
    tile_walkable_for_player,
)
from factoriax.constants import BlockType, Direction, ItemType, MachineType


def _dirt(h: int = 6, w: int = 6) -> jnp.ndarray:
    return jnp.full((h, w), int(BlockType.DIRT), dtype=jnp.int8)


def test_find_patches_groups_one_ore(make_state) -> None:
    m = _dirt()
    # 2x2 iron patch at columns 2-3, rows 1-2.
    for y in (1, 2):
        for x in (2, 3):
            m = m.at[y, x].set(int(BlockType.IRON))
    res = jnp.zeros((6, 6), dtype=jnp.int16)
    for y in (1, 2):
        for x in (2, 3):
            res = res.at[y, x].set(100)
    state = make_state(m, block_resources=res)

    patches = find_patches(state)

    assert len(patches) == 1
    assert patches[0].ore_block == int(BlockType.IRON)
    assert set(patches[0].tiles) == {(2, 1), (3, 1), (2, 2), (3, 2)}


def test_find_patches_excludes_depleted_tiles(make_state) -> None:
    m = _dirt().at[0, 0].set(int(BlockType.COAL)).at[0, 1].set(int(BlockType.COAL))
    # Only (0,0) still has resources; (1,0) is depleted.
    res = jnp.zeros((6, 6), dtype=jnp.int16).at[0, 0].set(50)
    state = make_state(m, block_resources=res)

    patches = find_patches(state)

    assert len(patches) == 1
    assert patches[0].tiles == ((0, 0),)


def test_player_pos_and_direction(make_state) -> None:
    state = make_state(
        _dirt(), player_position=(3, 4), player_direction=int(Direction.UP)
    )
    assert player_pos(state) == (3, 4)
    assert player_direction(state) == int(Direction.UP)


def test_inv_count(make_state) -> None:
    inv = jnp.zeros((1, 64), dtype=jnp.int16).at[0, int(ItemType.MINER)].set(3)
    state = make_state(_dirt(), player_inventory=inv)
    assert inv_count(state, int(ItemType.MINER)) == 3
    assert inv_count(state, int(ItemType.COAL)) == 0


def test_block_at(make_state) -> None:
    m = _dirt().at[2, 1].set(int(BlockType.WATER))
    state = make_state(m)
    assert block_at(state, 1, 2) == int(BlockType.WATER)
    assert block_at(state, 0, 0) == int(BlockType.DIRT)


def test_machine_and_entity_readers(make_state) -> None:
    # A belt facing RIGHT with 5 buffered iron at (3, 2).
    belt = int(MachineType.CONVEYOR_BELT)
    machines = [(3, 2, belt, int(Direction.RIGHT), int(ItemType.IRON_ORE), 5)]
    state = make_state(_dirt(), machines=machines)

    assert machine_at(state, 3, 2) == int(MachineType.CONVEYOR_BELT)
    assert machine_at(state, 0, 0) == int(MachineType.NONE)
    assert entity_at(state, 3, 2) >= 0
    assert entity_at(state, 0, 0) == -1
    assert ent_direction_at(state, 3, 2) == int(Direction.RIGHT)
    assert ent_buf(state, entity_at(state, 3, 2)) == (int(ItemType.IRON_ORE), 5)


def test_tile_free(make_state) -> None:
    m = _dirt().at[0, 1].set(int(BlockType.IRON))  # ore at (1, 0)
    machines = [(2, 0, int(MachineType.ASSEMBLER), 0, 0, 0)]
    state = make_state(m, machines=machines)

    assert tile_free(state, 0, 0) is True  # dirt, empty
    assert tile_free(state, 1, 0) is False  # ore tile
    assert tile_free(state, 2, 0) is False  # has a machine
    assert tile_free(state, -1, 0) is False  # out of bounds


def test_tile_walkable_for_player(make_state) -> None:
    m = _dirt().at[0, 1].set(int(BlockType.IRON)).at[0, 2].set(int(BlockType.WATER))
    machines = [
        (3, 0, int(MachineType.ASSEMBLER), 0, 0, 0),
        (4, 0, int(MachineType.CONVEYOR_BELT), int(Direction.RIGHT), 0, 0),
    ]
    state = make_state(m, machines=machines)

    assert tile_walkable_for_player(state, 0, 0) is True  # dirt
    assert tile_walkable_for_player(state, 1, 0) is True  # ore — walkable!
    assert tile_walkable_for_player(state, 2, 0) is False  # water
    assert tile_walkable_for_player(state, 3, 0) is False  # non-belt machine
    assert tile_walkable_for_player(state, 4, 0) is True  # belt — walkable
    assert tile_walkable_for_player(state, -1, 0) is False  # out of bounds
