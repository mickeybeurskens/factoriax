"""Unit tests for the scripted agent's action-emitting skills (JIT-free)."""

from __future__ import annotations

import jax.numpy as jnp

from baselines.easy_rocket.scripted.skills import (
    craft_action,
    direction_to,
    face_action,
    is_adjacent,
    move_action,
    place_action,
    rotate_action,
    stand_tile_for,
    step_toward_adjacent,
    step_toward_tile,
)
from factoriax.constants import Action, BlockType, Direction, ItemType, MachineType


def test_is_adjacent() -> None:
    assert is_adjacent((2, 2), (3, 2)) is True
    assert is_adjacent((2, 2), (2, 3)) is True
    assert is_adjacent((2, 2), (3, 3)) is False  # diagonal
    assert is_adjacent((2, 2), (2, 2)) is False  # same tile


def test_direction_to() -> None:
    assert direction_to((2, 2), (3, 2)) == int(Direction.RIGHT)
    assert direction_to((2, 2), (1, 2)) == int(Direction.LEFT)
    assert direction_to((2, 2), (2, 1)) == int(Direction.UP)
    assert direction_to((2, 2), (2, 3)) == int(Direction.DOWN)
    assert direction_to((2, 2), (4, 2)) is None  # not adjacent


def test_stand_tile_for() -> None:
    # To place at (5, 5) facing RIGHT, stand one tile to the left.
    assert stand_tile_for((5, 5), int(Direction.RIGHT)) == (4, 5)
    assert stand_tile_for((5, 5), int(Direction.UP)) == (5, 6)
    assert stand_tile_for((5, 5), int(Direction.DOWN)) == (5, 4)


def test_action_maps() -> None:
    assert craft_action(int(ItemType.MINER)) == int(Action.CRAFT_MINER)
    assert craft_action(int(ItemType.PALLET)) == int(Action.CRAFT_PALLET)
    assert place_action(int(ItemType.CONVEYOR_BELT)) == int(Action.PLACE_CONVEYOR_BELT)
    assert place_action(int(ItemType.ASSEMBLER)) == int(Action.PLACE_ASSEMBLER)
    assert rotate_action(int(Direction.UP)) == int(Action.ROTATE_UP)
    assert face_action(int(Direction.LEFT)) == int(Action.FACE_LEFT)
    assert move_action(int(Direction.RIGHT)) == int(Action.RIGHT)


def _dirt(h: int = 6, w: int = 6) -> jnp.ndarray:
    return jnp.full((h, w), int(BlockType.DIRT), dtype=jnp.int8)


def test_step_toward_tile_straight_line(make_state) -> None:
    state = make_state(_dirt(), player_position=(0, 0))
    # Target three tiles east; first step is RIGHT.
    assert step_toward_tile(state, 3, 0) == int(Action.RIGHT)


def test_step_toward_tile_routes_around_machine(make_state) -> None:
    # A non-belt machine at (1, 0) blocks the direct path east; the
    # BFS must detour (down or up) rather than walk into it.
    machines = [(1, 0, int(MachineType.ASSEMBLER), 0, 0, 0)]
    state = make_state(_dirt(), machines=machines, player_position=(0, 0))
    action = step_toward_tile(state, 3, 0)
    assert action != int(Action.RIGHT)
    assert action in (int(Action.DOWN), int(Action.UP))


def test_step_toward_tile_already_there(make_state) -> None:
    state = make_state(_dirt(), player_position=(2, 2))
    assert step_toward_tile(state, 2, 2) == int(Action.NOOP)


def test_step_toward_adjacent_stops_when_adjacent(make_state) -> None:
    # Player already orthogonally adjacent to the (unwalkable) ore
    # target → no movement needed.
    m = _dirt().at[2, 3].set(int(BlockType.IRON))
    state = make_state(m, player_position=(2, 2))
    assert step_toward_adjacent(state, 3, 2) == int(Action.NOOP)
