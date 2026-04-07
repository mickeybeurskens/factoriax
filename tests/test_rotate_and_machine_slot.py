"""Tests for ROTATE action (pouch inventory model).

Machine slot cycling tests are removed since the pouch system has no
slot cursors. Deposit/withdraw focused-slot tests are replaced by
typed compound action tests in test_game_logic.py.
"""

from __future__ import annotations

import jax.numpy as jnp

from factoriax import BlockType, Direction
from factoriax.constants import MachineType
from factoriax.game_logic import rotate_adjacent


class TestRotateAdjacent:
    """Tests for machine rotation."""

    def test_full_clockwise_cycle(self, state_factory) -> None:
        """Rotating four times should return to the starting direction."""
        state = state_factory(
            world_map=jnp.full(
                (3, 3), BlockType.DIRT, dtype=jnp.int32,
            ),
            player_position=(1, 0),
            player_direction=int(Direction.DOWN),
            machine_types=jnp.full(
                (3, 3), MachineType.NONE, dtype=jnp.int32,
            ).at[1, 1].set(MachineType.CONVEYOR_BELT),
            machine_direction=jnp.zeros(
                (3, 3), dtype=jnp.int32,
            ).at[1, 1].set(int(Direction.DOWN)),
        )
        dirs = [int(Direction.DOWN)]
        for _ in range(4):
            state = rotate_adjacent(state, 0)
            dirs.append(int(state.machine_direction[1, 1]))
        # Should cycle: DOWN -> RIGHT -> UP -> LEFT -> DOWN.
        assert dirs == [
            Direction.DOWN,
            Direction.RIGHT,
            Direction.UP,
            Direction.LEFT,
            Direction.DOWN,
        ]

    def test_noop_on_empty_tile(self, state_factory) -> None:
        """Rotating on a tile with no machine should be a no-op."""
        state = state_factory(
            world_map=jnp.full(
                (3, 3), BlockType.DIRT, dtype=jnp.int32,
            ),
            player_position=(1, 0),
            player_direction=int(Direction.DOWN),
        )
        new = rotate_adjacent(state, 0)
        assert jnp.array_equal(
            new.machine_direction, state.machine_direction,
        )

    def test_noop_out_of_bounds(self, state_factory) -> None:
        """Rotating toward an out-of-bounds tile should be a no-op."""
        state = state_factory(
            world_map=jnp.full(
                (2, 2), BlockType.DIRT, dtype=jnp.int32,
            ),
            player_position=(0, 0),
            player_direction=int(Direction.UP),
        )
        new = rotate_adjacent(state, 0)
        assert jnp.array_equal(
            new.machine_direction, state.machine_direction,
        )
