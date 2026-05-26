"""Tests for :meth:`PlaceMachineAt.verify`.

Verifies the four observable outcomes a placement can have:

- the target tile holds the expected ``(machine_type, direction)``
  pair (verify True);
- the tile is empty (verify False — the placement never landed);
- the tile holds a different machine type (verify False — wrong
  type, e.g. a coordinate clash with another stage's plan);
- the tile holds the right type but the wrong direction (verify
  False — orientation bug, common with miners and arms whose facing
  drives item flow).

Tests use a synthetic :class:`WorldView` rather than a live env step;
verify reads two arrays and is a pure function of those arrays.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from baselines.rocket.scripted.goals import PlaceMachineAt
from baselines.rocket.scripted.world_model import PlayerScalars, WorldView
from factoriax.engine.constants import Direction, Machine


def _make_view(
    *,
    width: int,
    height: int,
    machine_type_arr: np.ndarray,
    machine_direction_arr: np.ndarray,
) -> WorldView:
    """Build a :class:`WorldView` whose only meaningful fields are
    ``machine_type`` and ``machine_direction``.

    Other arrays are zero-filled. ``verify`` only reads the two
    meaningful fields, so the zero stubs are fine.
    """
    zero = np.zeros((height, width), dtype=np.int32)
    bool_zero = np.zeros((height, width), dtype=bool)
    return WorldView(
        block_type=zero,
        machine_type=machine_type_arr,
        block_resources=zero,
        slot0_type=zero,
        slot0_count=zero,
        slot1_type=zero,
        slot1_count=zero,
        slot2_type=zero,
        slot2_count=zero,
        machine_direction=machine_direction_arr,
        buffer_type=zero,
        player=PlayerScalars(
            pos_x=0,
            pos_y=0,
            direction=int(Direction.DOWN),
            timestep=0,
            facing_machine_type=0,
            facing_buffer_type=0,
            facing_buffer_count=0,
            inventory=np.zeros(64, dtype=np.int32),
        ),
        walkable=bool_zero,
    )


@pytest.fixture
def expected_pallet_view() -> WorldView:
    """3x3 grid with a PALLET facing DOWN at tile (1, 2) — the case
    PlaceMachineAt(PALLET, (1, 2), DOWN) was supposed to produce."""
    machine_type = np.zeros((3, 3), dtype=np.int32)
    machine_direction = np.zeros((3, 3), dtype=np.int32)
    machine_type[2, 1] = int(Machine.PALLET)
    machine_direction[2, 1] = int(Direction.DOWN)
    return _make_view(
        width=3,
        height=3,
        machine_type_arr=machine_type,
        machine_direction_arr=machine_direction,
    )


class TestVerifyPasses:
    """When the tile matches the goal's expectation, verify returns True."""

    def test_correct_type_and_direction(self, expected_pallet_view: WorldView) -> None:
        goal = PlaceMachineAt(Machine.PALLET, (1, 2), int(Direction.DOWN))
        assert goal.verify(expected_pallet_view) is True


class TestVerifyFails:
    """All three failure modes return False."""

    def test_tile_empty(self, expected_pallet_view: WorldView) -> None:
        """The target tile holds no machine — placement never landed."""
        empty_machine_type = np.zeros((3, 3), dtype=np.int32)
        empty_direction = np.zeros((3, 3), dtype=np.int32)
        view = replace(
            expected_pallet_view,
            machine_type=empty_machine_type,
            machine_direction=empty_direction,
        )
        goal = PlaceMachineAt(Machine.PALLET, (1, 2), int(Direction.DOWN))
        assert goal.verify(view) is False

    def test_wrong_machine_type(self, expected_pallet_view: WorldView) -> None:
        """The tile holds a MINER, not the expected PALLET."""
        wrong_type = expected_pallet_view.machine_type.copy()
        wrong_type[2, 1] = int(Machine.MINER)
        view = replace(expected_pallet_view, machine_type=wrong_type)
        goal = PlaceMachineAt(Machine.PALLET, (1, 2), int(Direction.DOWN))
        assert goal.verify(view) is False

    def test_wrong_direction(self, expected_pallet_view: WorldView) -> None:
        """The tile holds a PALLET facing the wrong way."""
        wrong_dir = expected_pallet_view.machine_direction.copy()
        wrong_dir[2, 1] = int(Direction.UP)
        view = replace(expected_pallet_view, machine_direction=wrong_dir)
        goal = PlaceMachineAt(Machine.PALLET, (1, 2), int(Direction.DOWN))
        assert goal.verify(view) is False


class TestRepr:
    """``__repr__`` is what the planner's diagnostic prints — it should
    name the machine and direction in human-readable form."""

    def test_repr_includes_human_readable_names(self) -> None:
        goal = PlaceMachineAt(Machine.PALLET, (1, 2), int(Direction.DOWN))
        text = repr(goal)
        assert "PALLET" in text
        assert "DOWN" in text
        assert "(1, 2)" in text
