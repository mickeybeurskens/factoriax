"""Tests for :class:`baselines.rocket.scripted.goals.VerifyLayout`.

Three concerns:

- :meth:`step` returns DONE in one tick (no-op as advertised).
- :meth:`verify` returns True for a layout that matches.
- :meth:`verify` returns False otherwise — the planner can then halt
  via the standard :attr:`Goal.verify_failure_action` path.
- The goal halts the planner end-to-end when wired in.
"""

from __future__ import annotations

import numpy as np
import pytest

from baselines.rocket.scripted.goals import VerifyLayout
from baselines.rocket.scripted.planner import Planner
from baselines.rocket.scripted.skills import Result
from baselines.rocket.scripted.world_model import PlayerScalars, WorldView
from factoriax.constants import Direction, Machine


def _make_view(machine_type_arr: np.ndarray) -> WorldView:
    """3x3 view from a single machine_type array (direction zero-filled)."""
    height, width = machine_type_arr.shape
    direction = np.zeros((height, width), dtype=np.int32)
    zero = np.zeros((height, width), dtype=np.int32)
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
        machine_direction=direction,
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
        walkable=np.zeros((height, width), dtype=bool),
    )


@pytest.fixture
def matching_view() -> WorldView:
    """Pallet at (0, 0) facing direction 0 (the zero-init for direction)."""
    machine_type = np.zeros((3, 3), dtype=np.int32)
    machine_type[0, 0] = int(Machine.PALLET)
    return _make_view(machine_type)


def test_step_returns_done_immediately(matching_view: WorldView) -> None:
    goal = VerifyLayout(expected={}, label="empty")
    result, action = goal.step(matching_view)
    assert result is Result.DONE
    assert action is not None  # NOOP integer, not ``None``.


def test_verify_passes_for_matching_layout(matching_view: WorldView) -> None:
    expected = {(0, 0): (int(Machine.PALLET), 0)}
    goal = VerifyLayout(expected, label="match")
    assert goal.verify(matching_view) is True


def test_verify_fails_for_missing_tile(matching_view: WorldView) -> None:
    expected = {(2, 2): (int(Machine.PALLET), 0)}
    goal = VerifyLayout(expected, label="missing")
    assert goal.verify(matching_view) is False


def test_planner_halts_on_verify_failure(matching_view: WorldView) -> None:
    """Drop a VerifyLayout into a Planner; it halts the run on miss."""
    expected_miss = {(2, 2): (int(Machine.PALLET), 0)}
    goal = VerifyLayout(expected_miss, label="halt-test")
    planner = Planner([goal], debug_log=True)

    result, _ = planner.step(matching_view)

    assert result is Result.FAIL
    diag = planner.verify_diagnostic
    assert diag is not None
    assert "halt-test" in diag.goal_repr


def test_planner_passes_on_clean_layout(matching_view: WorldView) -> None:
    expected = {(0, 0): (int(Machine.PALLET), 0)}
    planner = Planner(
        [VerifyLayout(expected, label="ok")],
        debug_log=True,
    )

    result, _ = planner.step(matching_view)

    assert result is Result.RUNNING
    assert planner.is_done
    assert planner.verify_diagnostic is None


def test_repr_includes_label_and_tile_count() -> None:
    goal = VerifyLayout(
        {(0, 0): (1, 0), (1, 1): (2, 1)},
        label="iron stage",
    )
    text = repr(goal)
    assert "iron stage" in text
    assert "n_tiles=2" in text
