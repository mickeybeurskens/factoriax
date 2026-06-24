"""Tests for factoriax.analysis.milestones."""

from __future__ import annotations

import numpy as np
import pytest

from factoriax.analysis.milestones import first_action_timestep
from factoriax.analysis.trajectory import Trajectory


def _make_single_player(actions_2d: list[list[int]]) -> Trajectory:
    """Build a (B, T) single-player trajectory."""
    return Trajectory(actions=np.array(actions_2d, dtype=np.int32))


def _make_multi_player(actions_3d: list[list[list[int]]]) -> Trajectory:
    """Build a (B, T, P) multi-player trajectory."""
    return Trajectory(actions=np.array(actions_3d, dtype=np.int32))


class TestFirstActionTimestep:
    def test_single_player_found(self) -> None:
        traj = _make_single_player([[0, 0, 5, 0], [0, 5, 0, 0]])
        result = first_action_timestep(traj, action_id=5)
        np.testing.assert_array_equal(result, [2, 1])

    def test_single_player_not_found(self) -> None:
        traj = _make_single_player([[0, 0, 0], [0, 0, 0]])
        result = first_action_timestep(traj, action_id=5)
        np.testing.assert_array_equal(result, [-1, -1])

    def test_multi_player_defaults_to_player_0(self) -> None:
        # B=1, T=4, P=2: shape is (B, T, P)
        # t=0: p0=3, p1=0 → player 0 first sees action 3 at t=0
        actions = [[[3, 0], [0, 3], [0, 0], [0, 0]]]
        traj = _make_multi_player(actions)
        result = first_action_timestep(traj, action_id=3)
        np.testing.assert_array_equal(result, [0])

    def test_multi_player_explicit_player_1(self) -> None:
        # t=0: p0=3, p1=0; t=1: p0=0, p1=3 → player 1 first sees action 3 at t=1
        actions = [[[3, 0], [0, 3], [0, 0], [0, 0]]]
        traj = _make_multi_player(actions)
        result = first_action_timestep(traj, action_id=3, player=1)
        np.testing.assert_array_equal(result, [1])

    def test_multi_player_action_absent(self) -> None:
        actions = [[[0, 0], [0, 0]]]
        traj = _make_multi_player(actions)
        result = first_action_timestep(traj, action_id=5)
        np.testing.assert_array_equal(result, [-1])
