"""Tests for :mod:`factoriax.analysis.milestones`.

:func:`first_action_timestep` answers one question for each episode.
At which timestep did this action first appear? The answer drives the
"time to first mine" and "time to first craft" figures. Two parts of
the contract therefore matter: the index base and the not-found value.

An index is a 0-based position along the time axis of
``Trajectory.actions``. An episode without the action reports ``-1``.
It reports neither ``None`` nor the episode length, because the result
is a plain int32 array. Plot code masks that array with ``result >= 0``.

A multi-player trajectory carries a third axis. These tests assert
which player the function reads, both when the caller names one and
when the caller names none.
"""

from __future__ import annotations

import numpy as np

from factoriax.analysis.milestones import first_action_timestep
from factoriax.analysis.trajectory import Trajectory


def _make_single_player(actions_2d: list[list[int]]) -> Trajectory:
    """Build a trajectory whose actions have shape ``(B, T)``.

    Parameters
    ----------
    actions_2d :
        One inner list for each episode. Each inner list holds the
        action ids of that episode, in timestep order. Every episode
        must have the same length.

    Returns
    -------
    Trajectory
        A trajectory with actions only. The other fields stay ``None``.
        :func:`first_action_timestep` reads no other field.
    """
    return Trajectory(actions=np.array(actions_2d, dtype=np.int32))


def _make_multi_player(actions_3d: list[list[list[int]]]) -> Trajectory:
    """Build a trajectory whose actions have shape ``(B, T, P)``.

    Parameters
    ----------
    actions_3d :
        Nested as ``[episode][timestep][player]``. The player axis is
        last. One timestep is therefore the list of the actions of
        every player at that moment.

    Returns
    -------
    Trajectory
        A trajectory that reports ``is_multi_player`` when ``P`` is
        more than 1. At ``P == 1`` the array is 3-D, but the trajectory
        counts as single-player.
    """
    return Trajectory(actions=np.array(actions_3d, dtype=np.int32))


class TestFirstActionTimestep:
    """First occurrence of one action id, per episode."""

    def test_single_player_found(self) -> None:
        """Each episode reports its own first hit, not the batch's."""
        traj = _make_single_player([[0, 0, 5, 0], [0, 5, 0, 0]])
        result = first_action_timestep(traj, action_id=5)
        np.testing.assert_array_equal(result, [2, 1])

    def test_single_player_not_found(self) -> None:
        """Report -1 for an action that never occurs, and not 0 or T."""
        traj = _make_single_player([[0, 0, 0], [0, 0, 0]])
        result = first_action_timestep(traj, action_id=5)
        np.testing.assert_array_equal(result, [-1, -1])

    def test_multi_player_defaults_to_player_0(self) -> None:
        """When the caller names no player, read player 0."""
        # B=1, T=4, P=2. Player 0 takes action 3 at t=0, player 1 at t=1.
        actions = [[[3, 0], [0, 3], [0, 0], [0, 0]]]
        traj = _make_multi_player(actions)
        result = first_action_timestep(traj, action_id=3)
        np.testing.assert_array_equal(result, [0])

    def test_multi_player_explicit_player_1(self) -> None:
        """Read the named player from the player axis."""
        # Same array as above. Player 1 first takes action 3 at t=1.
        actions = [[[3, 0], [0, 3], [0, 0], [0, 0]]]
        traj = _make_multi_player(actions)
        result = first_action_timestep(traj, action_id=3, player=1)
        np.testing.assert_array_equal(result, [1])

    def test_multi_player_action_absent(self) -> None:
        """The -1 sentinel survives the player selection path."""
        actions = [[[0, 0], [0, 0]]]
        traj = _make_multi_player(actions)
        result = first_action_timestep(traj, action_id=5)
        np.testing.assert_array_equal(result, [-1])

    def test_single_player_recorded_with_a_player_axis(self) -> None:
        """Read a ``(B, T, 1)`` recording as a ``(B, T)`` recording.

        Multi-agent code keeps the player axis for one agent too. This
        shape therefore arrives from real recordings.
        """
        actions = [[[0], [7], [0]]]
        traj = _make_multi_player(actions)
        result = first_action_timestep(traj, action_id=7)
        np.testing.assert_array_equal(result, [1])
