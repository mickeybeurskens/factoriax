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

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from factoriax.analysis.milestones import (
    achievement_timing,
    first_action_timestep,
    plot_achievement_progress,
    plot_achievement_timing,
    plot_first_action_timing,
)
from factoriax.analysis.trajectory import Trajectory
from tests.helpers.trajectories import (
    make_minimal_traj,
    make_multi_player_traj,
    make_single_player_traj,
)


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


class TestMilestones:
    """Tests for achievement timing and milestone event detection."""

    # ---- achievement_timing ----

    def test_achievement_timing_shape(self) -> None:
        """achievement_timing returns shape (num_eps, num_achievements)."""
        traj = make_multi_player_traj(8, 40, num_p=2)
        timing = achievement_timing(traj)
        assert timing.shape == (8, 2)

    def test_achievement_timing_never_unlocked_is_minus_one(self) -> None:
        """Episodes that never unlock an achievement get -1."""
        num_eps, num_steps = 4, 20
        achievements = np.zeros((num_eps, num_steps, 2), dtype=bool)
        traj = Trajectory(
            actions=np.zeros((num_eps, num_steps), dtype=np.int32),
            achievements=achievements,
        )
        timing = achievement_timing(traj)
        np.testing.assert_array_equal(timing[:, 1], -1)

    def test_achievement_timing_returns_first_true_timestep(self) -> None:
        """The reported timestep is the first True occurrence, not the last."""
        num_eps, num_steps = 4, 20
        achievements = np.zeros((num_eps, num_steps, 1), dtype=bool)
        achievements[:, 5:, 0] = True  # unlocked at t=5, stays True
        traj = Trajectory(
            actions=np.zeros((num_eps, num_steps), dtype=np.int32),
            achievements=achievements,
        )
        timing = achievement_timing(traj)
        np.testing.assert_array_equal(timing[:, 0], 5)

    def test_achievement_timing_raises_without_achievements(self) -> None:
        """achievement_timing raises ValueError when achievements field is absent."""
        traj = make_minimal_traj(4, 20)
        with pytest.raises(ValueError):
            achievement_timing(traj)

    def test_achievement_timing_partial_unlock(self) -> None:
        """Only episodes that unlock have non-negative timing."""
        num_eps, num_steps = 6, 30
        achievements = np.zeros((num_eps, num_steps, 1), dtype=bool)
        achievements[:3, 10:, 0] = True  # only first 3 episodes unlock at t=10
        traj = Trajectory(
            actions=np.zeros((num_eps, num_steps), dtype=np.int32),
            achievements=achievements,
        )
        timing = achievement_timing(traj)
        assert np.all(timing[:3, 0] == 10)
        assert np.all(timing[3:, 0] == -1)

    # ---- first_action_timestep ----

    def test_first_action_timestep_absent_action(self) -> None:
        """Returns -1 for all episodes when the action never appears."""
        traj = make_minimal_traj(4, 20)  # all zeros
        result = first_action_timestep(traj, action_id=5)
        np.testing.assert_array_equal(result, -1)

    def test_first_action_timestep_correct_index(self) -> None:
        """Returns the first index where the action appears."""
        num_eps, num_steps = 4, 20
        actions = np.zeros((num_eps, num_steps), dtype=np.int32)
        actions[:, 7] = 3  # action 3 first appears at t=7
        traj = Trajectory(actions=actions)
        result = first_action_timestep(traj, action_id=3)
        np.testing.assert_array_equal(result, 7)

    def test_first_action_timestep_multi_player(self) -> None:
        """first_action_timestep works for a specific player in multi-player traj."""
        num_eps, num_steps, num_p = 4, 20, 2
        actions = np.zeros((num_eps, num_steps, num_p), dtype=np.int32)
        actions[:, 5, 0] = 2  # player 0 takes action 2 at t=5
        actions[:, 10, 1] = 2  # player 1 takes action 2 at t=10
        traj = Trajectory(actions=actions)
        result_p0 = first_action_timestep(traj, action_id=2, player=0)
        result_p1 = first_action_timestep(traj, action_id=2, player=1)
        np.testing.assert_array_equal(result_p0, 5)
        np.testing.assert_array_equal(result_p1, 10)

    # ---- plot functions ----

    def test_plot_achievement_timing_returns_fig_ax(self) -> None:
        """plot_achievement_timing returns a (Figure, Axes) tuple."""
        traj = make_multi_player_traj(8, 40, num_p=2)
        fig, ax = plot_achievement_timing(traj)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close("all")

    def test_plot_achievement_timing_all_negative_no_crash(self) -> None:
        """plot_achievement_timing handles all-negative timing without error."""
        num_eps, num_steps = 4, 20
        achievements = np.zeros((num_eps, num_steps, 2), dtype=bool)
        traj = Trajectory(
            actions=np.zeros((num_eps, num_steps), dtype=np.int32),
            achievements=achievements,
        )
        fig, ax = plot_achievement_timing(traj)
        assert isinstance(fig, Figure)
        plt.close("all")

    def test_plot_achievement_progress_returns_fig_ax(self) -> None:
        """plot_achievement_progress returns a (Figure, Axes) tuple."""
        traj = make_multi_player_traj(8, 40, num_p=2)
        fig, ax = plot_achievement_progress(traj)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close("all")

    def test_plot_first_action_timing_returns_fig_ax(self) -> None:
        """plot_first_action_timing returns a (Figure, Axes) tuple."""
        traj = make_single_player_traj(8, 40, num_actions=6)
        fig, ax = plot_first_action_timing(traj, action_ids=[1, 2, 3])
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close("all")


# ---------------------------------------------------------------------------
# TestMultiagent
# ---------------------------------------------------------------------------
