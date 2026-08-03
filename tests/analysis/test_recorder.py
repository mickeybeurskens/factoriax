"""Tests for :mod:`factoriax.analysis.recorder`.

:class:`RolloutRecorder` collects the per-step arrays of a rollout and pads
them into the rectangular shape that :class:`Trajectory` needs. Episodes end
at different timesteps, so the padding rule and the step limit are the two
parts of the contract that matter here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from factoriax.analysis.recorder import RolloutRecorder


@dataclass
class FakeRollout:
    """Minimal rollout struct for testing RolloutRecorder.

    Attributes:
        action: Action array shaped (num_steps, num_envs).
        reward: Reward array shaped (num_steps, num_envs).
        done: Done flags shaped (num_steps, num_envs).
    """

    action: np.ndarray
    reward: np.ndarray
    done: np.ndarray

class TestRolloutRecorder:
    """Tests for RolloutRecorder: segmentation, padding, and output shape."""

    def _make_rollout(
        self,
        num_steps: int,
        num_envs: int,
        done_steps: list[tuple[int, int]] | None = None,
    ) -> FakeRollout:
        """Build a FakeRollout with specified done positions.

        Parameters
        ----------
        num_steps
            Number of timesteps.
        num_envs
            Number of environments.
        done_steps
            ``(t, env_idx)`` pairs where ``done`` is True.

        Returns
        -------
        FakeRollout
            Rollout with sequential action values for easy verification.
        """
        actions = (
            np.arange(num_steps * num_envs, dtype=np.int32).reshape(num_steps, num_envs)
            % 10
        )
        rewards = np.zeros((num_steps, num_envs), dtype=np.float32)
        dones = np.zeros((num_steps, num_envs), dtype=bool)
        if done_steps:
            for t_idx, env_idx in done_steps:
                dones[t_idx, env_idx] = True
        return FakeRollout(action=actions, reward=rewards, done=dones)

    def test_finish_empty_raises(self) -> None:
        """finish() on an empty recorder raises ValueError."""
        rec = RolloutRecorder()
        with pytest.raises(ValueError):
            rec.finish()

    def test_is_full_without_max_episodes(self) -> None:
        """is_full is always False when max_episodes is not set."""
        rec = RolloutRecorder()
        assert rec.is_full is False
        rollout = self._make_rollout(5, 1, done_steps=[(4, 0)])
        rec.record(rollout)
        assert rec.is_full is False

    def test_is_full_with_max_episodes(self) -> None:
        """is_full becomes True once enough complete episodes are recorded."""
        rec = RolloutRecorder(max_episodes=2)
        assert rec.is_full is False
        rollout = self._make_rollout(10, 1, done_steps=[(4, 0), (9, 0)])
        rec.record(rollout)
        assert rec.is_full is True

    def test_num_recorded_steps_accumulates(self) -> None:
        """num_recorded_steps adds up correctly across multiple record() calls."""
        rec = RolloutRecorder()
        rollout = self._make_rollout(5, 2)
        rec.record(rollout)
        assert rec.num_recorded_steps == 5
        rec.record(rollout)
        assert rec.num_recorded_steps == 10

    def test_single_env_single_episode_action_values(self) -> None:
        """Actions in the output trajectory match the input rollout exactly."""
        num_steps, num_envs = 5, 1
        actions_arr = np.array([[1, 2, 3, 4, 5]], dtype=np.int32).T  # (5, 1)
        dones = np.zeros((num_steps, num_envs), dtype=bool)
        dones[-1, 0] = True
        rollout = FakeRollout(
            action=actions_arr,
            reward=np.zeros((num_steps, num_envs), dtype=np.float32),
            done=dones,
        )
        rec = RolloutRecorder()
        rec.record(rollout)
        traj = rec.finish()
        np.testing.assert_array_equal(traj.actions[0], [1, 2, 3, 4, 5])

    def test_pad_incomplete_true_includes_partial_episode(self) -> None:
        """pad_incomplete=True includes the trailing unfinished episode."""
        rollout = self._make_rollout(10, 1, done_steps=[(4, 0)])
        rec = RolloutRecorder()
        rec.record(rollout)
        traj = rec.finish(pad_incomplete=True)
        assert traj.num_episodes == 2

    def test_pad_incomplete_false_excludes_partial_episode(self) -> None:
        """pad_incomplete=False only includes fully completed episodes."""
        rollout = self._make_rollout(10, 1, done_steps=[(4, 0)])
        rec = RolloutRecorder()
        rec.record(rollout)
        traj = rec.finish(pad_incomplete=False)
        assert traj.num_episodes == 1

    def test_no_complete_episodes_raises_when_no_pad(self) -> None:
        """finish(pad_incomplete=False) raises ValueError when no episodes complete."""
        rollout = self._make_rollout(5, 1, done_steps=None)
        rec = RolloutRecorder()
        rec.record(rollout)
        with pytest.raises(ValueError):
            rec.finish(pad_incomplete=False)

    def test_episodes_padded_to_uniform_length(self) -> None:
        """All episodes in the output trajectory have the same length."""
        rollout = self._make_rollout(10, 1, done_steps=[(2, 0), (7, 0)])
        rec = RolloutRecorder()
        rec.record(rollout)
        traj = rec.finish(pad_incomplete=True)
        assert traj.actions.ndim == 2
        assert traj.num_episodes >= 2

    def test_padding_values_are_zeros(self) -> None:
        """Padding positions at the end of short episodes contain zeros."""
        num_steps, num_envs = 10, 1
        # Episode 1 ends at t=2 (length 3), Episode 2 ends at t=7 (length 5)
        actions_arr = np.arange(1, num_steps + 1, dtype=np.int32).reshape(
            num_steps, num_envs
        )
        dones = np.zeros((num_steps, num_envs), dtype=bool)
        dones[2, 0] = True
        dones[7, 0] = True
        rollout = FakeRollout(
            action=actions_arr,
            reward=np.zeros((num_steps, num_envs), dtype=np.float32),
            done=dones,
        )
        rec = RolloutRecorder()
        rec.record(rollout)
        traj = rec.finish(pad_incomplete=True)
        # Episode 0 has length 3. Positions 3..max_len are zero-padded
        max_len = traj.actions.shape[1]
        if max_len > 3:
            np.testing.assert_array_equal(traj.actions[0, 3:], 0)

    def test_reset_clears_all_state(self) -> None:
        """reset() returns the recorder to its initial empty state."""
        rollout = self._make_rollout(5, 1, done_steps=[(4, 0)])
        rec = RolloutRecorder()
        rec.record(rollout)
        assert rec.num_recorded_steps == 5
        rec.reset()
        assert rec.num_recorded_steps == 0
        with pytest.raises(ValueError):
            rec.finish()

    def test_finish_returns_trajectory_without_schemes(self) -> None:
        """Recorder output has no scheme fields set by default."""
        num_steps, num_envs = 5, 2
        dones = np.zeros((num_steps, num_envs), dtype=bool)
        dones[-1, :] = True
        rollout = FakeRollout(
            action=np.zeros((num_steps, num_envs), dtype=np.int32),
            reward=np.zeros((num_steps, num_envs), dtype=np.float32),
            done=dones,
        )
        rec = RolloutRecorder()
        rec.record(rollout)
        traj = rec.finish()
        assert traj.observation_scheme is None
        assert traj.reward_scheme is None
        assert traj.cost_scheme is None

    def test_multi_env_produces_one_episode_per_env(self) -> None:
        """Each env completing one episode yields one episode per env."""
        num_steps, num_envs = 5, 3
        dones = np.zeros((num_steps, num_envs), dtype=bool)
        dones[-1, :] = True
        rollout = FakeRollout(
            action=np.zeros((num_steps, num_envs), dtype=np.int32),
            reward=np.zeros((num_steps, num_envs), dtype=np.float32),
            done=dones,
        )
        rec = RolloutRecorder()
        rec.record(rollout)
        traj = rec.finish(pad_incomplete=False)
        assert traj.num_episodes == num_envs

    def test_stops_recording_when_full(self) -> None:
        """record() is a no-op once is_full is True."""
        num_steps, num_envs = 5, 1
        rollout = self._make_rollout(num_steps, num_envs, done_steps=[(4, 0)])
        rec = RolloutRecorder(max_episodes=1)
        rec.record(rollout)
        assert rec.is_full is True
        rec.record(rollout)  # no-op
        assert rec.num_recorded_steps == num_steps  # unchanged

    def test_max_episodes_limits_output_count(self) -> None:
        """Output trajectory respects max_episodes limit."""
        num_steps, num_envs = 10, 2
        dones = np.zeros((num_steps, num_envs), dtype=bool)
        dones[4, :] = True
        dones[9, :] = True
        rollout = FakeRollout(
            action=np.zeros((num_steps, num_envs), dtype=np.int32),
            reward=np.zeros((num_steps, num_envs), dtype=np.float32),
            done=dones,
        )
        rec = RolloutRecorder(max_episodes=3)
        rec.record(rollout)
        traj = rec.finish(pad_incomplete=False)
        assert traj.num_episodes <= 3
