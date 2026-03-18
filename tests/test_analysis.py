"""Tests for factoriax/analysis/ module.

Covers trajectory construction, action analysis, state analysis,
milestone/achievement analysis, multi-agent analysis, and rollout recording.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # Must be before any pyplot import

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from factoriax.analysis.actions import (
    action_entropy,
    action_ngrams,
    action_raster,
    plot_action_distribution,
    plot_entropy,
    plot_ngrams,
    plot_run_lengths,
    plot_transition_matrix,
    run_lengths,
    transition_matrix,
)
from factoriax.analysis.milestones import (
    achievement_timing,
    first_action_timestep,
    plot_achievement_progress,
    plot_achievement_timing,
    plot_first_action_timing,
)
from factoriax.analysis.multiagent import (
    joint_action_matrix,
    plot_comparative_raster,
    plot_joint_actions,
    plot_role_divergence,
    plot_spatial_overlap,
    role_divergence,
    spatial_overlap,
)
from factoriax.analysis.recorder import RolloutRecorder
from factoriax.analysis.state import (
    inventory_over_time,
    plot_inventory,
    plot_position_heatmap,
    plot_resource_depletion,
    position_heatmap,
)
from factoriax.analysis.trajectory import Trajectory

# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------


def make_single_player_traj(
    num_eps: int, num_steps: int, num_actions: int = 6, seed: int = 0
) -> Trajectory:
    """Build a (num_eps, num_steps) single-player trajectory with random actions.

    Args:
        num_eps: Number of episodes.
        num_steps: Episode length.
        num_actions: Action space size.
        seed: RNG seed for reproducibility.

    Returns:
        Single-player Trajectory with integer actions in [0, num_actions).
    """
    rng = np.random.default_rng(seed)
    actions = rng.integers(0, num_actions, size=(num_eps, num_steps)).astype(np.int32)
    return Trajectory(actions=actions)


def make_multi_player_traj(
    num_eps: int,
    num_steps: int,
    num_p: int = 2,
    num_actions: int = 6,
    seed: int = 0,
) -> Trajectory:
    """Build a multi-player trajectory with all optional fields populated.

    Args:
        num_eps: Number of episodes.
        num_steps: Episode length.
        num_p: Number of players.
        num_actions: Action space size.
        seed: RNG seed for reproducibility.

    Returns:
        Multi-player Trajectory with actions, positions, inventory, achievements,
        rewards, and timesteps.
    """
    rng = np.random.default_rng(seed)
    actions = rng.integers(0, num_actions, size=(num_eps, num_steps, num_p)).astype(
        np.int32
    )
    positions = rng.integers(0, 32, size=(num_eps, num_steps, num_p, 2)).astype(
        np.int32
    )
    inventory_items = rng.integers(0, 3, size=(num_eps, num_steps, num_p, 3)).astype(
        np.int32
    )
    inventory_counts = rng.integers(0, 20, size=(num_eps, num_steps, num_p, 3)).astype(
        np.int32
    )
    achievements = np.zeros((num_eps, num_steps, 2), dtype=bool)
    achievements[:, num_steps // 2 :, 0] = True
    achievements[:, num_steps // 4 :, 1] = True
    rewards = rng.standard_normal(size=(num_eps, num_steps)).astype(np.float32)
    timesteps = (
        np.broadcast_to(np.arange(num_steps), (num_eps, num_steps))
        .copy()
        .astype(np.int32)
    )
    return Trajectory(
        actions=actions,
        positions=positions,
        inventory_items=inventory_items,
        inventory_counts=inventory_counts,
        achievements=achievements,
        rewards=rewards,
        timesteps=timesteps,
    )


def make_minimal_traj(num_eps: int, num_steps: int) -> Trajectory:
    """Build a trajectory with only zero actions and no optional fields.

    Args:
        num_eps: Number of episodes.
        num_steps: Episode length.

    Returns:
        Minimal Trajectory for error-condition tests.
    """
    return Trajectory(actions=np.zeros((num_eps, num_steps), dtype=np.int32))


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


# ---------------------------------------------------------------------------
# TestTrajectory
# ---------------------------------------------------------------------------


class TestTrajectory:
    """Tests for Trajectory construction, shape properties, slicing, and I/O."""

    def test_1d_actions_coerced(self) -> None:
        """1D action array is promoted to (1, T)."""
        traj = Trajectory(actions=np.zeros(50, dtype=np.int32))
        assert traj.num_episodes == 1
        assert traj.episode_length == 50

    def test_2d_single_player(self) -> None:
        """2D actions give single-player trajectory."""
        traj = make_single_player_traj(8, 50)
        assert traj.is_multi_player is False
        assert traj.num_players == 1

    def test_3d_multi_player(self) -> None:
        """3D actions give multi-player trajectory with correct player count."""
        traj = make_multi_player_traj(8, 50, num_p=3)
        assert traj.is_multi_player is True
        assert traj.num_players == 3

    def test_4d_raises_value_error(self) -> None:
        """4D action array raises ValueError."""
        with pytest.raises(ValueError):
            Trajectory(actions=np.zeros((2, 3, 4, 5), dtype=np.int32))

    def test_episode_returns_one_episode(self) -> None:
        """episode(i) returns a single-episode Trajectory with correct data."""
        traj = make_single_player_traj(8, 50)
        ep = traj.episode(2)
        assert ep.num_episodes == 1
        np.testing.assert_array_equal(ep.actions[0], traj.actions[2])

    def test_episode_negative_index(self) -> None:
        """episode(-1) returns the last episode."""
        traj = make_single_player_traj(8, 50)
        ep = traj.episode(-1)
        assert ep.num_episodes == 1
        np.testing.assert_array_equal(ep.actions[0], traj.actions[-1])

    def test_episodes_slice(self) -> None:
        """episodes(slice(1, 3)) returns two episodes."""
        traj = make_single_player_traj(8, 50)
        sliced = traj.episodes(slice(1, 3))
        assert sliced.actions.shape == (2, 50)

    def test_episodes_array_index(self) -> None:
        """episodes(np.array([0, 2, 4])) returns three specific episodes."""
        traj = make_single_player_traj(8, 50)
        indexed = traj.episodes(np.array([0, 2, 4]))
        assert indexed.actions.shape == (3, 50)

    def test_player_on_single_player(self) -> None:
        """player(0) on a single-player trajectory returns a single-player view."""
        traj = make_single_player_traj(8, 50)
        result = traj.player(0)
        assert result.is_multi_player is False

    def test_player_extracts_correct_player(self) -> None:
        """player(1) actions match traj.actions[:, :, 1]."""
        traj = make_multi_player_traj(8, 50, num_p=2)
        p1 = traj.player(1)
        np.testing.assert_array_equal(p1.actions, traj.actions[:, :, 1])

    def test_player_preserves_optional_fields(self) -> None:
        """player(0) squeezes positions to (num_eps, num_steps, 2)."""
        traj = make_multi_player_traj(8, 50, num_p=2)
        p0 = traj.player(0)
        assert p0.positions is not None
        assert p0.positions.shape == (8, 50, 2)

    def test_time_slice_shape(self) -> None:
        """time_slice(10, 40) returns episode_length == 30."""
        traj = make_single_player_traj(8, 50)
        sliced = traj.time_slice(10, 40)
        assert sliced.episode_length == 30

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        """All array shapes and dtypes survive save/load."""
        traj = make_multi_player_traj(4, 30, num_p=2)
        save_path = str(tmp_path / "traj.npz")
        traj.save(save_path)
        loaded = Trajectory.load(save_path)
        np.testing.assert_array_equal(loaded.actions, traj.actions)
        assert loaded.positions is not None
        assert traj.positions is not None
        assert loaded.positions.shape == traj.positions.shape
        assert loaded.actions.dtype == traj.actions.dtype
        assert loaded.achievements is not None
        assert traj.achievements is not None
        assert loaded.achievements.shape == traj.achievements.shape


# ---------------------------------------------------------------------------
# TestActions
# ---------------------------------------------------------------------------


class TestActions:
    """Tests for action sequence analysis functions."""

    # ---- transition_matrix ----

    @pytest.mark.parametrize("num_players", [1, 2])
    def test_transition_matrix_shape(self, num_players: int) -> None:
        """Transition matrix has shape (num_actions, num_actions)."""
        num_actions = 6
        if num_players == 1:
            traj = make_single_player_traj(4, 50, num_actions=num_actions)
        else:
            traj = make_multi_player_traj(4, 50, num_p=2, num_actions=num_actions)
        mat = transition_matrix(traj, player=0, num_actions=num_actions)
        assert mat.shape == (num_actions, num_actions)

    @pytest.mark.parametrize("num_players", [1, 2])
    def test_transition_matrix_rows_sum_to_one(self, num_players: int) -> None:
        """Normalized transition matrix rows sum to 1 (or 0 for absent actions)."""
        num_actions = 6
        if num_players == 1:
            traj = make_single_player_traj(4, 50, num_actions=num_actions)
        else:
            traj = make_multi_player_traj(4, 50, num_p=2, num_actions=num_actions)
        mat = transition_matrix(traj, player=0, num_actions=num_actions, normalize=True)
        row_sums = mat.sum(axis=1)
        for row_sum in row_sums:
            assert row_sum == pytest.approx(0.0) or row_sum == pytest.approx(1.0)

    @pytest.mark.parametrize("num_players", [1, 2])
    def test_transition_matrix_no_nan(self, num_players: int) -> None:
        """Transition matrix contains no NaN values, even for zero rows."""
        num_actions = 6
        if num_players == 1:
            traj = make_single_player_traj(4, 50, num_actions=num_actions)
        else:
            traj = make_multi_player_traj(4, 50, num_p=2, num_actions=num_actions)
        mat = transition_matrix(traj, player=0, num_actions=num_actions)
        assert not np.any(np.isnan(mat))

    @pytest.mark.parametrize("num_players", [1, 2])
    def test_transition_matrix_unnormalized_total(self, num_players: int) -> None:
        """Unnormalized matrix sums to num_eps * (num_steps - 1)."""
        num_eps, num_steps, num_actions = 4, 50, 6
        if num_players == 1:
            traj = make_single_player_traj(num_eps, num_steps, num_actions=num_actions)
        else:
            traj = make_multi_player_traj(
                num_eps, num_steps, num_p=2, num_actions=num_actions
            )
        mat = transition_matrix(
            traj, player=0, num_actions=num_actions, normalize=False
        )
        assert mat.sum() == pytest.approx(num_eps * (num_steps - 1))

    # ---- action_ngrams ----

    @pytest.mark.parametrize("num_players", [1, 2])
    def test_action_ngrams_most_common_bigram(self, num_players: int) -> None:
        """Bigram (0, 1) is most common in the sequence [0, 1, 0, 1]."""
        if num_players == 1:
            actions = np.array([[0, 1, 0, 1]], dtype=np.int32)
        else:
            actions = np.stack(
                [np.array([[0, 1, 0, 1]], dtype=np.int32)] * 2, axis=-1
            )  # (1, 4, 2)
        traj = Trajectory(actions=actions)
        ngrams = action_ngrams(traj, n=2, player=0, top_k=5)
        top_gram, top_count = ngrams[0]
        assert top_gram in {(0, 1), (1, 0)}  # type: ignore[comparison-overlap]
        assert top_count >= 2

    # ---- action_entropy ----

    @pytest.mark.parametrize("num_players", [1, 2])
    def test_action_entropy_constant_is_zero(self, num_players: int) -> None:
        """Entropy is 0 when all episodes take the same action at every timestep."""
        num_eps, num_steps, num_actions = 12, 20, 6
        if num_players == 1:
            actions = np.zeros((num_eps, num_steps), dtype=np.int32)
        else:
            actions = np.zeros((num_eps, num_steps, 2), dtype=np.int32)
        traj = Trajectory(actions=actions)
        ent = action_entropy(traj, player=0, num_actions=num_actions)
        np.testing.assert_array_almost_equal(ent, 0.0)

    @pytest.mark.parametrize("num_players", [1, 2])
    def test_action_entropy_uniform_is_max(self, num_players: int) -> None:
        """Entropy equals log2(num_actions) when action distribution is uniform."""
        num_actions = 6
        num_eps = 12  # divisible by num_actions
        num_steps = 20
        # Each action appears num_eps // num_actions times per timestep
        base = np.tile(np.arange(num_actions), num_eps // num_actions).reshape(
            num_eps, 1
        )
        base_actions = (
            np.broadcast_to(base, (num_eps, num_steps)).copy().astype(np.int32)
        )
        if num_players == 1:
            actions = base_actions
        else:
            actions = np.stack([base_actions, base_actions], axis=-1)
        traj = Trajectory(actions=actions)
        ent = action_entropy(traj, player=0, num_actions=num_actions)
        expected = np.log2(num_actions)
        np.testing.assert_allclose(ent, expected, atol=1e-6)

    # ---- run_lengths ----

    @pytest.mark.parametrize("num_players", [1, 2])
    def test_run_lengths_single_action(self, num_players: int) -> None:
        """Single repeated action produces one run of length num_steps."""
        num_steps = 10
        if num_players == 1:
            actions = np.zeros((1, num_steps), dtype=np.int32)
        else:
            actions = np.zeros((1, num_steps, 2), dtype=np.int32)
        traj = Trajectory(actions=actions)
        result = run_lengths(traj, player=0)
        assert 0 in result
        assert result[0] == [num_steps]

    def test_run_lengths_empty_episode_no_crash(self) -> None:
        """Zero-length episodes are skipped without raising an error."""
        traj = Trajectory(actions=np.zeros((3, 0), dtype=np.int32))
        result = run_lengths(traj)
        assert result == {}

    # ---- plot functions return (Figure, Axes) ----

    def test_action_raster_returns_fig_ax(self) -> None:
        """action_raster returns a (Figure, Axes) tuple."""
        traj = make_single_player_traj(4, 30)
        fig, ax = action_raster(traj, num_actions=6)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close("all")

    def test_plot_transition_matrix_returns_fig_ax(self) -> None:
        """plot_transition_matrix returns a (Figure, Axes) tuple."""
        traj = make_single_player_traj(4, 30, num_actions=6)
        fig, ax = plot_transition_matrix(traj, num_actions=6)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close("all")

    def test_plot_entropy_returns_fig_ax(self) -> None:
        """plot_entropy returns a (Figure, Axes) tuple."""
        traj = make_single_player_traj(4, 30, num_actions=6)
        fig, ax = plot_entropy(traj, num_actions=6)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close("all")

    def test_plot_run_lengths_returns_fig_ax(self) -> None:
        """plot_run_lengths returns a (Figure, Axes) tuple."""
        traj = make_single_player_traj(4, 30, num_actions=6)
        fig, ax = plot_run_lengths(traj, num_actions=6)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close("all")

    def test_plot_ngrams_returns_fig_ax(self) -> None:
        """plot_ngrams returns a (Figure, Axes) tuple."""
        traj = make_single_player_traj(4, 30, num_actions=6)
        fig, ax = plot_ngrams(traj)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close("all")

    def test_plot_action_distribution_returns_fig_ax(self) -> None:
        """plot_action_distribution returns a (Figure, Axes) tuple."""
        traj = make_single_player_traj(4, 30, num_actions=6)
        fig, ax = plot_action_distribution(traj, num_actions=6)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close("all")


# ---------------------------------------------------------------------------
# TestState
# ---------------------------------------------------------------------------


class TestState:
    """Tests for state evolution analysis (inventory, positions, resources)."""

    def _make_single_player_inventory_traj(
        self, num_eps: int, num_steps: int, num_slots: int = 3
    ) -> Trajectory:
        """Build a single-player trajectory with inventory fields shaped
        (num_eps, num_steps, num_slots).

        Args:
            num_eps: Number of episodes.
            num_steps: Episode length.
            num_slots: Inventory slot count.

        Returns:
            Trajectory with randomly populated inventory fields.
        """
        rng = np.random.default_rng(42)
        actions = np.zeros((num_eps, num_steps), dtype=np.int32)
        inventory_items = rng.integers(
            0, 3, size=(num_eps, num_steps, num_slots)
        ).astype(np.int32)
        inventory_counts = rng.integers(
            0, 20, size=(num_eps, num_steps, num_slots)
        ).astype(np.int32)
        return Trajectory(
            actions=actions,
            inventory_items=inventory_items,
            inventory_counts=inventory_counts,
        )

    # ---- inventory_over_time ----

    def test_inventory_over_time_shape(self) -> None:
        """inventory_over_time returns shape (num_steps, num_item_types)."""
        num_eps, num_steps, num_item_types = 4, 20, 3
        traj = self._make_single_player_inventory_traj(num_eps, num_steps)
        result = inventory_over_time(traj, player=0, num_item_types=num_item_types)
        assert result.shape == (num_steps, num_item_types)

    def test_inventory_over_time_raises_without_fields(self) -> None:
        """inventory_over_time raises ValueError when inventory fields are absent."""
        traj = make_minimal_traj(4, 20)
        with pytest.raises(ValueError):
            inventory_over_time(traj)

    def test_inventory_over_time_values_correct(self) -> None:
        """Counts for known item distribution match expected totals."""
        num_eps, num_steps, num_slots = 4, 20, 3
        # All slots contain item type 1 with count 2
        inventory_items = np.ones((num_eps, num_steps, num_slots), dtype=np.int32)
        inventory_counts = np.full((num_eps, num_steps, num_slots), 2, dtype=np.int32)
        traj = Trajectory(
            actions=np.zeros((num_eps, num_steps), dtype=np.int32),
            inventory_items=inventory_items,
            inventory_counts=inventory_counts,
        )
        result = inventory_over_time(traj, player=0, num_item_types=3)
        # num_slots=3 × count=2 = 6 for item type 1; 0 for others
        np.testing.assert_array_almost_equal(result[:, 1], 6.0)
        np.testing.assert_array_almost_equal(result[:, 0], 0.0)
        np.testing.assert_array_almost_equal(result[:, 2], 0.0)

    def test_inventory_over_time_multi_player(self) -> None:
        """inventory_over_time works on multi-player trajectory for one player."""
        traj = make_multi_player_traj(4, 20, num_p=2)
        result = inventory_over_time(traj, player=0, num_item_types=3)
        assert result.shape == (20, 3)

    # ---- position_heatmap ----

    def test_position_heatmap_shape(self) -> None:
        """position_heatmap returns shape (map_height, map_width)."""
        traj = make_multi_player_traj(4, 20, num_p=2)
        hm = position_heatmap(traj, player=0, map_width=32, map_height=32)
        assert hm.shape == (32, 32)

    def test_position_heatmap_sums_to_one(self) -> None:
        """position_heatmap sums to 1.0."""
        traj = make_multi_player_traj(4, 20, num_p=2)
        hm = position_heatmap(traj, player=0, map_width=32, map_height=32)
        assert hm.sum() == pytest.approx(1.0)

    def test_position_heatmap_all_same_position(self) -> None:
        """All mass concentrates at the single visited cell."""
        num_eps, num_steps, num_p = 4, 20, 2
        positions = np.zeros((num_eps, num_steps, num_p, 2), dtype=np.int32)
        positions[:, :, :, 0] = 5  # x = 5
        positions[:, :, :, 1] = 7  # y = 7
        traj = Trajectory(
            actions=np.zeros((num_eps, num_steps, num_p), dtype=np.int32),
            positions=positions,
        )
        hm = position_heatmap(traj, player=0, map_width=32, map_height=32)
        assert hm[7, 5] == pytest.approx(1.0)
        mask = np.zeros((32, 32), dtype=bool)
        mask[7, 5] = True
        assert hm[~mask].sum() == pytest.approx(0.0)

    def test_position_heatmap_clips_out_of_bounds(self) -> None:
        """Out-of-bounds coordinates are clipped to boundary, not dropped."""
        num_eps, num_steps, num_p = 2, 10, 2
        # Both players at coordinate 100, which clips to (31, 31)
        positions = np.full((num_eps, num_steps, num_p, 2), 100, dtype=np.int32)
        traj = Trajectory(
            actions=np.zeros((num_eps, num_steps, num_p), dtype=np.int32),
            positions=positions,
        )
        hm = position_heatmap(traj, player=0, map_width=32, map_height=32)
        assert hm.sum() == pytest.approx(1.0)
        assert hm[31, 31] == pytest.approx(1.0)

    def test_position_heatmap_raises_without_positions(self) -> None:
        """position_heatmap raises ValueError when positions field is absent."""
        traj = make_minimal_traj(4, 20)
        with pytest.raises(ValueError):
            position_heatmap(traj)

    def test_position_heatmap_time_range(self) -> None:
        """time_range restricts which timesteps contribute to the heatmap."""
        traj = make_multi_player_traj(4, 20, num_p=2)
        hm_full = position_heatmap(traj, player=0)
        hm_partial = position_heatmap(traj, player=0, time_range=(0, 5))
        assert hm_full.shape == hm_partial.shape
        assert hm_partial.sum() == pytest.approx(1.0)

    def test_plot_inventory_returns_fig_ax(self) -> None:
        """plot_inventory returns a (Figure, Axes) tuple."""
        traj = self._make_single_player_inventory_traj(4, 20)
        fig, ax = plot_inventory(traj, player=0, num_item_types=3)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close("all")

    def test_plot_position_heatmap_returns_fig_ax(self) -> None:
        """plot_position_heatmap returns a (Figure, Axes) tuple."""
        traj = make_multi_player_traj(4, 20, num_p=2)
        fig, ax = plot_position_heatmap(traj, player=0)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close("all")

    def test_plot_resource_depletion_raises_without_metadata(self) -> None:
        """plot_resource_depletion raises ValueError when block_resources is missing."""
        traj = make_minimal_traj(4, 20)
        with pytest.raises(ValueError):
            plot_resource_depletion(traj)

    def test_plot_resource_depletion_raises_keyerror_converted(self) -> None:
        """Missing block_resources raises ValueError, not KeyError."""
        traj = Trajectory(
            actions=np.zeros((4, 20), dtype=np.int32),
            metadata={"other_key": 42},
        )
        with pytest.raises(ValueError):
            plot_resource_depletion(traj)

    def test_plot_resource_depletion_with_metadata_returns_fig_ax(self) -> None:
        """plot_resource_depletion returns (Figure, Axes) when metadata is present."""
        num_eps, num_steps = 4, 20
        height, width = 8, 8
        resources = np.random.randint(
            0, 50, size=(num_eps, num_steps, height, width)
        ).astype(np.float32)
        traj = Trajectory(
            actions=np.zeros((num_eps, num_steps), dtype=np.int32),
            metadata={"block_resources": resources},
        )
        fig, ax = plot_resource_depletion(traj)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close("all")


# ---------------------------------------------------------------------------
# TestMilestones
# ---------------------------------------------------------------------------


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


class TestMultiagent:
    """Tests for multi-agent coordination analysis."""

    # ---- role_divergence ----

    def test_role_divergence_raises_for_single_player(self) -> None:
        """role_divergence raises ValueError on single-player input."""
        traj = make_single_player_traj(4, 20)
        with pytest.raises(ValueError):
            role_divergence(traj)

    def test_role_divergence_diagonal_is_zero(self) -> None:
        """JSD of a player with itself is 0."""
        traj = make_multi_player_traj(4, 40, num_p=2)
        jsd = role_divergence(traj, num_actions=6)
        np.testing.assert_array_almost_equal(np.diag(jsd), 0.0)

    def test_role_divergence_symmetric(self) -> None:
        """JSD matrix is symmetric."""
        traj = make_multi_player_traj(4, 40, num_p=2)
        jsd = role_divergence(traj, num_actions=6)
        np.testing.assert_array_almost_equal(jsd, jsd.T)

    def test_role_divergence_values_in_unit_interval(self) -> None:
        """All JSD values lie in [0, 1]."""
        traj = make_multi_player_traj(4, 40, num_p=3, num_actions=6)
        jsd = role_divergence(traj, num_actions=6)
        assert np.all(jsd >= -1e-9)
        assert np.all(jsd <= 1.0 + 1e-9)

    def test_role_divergence_identical_players_near_zero(self) -> None:
        """Players with identical action distributions have near-zero JSD."""
        num_eps, num_steps = 8, 40
        base_actions = (
            np.random.default_rng(0)
            .integers(0, 6, size=(num_eps, num_steps))
            .astype(np.int32)
        )
        actions = np.stack([base_actions, base_actions], axis=-1)
        traj = Trajectory(actions=actions)
        jsd = role_divergence(traj, num_actions=6)
        assert jsd[0, 1] < 0.01

    def test_role_divergence_disjoint_distributions_near_one(self) -> None:
        """Players with disjoint action distributions have JSD near 1."""
        num_eps, num_steps = 4, 40
        actions = np.zeros((num_eps, num_steps, 2), dtype=np.int32)
        actions[:, :, 0] = 0  # player 0 always action 0
        actions[:, :, 1] = 1  # player 1 always action 1
        traj = Trajectory(actions=actions)
        jsd = role_divergence(traj, num_actions=6)
        assert jsd[0, 1] == pytest.approx(1.0, abs=1e-6)

    def test_role_divergence_with_time_range(self) -> None:
        """role_divergence accepts time_range without error."""
        traj = make_multi_player_traj(4, 40, num_p=2)
        jsd = role_divergence(traj, num_actions=6, time_range=(0, 20))
        assert jsd.shape == (2, 2)

    # ---- spatial_overlap ----

    def test_spatial_overlap_raises_for_single_player(self) -> None:
        """spatial_overlap raises ValueError on single-player input."""
        traj = make_single_player_traj(4, 20)
        with pytest.raises(ValueError):
            spatial_overlap(traj)

    def test_spatial_overlap_raises_without_positions(self) -> None:
        """spatial_overlap raises ValueError when positions field is absent."""
        traj = make_multi_player_traj(4, 20, num_p=2)
        traj_no_pos = Trajectory(actions=traj.actions)
        with pytest.raises(ValueError):
            spatial_overlap(traj_no_pos)

    def test_spatial_overlap_all_same_position_is_one(self) -> None:
        """Overlap is 1.0 when all players are at the same tile."""
        num_eps, num_steps, num_p = 4, 20, 2
        positions = np.zeros((num_eps, num_steps, num_p, 2), dtype=np.int32)
        traj = Trajectory(
            actions=np.zeros((num_eps, num_steps, num_p), dtype=np.int32),
            positions=positions,
        )
        overlap = spatial_overlap(traj, distance_threshold=0)
        np.testing.assert_array_almost_equal(overlap, 1.0)

    def test_spatial_overlap_far_apart_is_zero(self) -> None:
        """Overlap is 0.0 when players are maximally separated."""
        num_eps, num_steps, num_p = 4, 20, 2
        positions = np.zeros((num_eps, num_steps, num_p, 2), dtype=np.int32)
        positions[:, :, 1, :] = 31  # player 1 at far corner (L1 distance = 62)
        traj = Trajectory(
            actions=np.zeros((num_eps, num_steps, num_p), dtype=np.int32),
            positions=positions,
        )
        overlap = spatial_overlap(traj, distance_threshold=0)
        np.testing.assert_array_almost_equal(overlap, 0.0)

    def test_spatial_overlap_larger_threshold_monotone(self) -> None:
        """Larger distance thresholds produce overlap values >= smaller thresholds."""
        traj = make_multi_player_traj(4, 20, num_p=2)
        overlap_0 = spatial_overlap(traj, distance_threshold=0)
        overlap_5 = spatial_overlap(traj, distance_threshold=5)
        assert np.all(overlap_5 >= overlap_0 - 1e-9)

    def test_spatial_overlap_shape(self) -> None:
        """spatial_overlap returns shape (num_steps,)."""
        traj = make_multi_player_traj(4, 20, num_p=2)
        overlap = spatial_overlap(traj, distance_threshold=0)
        assert overlap.shape == (20,)

    # ---- joint_action_matrix ----

    def test_joint_action_matrix_raises_for_single_player(self) -> None:
        """joint_action_matrix raises ValueError on single-player input."""
        traj = make_single_player_traj(4, 20)
        with pytest.raises(ValueError):
            joint_action_matrix(traj)

    def test_joint_action_matrix_shape(self) -> None:
        """joint_action_matrix returns shape (num_actions, num_actions)."""
        num_actions = 6
        traj = make_multi_player_traj(4, 20, num_p=2, num_actions=num_actions)
        mat = joint_action_matrix(traj, num_actions=num_actions)
        assert mat.shape == (num_actions, num_actions)

    def test_joint_action_matrix_normalized_sums_to_one(self) -> None:
        """Normalized joint action matrix sums to 1.0."""
        traj = make_multi_player_traj(4, 20, num_p=2, num_actions=6)
        mat = joint_action_matrix(traj, num_actions=6, normalize=True)
        assert mat.sum() == pytest.approx(1.0)

    def test_joint_action_matrix_unnormalized_total(self) -> None:
        """Unnormalized matrix sums to num_eps * num_steps."""
        num_eps, num_steps = 4, 20
        traj = make_multi_player_traj(num_eps, num_steps, num_p=2, num_actions=6)
        mat = joint_action_matrix(traj, num_actions=6, normalize=False)
        assert mat.sum() == pytest.approx(num_eps * num_steps)

    # ---- plot functions ----

    def test_plot_comparative_raster_returns_fig_axes(self) -> None:
        """plot_comparative_raster returns (Figure, ndarray of Axes)."""
        traj = make_multi_player_traj(4, 30, num_p=2, num_actions=6)
        fig, axes = plot_comparative_raster(traj, num_actions=6)
        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        plt.close("all")

    def test_plot_role_divergence_returns_fig_ax(self) -> None:
        """plot_role_divergence returns a (Figure, Axes) tuple."""
        traj = make_multi_player_traj(4, 30, num_p=2, num_actions=6)
        fig, ax = plot_role_divergence(traj, num_actions=6)
        assert isinstance(fig, Figure)
        plt.close("all")

    def test_plot_spatial_overlap_returns_fig_ax(self) -> None:
        """plot_spatial_overlap returns a (Figure, Axes) tuple."""
        traj = make_multi_player_traj(4, 30, num_p=2)
        fig, ax = plot_spatial_overlap(traj)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close("all")

    def test_plot_joint_actions_returns_fig_ax(self) -> None:
        """plot_joint_actions returns a (Figure, Axes) tuple."""
        traj = make_multi_player_traj(4, 30, num_p=2, num_actions=6)
        fig, ax = plot_joint_actions(traj, num_actions=6)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close("all")


# ---------------------------------------------------------------------------
# TestRolloutRecorder
# ---------------------------------------------------------------------------


class TestRolloutRecorder:
    """Tests for RolloutRecorder: segmentation, padding, and metadata."""

    def _make_rollout(
        self,
        num_steps: int,
        num_envs: int,
        done_steps: list[tuple[int, int]] | None = None,
    ) -> FakeRollout:
        """Build a FakeRollout with specified done positions.

        Args:
            num_steps: Number of timesteps.
            num_envs: Number of environments.
            done_steps: List of (t, env_idx) pairs where done=True.

        Returns:
            FakeRollout with sequential action values for easy verification.
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
        # Episode 0 has length 3; positions 3..max_len should be zero-padded
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

    def test_metadata_contains_recording_statistics(self) -> None:
        """Output Trajectory metadata includes recording statistics."""
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
        assert "num_envs" in traj.metadata
        assert "total_steps_recorded" in traj.metadata
        assert "num_complete_episodes" in traj.metadata

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
