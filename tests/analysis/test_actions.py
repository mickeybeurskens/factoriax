"""Tests for :mod:`factoriax.analysis.actions`.

Two groups live here. The statistics group covers ``action_entropy``,
``action_ngrams``, ``run_lengths``, and ``transition_matrix``, plus the plot
function that draws each one. The palette group covers
``DEFAULT_ACTION_LABELS`` and ``DEFAULT_ACTION_COLORS`` against the ``Action``
enum, because a short list makes a colormap clamp and renders unlike actions
in the same color.

The ``conftest.py`` beside this file pins matplotlib to the Agg backend, so a
plot test needs no display.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from factoriax.analysis.actions import (
    DEFAULT_ACTION_COLORS,
    DEFAULT_ACTION_LABELS,
    _get_action_cmap,
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
from factoriax.analysis.trajectory import Trajectory
from factoriax.engine.constants import NUM_ACTIONS, Action
from tests.helpers.trajectories import (
    make_multi_player_traj,
    make_single_player_traj,
)


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


class TestActionLabels:
    """``DEFAULT_ACTION_LABELS`` against the ``Action`` enum."""

    def test_covers_every_action(self) -> None:
        """The list holds one name for each action."""
        assert len(DEFAULT_ACTION_LABELS) == NUM_ACTIONS

    def test_entry_names_the_action_with_that_value(self) -> None:
        """Entry ``i`` is the name of the action whose value is ``i``."""
        for i in range(NUM_ACTIONS):
            assert DEFAULT_ACTION_LABELS[i] == Action(i).name


class TestActionColors:
    """``DEFAULT_ACTION_COLORS`` against the ``Action`` enum."""

    def test_covers_every_action(self) -> None:
        """The list holds one color for each action.

        A shorter list is what makes the colormap short, and a short
        colormap silently reuses its last color.
        """
        assert len(DEFAULT_ACTION_COLORS) == NUM_ACTIONS

    def test_colors_are_distinct(self) -> None:
        """No two actions share a color."""
        assert len(set(DEFAULT_ACTION_COLORS)) == len(DEFAULT_ACTION_COLORS)

    def test_default_colormap_separates_every_action(self) -> None:
        """A default colormap gives each action its own color.

        This is the failure a reader meets. With a short color list,
        ``cmap(i)`` clamps. Dozens of actions then render alike, in
        the raster and in its legend.
        """
        cmap = _get_action_cmap(NUM_ACTIONS)
        assert cmap.N == NUM_ACTIONS
        assert len({cmap(i) for i in range(NUM_ACTIONS)}) == NUM_ACTIONS

