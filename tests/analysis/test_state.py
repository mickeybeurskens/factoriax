"""Tests for :mod:`factoriax.analysis.state`.

Three groups live here.

The reader group covers ``inventory_over_time`` and ``position_heatmap``,
plus the plot function that draws each one.

The shape group asserts that the readers accept what the engine produces.
``EnvState.player_inventory`` has shape ``(P, NUM_ITEM_TYPES)``, so a
recording stacked by
:func:`~factoriax.analysis.trajectory.states_to_trajectory` always carries a
player axis and comes out as ``(B, T, P, N)``. That holds for one player as
much as for several. ``Trajectory.is_multi_player`` counts players in the
action array and not in the inventory, so a one-player recording reports
``False`` while its inventory still has the player axis. A reader that
branches on ``is_multi_player`` therefore skips the player index on exactly
the recordings that still need it, and returns one axis too many.

The palette group covers ``DEFAULT_ITEM_LABELS`` and ``DEFAULT_ITEM_COLORS``
against the ``ItemType`` enum.

The ``conftest.py`` beside this file pins matplotlib to the Agg backend, so a
plot test needs no display.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from factoriax.analysis.state import (
    DEFAULT_ITEM_COLORS,
    DEFAULT_ITEM_LABELS,
    inventory_over_time,
    plot_inventory,
    plot_position_heatmap,
    plot_resource_depletion,
    position_heatmap,
)
from factoriax.analysis.trajectory import Trajectory
from factoriax.engine.constants import NUM_ITEM_TYPES, ItemType
from tests.helpers.trajectories import make_minimal_traj, make_multi_player_traj


class TestState:
    """Tests for state evolution analysis (inventory, positions, resources)."""

    def _make_single_player_inventory_traj(
        self, num_eps: int, num_steps: int, num_item_types: int = 3
    ) -> Trajectory:
        """Build a single-player trajectory with player_inventory shaped
        ``(num_eps, num_steps, num_item_types)``.

        Parameters
        ----------
        num_eps
            Number of episodes.
        num_steps
            Episode length.
        num_item_types
            Distinct item types to populate.

        Returns
        -------
        Trajectory
            Trajectory with a randomly populated ``player_inventory``.
        """
        rng = np.random.default_rng(42)
        actions = np.zeros((num_eps, num_steps), dtype=np.int32)
        player_inventory = rng.integers(
            0, 20, size=(num_eps, num_steps, num_item_types)
        ).astype(np.int32)
        return Trajectory(
            actions=actions,
            player_inventory=player_inventory,
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
        num_eps, num_steps, num_item_types = 4, 20, 3
        # Each step the player has 6 of item type 1, 0 of the others.
        player_inventory = np.zeros(
            (num_eps, num_steps, num_item_types), dtype=np.int32
        )
        player_inventory[..., 1] = 6
        traj = Trajectory(
            actions=np.zeros((num_eps, num_steps), dtype=np.int32),
            player_inventory=player_inventory,
        )
        result = inventory_over_time(traj, player=0, num_item_types=num_item_types)
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
        hm = position_heatmap(traj, player=0)
        assert hm.shape == (32, 32)

    def test_position_heatmap_sums_to_one(self) -> None:
        """position_heatmap sums to 1.0."""
        traj = make_multi_player_traj(4, 20, num_p=2)
        hm = position_heatmap(traj, player=0)
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
        hm = position_heatmap(traj, player=0)
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
        hm = position_heatmap(traj, player=0)
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

    def test_plot_resource_depletion_raises_without_field(self) -> None:
        """plot_resource_depletion raises ValueError when block_resources is missing."""
        traj = make_minimal_traj(4, 20)
        with pytest.raises(ValueError):
            plot_resource_depletion(traj)

    def test_plot_resource_depletion_raises_without_block_resources_field(
        self,
    ) -> None:
        """Missing block_resources field raises ValueError."""
        traj = Trajectory(
            actions=np.zeros((4, 20), dtype=np.int32),
        )
        with pytest.raises(ValueError):
            plot_resource_depletion(traj)

    def test_plot_resource_depletion_with_block_resources_returns_fig_ax(
        self,
    ) -> None:
        """plot_resource_depletion returns (Figure, Axes) when field is set."""
        num_eps, num_steps = 4, 20
        height, width = 8, 8
        resources = np.random.randint(
            0, 50, size=(num_eps, num_steps, height, width)
        ).astype(np.float32)
        traj = Trajectory(
            actions=np.zeros((num_eps, num_steps), dtype=np.int32),
            block_resources=resources,
        )
        fig, ax = plot_resource_depletion(traj)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close("all")


# ---------------------------------------------------------------------------
# TestMilestones
# ---------------------------------------------------------------------------


def _recording(num_players: int = 1, steps: int = 6, episodes: int = 2) -> Trajectory:
    """Return a recording shaped the way a real rollout is shaped.

    Parameters
    ----------
    num_players :
        The length of the player axis of the inventory. The action
        array stays two-axis at 1, which is what makes a one-player
        recording report ``is_multi_player`` as ``False``.
    steps :
        How many timesteps to build.
    episodes :
        How many episodes to build.

    Returns
    -------
    Trajectory
        Actions of shape ``(B, T)`` for one player, or ``(B, T, P)``
        for several, and an inventory of shape ``(B, T, P, N)``. Every
        player holds 7 tin ore and nothing else, so one known item
        carries a non-zero count.
    """
    inv = np.zeros((episodes, steps, num_players, NUM_ITEM_TYPES), dtype=np.int32)
    inv[..., int(ItemType.TIN_ORE)] = 7
    if num_players == 1:
        actions = np.zeros((episodes, steps), dtype=np.int32)
    else:
        actions = np.zeros((episodes, steps, num_players), dtype=np.int32)
    return Trajectory(actions=actions, player_inventory=inv)


class TestInventoryOverTime:
    """Shape and values of the inventory reader."""

    def test_single_player_recording_drops_the_player_axis(self) -> None:
        """A ``(B, T, 1, N)`` inventory reduces to ``(T, N)``.

        This is the shape every one-player rollout produces. A result
        with three axes breaks :func:`plot_inventory`, which indexes
        the second axis by item.
        """
        traj = _recording(num_players=1, steps=6)
        assert traj.is_multi_player is False

        out = inventory_over_time(traj)

        assert out.shape == (6, NUM_ITEM_TYPES)

    def test_multi_player_recording_selects_the_named_player(self) -> None:
        """A ``(B, T, P, N)`` inventory reduces to ``(T, N)`` as well."""
        traj = _recording(num_players=3, steps=6)
        assert traj.is_multi_player is True

        out = inventory_over_time(traj, player=2)

        assert out.shape == (6, NUM_ITEM_TYPES)

    def test_three_axis_inventory_still_works(self) -> None:
        """An inventory built by hand with no player axis is accepted."""
        inv = np.zeros((2, 6, NUM_ITEM_TYPES), dtype=np.int32)
        inv[..., int(ItemType.TIN_ORE)] = 7
        traj = Trajectory(
            actions=np.zeros((2, 6), dtype=np.int32),
            player_inventory=inv,
        )

        out = inventory_over_time(traj)

        assert out.shape == (6, NUM_ITEM_TYPES)

    def test_counts_land_on_the_right_item(self) -> None:
        """The value of an item appears in the column of its id."""
        traj = _recording(num_players=1, steps=6)

        out = inventory_over_time(traj)

        assert out[0, int(ItemType.TIN_ORE)] == 7
        assert out[0, int(ItemType.IRON_ORE)] == 0


class TestPlotInventory:
    """The plot over a recording shaped like a real rollout."""

    def test_draws_a_single_player_recording(self) -> None:
        """A one-player rollout plots without raising.

        Before the reader handled the player axis, this call raised
        ``IndexError`` on the second item, because it indexed the
        player axis by item id.
        """
        traj = _recording(num_players=1, steps=6)

        fig, ax = plot_inventory(traj)

        assert len(ax.get_lines()) == NUM_ITEM_TYPES - 1  # EMPTY excluded
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_labels_the_line_of_each_item(self) -> None:
        """Each line carries the name of the item whose id it holds."""
        traj = _recording(num_players=1, steps=6)

        fig, ax = plot_inventory(traj)

        labels = [line.get_label() for line in ax.get_lines()]
        # EMPTY is skipped, so line i holds item id i + 1.
        assert labels[int(ItemType.TIN_ORE) - 1] == "TIN_ORE"
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestItemLabels:
    """``DEFAULT_ITEM_LABELS`` against the ``ItemType`` enum."""

    def test_covers_every_item(self) -> None:
        """The list holds one name for each item type."""
        assert len(DEFAULT_ITEM_LABELS) == NUM_ITEM_TYPES

    def test_entry_names_the_item_with_that_value(self) -> None:
        """Entry ``i`` is the name of the item whose value is ``i``.

        The hand-written list had ``IRON``, ``COPPER``, and ``MINER``
        at ids 2, 3, and 4, where the engine has ``IRON_ORE``,
        ``COPPER_ORE``, and ``TIN_ORE``. A default inventory plot
        therefore labelled the tin ore line "MINER".
        """
        for i in range(NUM_ITEM_TYPES):
            assert DEFAULT_ITEM_LABELS[i] == ItemType(i).name


class TestItemColors:
    """``DEFAULT_ITEM_COLORS`` against the label list."""

    def test_covers_every_item(self) -> None:
        """The list holds one color for each item type."""
        assert len(DEFAULT_ITEM_COLORS) == NUM_ITEM_TYPES

    def test_colors_are_distinct(self) -> None:
        """No two items share a color."""
        assert len(set(DEFAULT_ITEM_COLORS)) == len(DEFAULT_ITEM_COLORS)
