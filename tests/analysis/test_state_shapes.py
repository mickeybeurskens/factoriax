"""Tests that the state readers accept the shapes the engine produces.

``EnvState.player_inventory`` has shape ``(P, NUM_ITEM_TYPES)``, so a
recording stacked by
:func:`~factoriax.analysis.trajectory.states_to_trajectory` always
carries a player axis and comes out as ``(B, T, P, N)``. That holds for
one player as much as for several.

``Trajectory.is_multi_player`` counts players in the action array, not
in the inventory. A one-player recording therefore reports ``False``
while its inventory still has the player axis. A reader that branches
on ``is_multi_player`` skips the player index on exactly the recordings
that still need it, and returns one axis too many.

These tests use the four-axis shape throughout, because that is what a
rollout gives. A three-axis inventory is accepted as well, for
recordings built by hand.
"""

from __future__ import annotations

import numpy as np

from factoriax.analysis.state import inventory_over_time, plot_inventory
from factoriax.analysis.trajectory import Trajectory
from factoriax.engine.constants import NUM_ITEM_TYPES, ItemType


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
