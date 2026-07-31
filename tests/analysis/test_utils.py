"""Tests for :mod:`factoriax.analysis.utils`.

Every plot in the package uses these two helpers.

:func:`resolve_ax` decides whether a plot makes its own figure or draws
into a figure that the caller owns. The two cases differ in who must
close the figure.

:func:`resolve_player_actions` reduces the action array of a trajectory
to two axes, batch and time. It selects one player when there are
several. Recordings arrive in three shapes:

* ``(B, T)`` from a single-agent scenario.
* ``(B, T, P)``, with ``P`` more than 1, from a multi-agent scenario.
* ``(B, T, 1)`` from a single agent recorded by multi-agent code.

All three must return as ``(B, T)``.

The package ``conftest`` selects the Agg backend before these modules
import ``pyplot``. ``plt.subplots`` therefore never asks for a window.
Every test closes the figures that it makes. An open figure stays in
the ``pyplot`` registry for the rest of the session. At twenty open
figures, ``pyplot`` gives a warning.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from factoriax.analysis.trajectory import Trajectory
from factoriax.analysis.utils import resolve_ax, resolve_player_actions


class TestResolveAx:
    """Figure ownership for the two call forms."""

    def test_none_creates_new_figure(self) -> None:
        """When ``ax`` is ``None``, make a new figure and axes."""
        fig, ax = resolve_ax(None, figsize=(6, 4))
        assert fig is not None
        assert ax is not None
        assert ax.figure is fig
        plt.close(fig)

    def test_none_respects_figsize(self) -> None:
        """Give the new figure the requested size, in inches."""
        fig, ax = resolve_ax(None, figsize=(8, 3))
        w, h = fig.get_size_inches()
        assert (w, h) == (8.0, 3.0)
        plt.close(fig)

    def test_existing_ax_returns_its_figure(self) -> None:
        """Return a caller axes with the figure that owns it."""
        existing_fig, existing_ax = plt.subplots(figsize=(5, 5))
        fig, ax = resolve_ax(existing_ax, figsize=(99, 99))
        assert fig is existing_fig
        assert ax is existing_ax
        plt.close(existing_fig)

    def test_existing_ax_ignores_figsize(self) -> None:
        """When the caller supplies an axes, ignore ``figsize``.

        The caller owns the layout in this case. A resize from inside a
        plot helper moves every other subplot on that figure.
        """
        existing_fig, existing_ax = plt.subplots(figsize=(5, 5))
        resolve_ax(existing_ax, figsize=(99, 99))
        w, h = existing_fig.get_size_inches()
        assert (w, h) == (5.0, 5.0)
        plt.close(existing_fig)


class TestResolvePlayerActions:
    """Reduction of the action array to ``(B, T)``."""

    def test_two_axis_actions_pass_through(self) -> None:
        """Return a two-axis array without a change."""
        traj = Trajectory(actions=np.arange(6, dtype=np.int32).reshape(2, 3))
        out = resolve_player_actions(traj, player=None)
        np.testing.assert_array_equal(out, [[0, 1, 2], [3, 4, 5]])

    def test_multi_player_defaults_to_player_zero(self) -> None:
        """When the caller names no player, read player 0."""
        actions = np.array([[[1, 9], [2, 9], [3, 9]]], dtype=np.int32)
        out = resolve_player_actions(Trajectory(actions=actions), player=None)
        assert out.shape == (1, 3)
        np.testing.assert_array_equal(out, [[1, 2, 3]])

    def test_multi_player_selects_the_named_player(self) -> None:
        """Read the named player from the player axis."""
        actions = np.array([[[1, 9], [2, 9], [3, 9]]], dtype=np.int32)
        out = resolve_player_actions(Trajectory(actions=actions), player=1)
        np.testing.assert_array_equal(out, [[9, 9, 9]])

    def test_single_player_kept_in_three_axis_form(self) -> None:
        """Reduce a ``(B, T, 1)`` recording to ``(B, T)`` as well.

        ``Trajectory.is_multi_player`` counts players, not axes. A
        recording of one agent by multi-agent code therefore reports
        ``False`` and still carries a player axis of length 1. Returned
        without a change, it gives callers three axes where they unpack
        two.
        """
        actions = np.array([[[1], [2], [3]]], dtype=np.int32)
        traj = Trajectory(actions=actions)
        assert not traj.is_multi_player

        out = resolve_player_actions(traj, player=None)

        assert out.shape == (1, 3)
        np.testing.assert_array_equal(out, [[1, 2, 3]])

    def test_player_out_of_range_raises(self) -> None:
        """When the player axis has no such player, raise ``IndexError``."""
        traj = Trajectory(actions=np.array([[[1], [2]]], dtype=np.int32))
        with pytest.raises(IndexError):
            resolve_player_actions(traj, player=1)
