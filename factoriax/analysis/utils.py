"""Helpers that every plot in the analysis package shares.

:func:`resolve_ax` settles who owns the figure. :func:`resolve_player_actions`
settles which player a plot reads. Both exist so that each plot
function can accept the same two arguments and behave the same way.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .trajectory import Trajectory


def resolve_ax(
    ax: Axes | None,
    figsize: tuple[float, float],
) -> tuple[Figure, Axes]:
    """Return the axes to draw on, and the figure that owns it.

    A plot function calls this first. With ``ax`` of ``None`` it makes
    its own figure, and the caller must close that figure. With an
    ``ax`` from the caller it draws into the caller's layout, and the
    caller keeps ownership.

    Parameters
    ----------
    ax :
        The axes to draw on, or ``None`` to make a new figure.
    figsize :
        The width and height of the new figure, in inches. A caller
        that supplies ``ax`` owns the layout, so this value is then
        ignored. A resize would move every other subplot on that
        figure.

    Returns
    -------
    tuple
        ``(figure, axes)``. With ``ax`` of ``None``, both are new. With
        an ``ax`` from the caller, the figure is ``ax.figure`` and the
        axes is ``ax`` itself.
    """
    if ax is None:
        return plt.subplots(figsize=figsize)
    return ax.figure, ax  # type: ignore[return-value]


def resolve_player_actions(traj: Trajectory, player: int | None) -> np.ndarray:
    """Reduce a trajectory's actions to one array per episode and step.

    Recordings arrive in three shapes. A single-agent scenario gives
    ``(B, T)``. A multi-agent scenario gives ``(B, T, P)``. A single
    agent recorded by multi-agent code gives ``(B, T, 1)``. This last
    shape reports ``is_multi_player`` as ``False`` and still carries
    the player axis. All three return as ``(B, T)``.

    Parameters
    ----------
    traj :
        The recording to read. The function changes nothing. Where the
        shape allows it, the result is a view into ``traj.actions``.
    player :
        The player to keep. ``None`` means player 0. A recording with
        no player axis ignores this value, because it holds the
        actions of one player only.

    Returns
    -------
    numpy.ndarray
        Shape ``(B, T)``, holding action ids.

    Raises
    ------
    IndexError
        When ``player`` names a slot the player axis does not have.
    """
    if player is None:
        player = 0
    if traj.is_multi_player:
        return traj.player(player).actions
    actions = traj.actions
    if actions.ndim == 3:
        return actions[:, :, player]
    return actions
