"""Shared low-level helpers for the analysis sub-package."""

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
    """

    Parameters
    ----------
    ax: Axes | None :
        
    figsize: tuple[float :
        
    float] :
        

    Returns
    -------
    type
        

    """
    if ax is None:
        return plt.subplots(figsize=figsize)
    return ax.figure, ax  # type: ignore[return-value]


def resolve_player_actions(traj: Trajectory, player: int | None) -> np.ndarray:
    """Extract a (B, T) action array, selecting a player if multi-player.

    Parameters
    ----------
    traj: Trajectory :
        
    player: int | None :
        

    Returns
    -------

    """
    if traj.is_multi_player:
        if player is None:
            player = 0
        return traj.player(player).actions
    return traj.actions
