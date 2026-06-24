"""Shared low-level helpers for the analysis sub-package."""

from __future__ import annotations

import numpy as np

from .trajectory import Trajectory


def resolve_player_actions(traj: Trajectory, player: int | None) -> np.ndarray:
    """Extract a (B, T) action array, selecting a player if multi-player."""
    if traj.is_multi_player:
        if player is None:
            player = 0
        return traj.player(player).actions
    return traj.actions
