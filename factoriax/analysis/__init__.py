"""FactoriaX Analysis — trajectory analysis and visualization toolkit.

Quick start
-----------
>>> from factoriax.analysis import Trajectory
>>> traj = Trajectory.load("rollout.npz")
>>> from factoriax.analysis import actions
>>> fig, ax = actions.action_raster(traj)

Submodules
----------
- ``trajectory`` — Trajectory container with I/O and slicing
- ``actions`` — Action sequence patterns (raster, transitions, n-grams, entropy)
- ``multiagent`` — Multi-player coordination (role divergence, spatial overlap)
- ``state`` — State evolution (inventory, movement heatmaps)
- ``milestones`` — Achievement timing and first-action analysis
- ``video`` — Frame composition + MP4 encoding for episode rollouts
- ``eval`` — :class:`EvalRollout` container and summary plots
"""

from . import actions, eval, milestones, multiagent, state, video
from .trajectory import Trajectory

__all__ = [
    "Trajectory",
    "actions",
    "eval",
    "milestones",
    "multiagent",
    "state",
    "video",
]
