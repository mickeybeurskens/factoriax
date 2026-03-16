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
"""

from . import actions, milestones, multiagent, state
from .trajectory import Trajectory

__all__ = [
    "Trajectory",
    "actions",
    "milestones",
    "multiagent",
    "state",
]
